"""qwen3_5_moe architecture plugin (Qwen3.5/3.6 MoE, hybrid GDN + attention).

Consumes the public HF export directly (no conversion artifact):
  * routed experts: `...mlp.experts.{gate_up_proj,down_proj}.escha_{code,rin,rout}`
    (E-stacked; `escha_s_in`/`escha_s_out` are all-ones and `escha_config` is
    redundant — both dropped, like every other escha runtime).
  * dense linears / embed / lm_head: `weight_int8` + `weight_scale` pairs ->
    exact MLX affine-Q8 repack (escha_mlx.quant).
  * everything else fp16 -> mlx-lm's own sanitize (language_model renames,
    conv1d layout, the (1+w) norm shift) + update.

The model skeleton, GDN kernels, attention and KV/state caches are mlx-lm's;
only the MoE block and the quantized dense modules are replaced. Post-load
quirks: GDN recurrent-state dtype cache (escha_mlx.gdn_cache) and the
last-position LM head (escha_mlx.loader.LastPositionHead).

Routing convention (the eschamoe serving convention): fp32 logits from the
fp16 gate Linear (one f16 round), top-k, softmax over the top-k values
(== softmax-over-E + renorm). Each (token, slot) pair becomes one GEMV row:
row_expert[m] selects the expert stream, so the whole MoE path stays
device-resident — no host synchronization per step.

Known benign edge case: when f16-rounded router logits tie exactly at the k
boundary, argpartition may pick either of the equally-scored experts — a
rare one-expert set difference between runtimes, not a decode bug.

Expert forward per row (see escha_mlx.ref for the rounding contract):
    xh   = f16( H128(x * rin[e]) * RS )
    mid  = xh @ decode(code[e])                       (f32, fused Metal kernel)
    gu16 = f16( H128(mid) * RS * rout[e] )            -> silu(g)*u -> h
    xh2  = f16( H128(h * rin_dn[e]) * RS )
    d16  = f16( H128(xh2 @ decode_dn) * RS * rout_dn[e] )
    out[token] += f32(d16) * f32(f16(score))          (f32 accumulate)
plus the shared expert (int8-Q8 SwiGLU, sigmoid-gated).

Paths:
  * fused (default on Metal)  — escha_mlx.msl kernels.
  * ops   (ESCHA_MLX_MOE=ops or no Metal) — numpy tile decode + mx matmul.
    Slow; exists so the full model runs (and is testable) on any backend.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten

from .. import gdn_cache, moe, msl, quant, ref
from ..loader import LastPositionHead, resolve_module, strip_lm_prefix, use_last_logit

from mlx_lm.models.qwen3_next import Qwen3NextAttention, scaled_dot_product_attention


class FusedQwen3NextAttention(nn.Module):
    """q/k/v projections as ONE affine-Q8 call.

    ``q_proj`` / ``k_proj`` / ``v_proj`` are three EschaQ8Linear matmuls of the
    same input.  Affine-Q8 rows are independently dequantized, so stacking the
    three stacked weights along the OUTPUT axis and calling one
    ``qkv_proj`` produces bit-identical rows to the three separate matmuls
    (the same free-regroup the shared expert uses).  This trades three
    launches + payload reads per attention layer for one.

    Kept as a re-classed instance (``__dict__.update`` of the original
    attention) so rope / q_norm / k_norm / o_proj / scale are preserved; only
    ``__call__`` differs, splitting the fused output back into q/k/v.
    """
    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, D = x.shape
        nq = self.num_attention_heads * self.head_dim
        nk = self.num_key_value_heads * self.head_dim
        qkv = self.qkv_proj(x)                       # [B, L, 2*nq + 2*nk]
        qg = qkv[..., :2 * nq]
        queries, gate = mx.split(
            qg.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        keys = qkv[..., 2 * nq:2 * nq + nk]
        values = qkv[..., 2 * nq + nk:2 * nq + 2 * nk]

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1))\
            .transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1)\
            .transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))


class FusedGatedDeltaNet(nn.Module):
    """GDN attention with the two tiny in-projections (b, a) stacked.

    The GDN layer runs ``in_proj_b`` and ``in_proj_a`` on the SAME input; both
    are [32, hidden] fp16 linears feeding the delta gate.  Stacking them along
    the OUTPUT axis into one [64, hidden] fp16 Linear is bit-identical (no
    re-rounding; pure view split on the output).  One launch + one payload
    read instead of two.  Verified bit-exact on the full-model route.

    The Q8 qkv+z stack was ALSO tried and is NOT shipped: standalone it matched
    on every sampled layer, but the real route diverged (wave3's documented
    output-axis-stacking wall — a different matmul tiling for the wider output
    changes the K-reduction order).  b+a stays safe because fp16 matmul over
    the same K=2048 does not change tiling between 32 and 64 output columns on
    this MLX build; the full-model check passes.

    Kept as a re-classed instance so conv1d / dt_bias / A_log / norm /
    out_proj / sharding_group are preserved; only ``__call__`` differs."""
    def __call__(self, inputs, mask=None, cache=None):
        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        ba = self.in_proj_ba(inputs)                 # [B, S, 2*num_v_heads]
        b = ba[..., : self.num_v_heads]
        a = ba[..., self.num_v_heads:]

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        from mlx_lm.models.qwen3_5 import gated_delta_update
        out, state = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
            use_kernel=not self.training)

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))
        if self.sharding_group is not None:
            from mlx import distributed
            out = distributed.all_sum(out, group=self.sharding_group)
        return out


def use_fuse_gdn() -> bool:
    """Stack the GDN layer's tiny b/a in-projections into one fp16 linear
    (ESCHA_MLX_FUSE_GDN=0 disables).  Bit-identical (verified on the full-model
    route); one launch + one payload read per GDN layer instead of two."""
    return os.environ.get("ESCHA_MLX_FUSE_GDN", "1") != "0"


def fuse_gdn_layers(model, n_layers: int) -> None:
    """Re-class each GDN ``linear_attn`` as FusedGatedDeltaNet with the stacked
    b/a in-projection, leaving qkv/z and everything else intact."""
    layers = model.language_model.model.layers
    for i in range(n_layers):
        l = layers[i]
        orig = getattr(l, "linear_attn", None)
        if orig is None:
            continue
        fused = FusedGatedDeltaNet.__new__(FusedGatedDeltaNet)
        fused._training = orig._training
        for name in ("hidden_size", "num_v_heads", "num_k_heads", "head_k_dim",
                     "head_v_dim", "key_dim", "value_dim", "conv_kernel_size",
                     "layer_norm_epsilon", "conv_dim", "conv1d", "dt_bias",
                     "A_log", "norm", "out_proj", "sharding_group",
                     "in_proj_qkv", "in_proj_z"):
            setattr(fused, name, getattr(orig, name))
        fused.in_proj_ba = nn.Linear(orig.hidden_size, 2 * orig.num_v_heads,
                                     bias=False)
        fused.in_proj_ba.weight = mx.concatenate(
            [orig.in_proj_b.weight, orig.in_proj_a.weight], axis=0)
        l.linear_attn = fused


def use_fuse_attn() -> bool:
    """Fuse attention q/k/v into one affine-Q8 call (ESCHA_MLX_FUSE_ATTN=0
    disables).  Bit-identical (affine-Q8 rows dequantize independently, so
    stacking along the output axis changes no output element).  One launch
    + one payload read per attention layer instead of three.  Measured neutral
    on a clean 40-core GPU (14.66 vs 14.67 ms decode step) but removes launches
    and one payload read per step.  Default OFF: stacking q/k/v along the output
    axis is only valid when num_key_value_heads == num_attention_heads (no GQA);
    it regresses GQA/synthetic smaller-kv configs (tests/test_models.py).  Keep
    the opt-in launch reduction available for MPA checkpoints."""
    return os.environ.get("ESCHA_MLX_FUSE_ATTN", "0") == "1"


def fuse_attention_layers(model, n_layers: int) -> None:
    """Re-class each attention `self_attn` as FusedQwen3NextAttention with a
    stacked qkv_proj (one Q8 linear), leaving everything else intact."""
    layers = model.language_model.model.layers
    for i in range(n_layers):
        l = layers[i]
        if getattr(l, "self_attn", None) is None:
            continue
        orig = l.self_attn
        q, k, v = orig.q_proj, orig.k_proj, orig.v_proj
        fused = FusedQwen3NextAttention.__new__(FusedQwen3NextAttention)
        # copy the state the fused forward touches (assigning each via setattr
        # registers submodules in the new instance's own _modules)
        for name in ("num_attention_heads", "num_key_value_heads", "head_dim",
                     "scale", "q_norm", "k_norm", "o_proj", "rope"):
            setattr(fused, name, getattr(orig, name))
        fused.qkv_proj = quant.EschaQ8Linear(
            mx.concatenate([q.weight, k.weight, v.weight], axis=0),
            mx.concatenate([q.scales, k.scales, v.scales], axis=0),
            mx.concatenate([q.biases, k.biases, v.biases], axis=0),
            q._group_size,
        )
        l.self_attn = fused


logger = logging.getLogger(__name__)

MODEL_TYPE = "qwen3_5_moe"

RS = ref.RS

# Rows-per-group for the m >= 2048 (prefill) band, per MEASURED machine.
# Keys are mx.device_info()['device_name']; machines without a datapoint keep
# the 10-core M4 policy (R=12, doc §13.1).  The 40-core M4 Max prefers R=8:
# at m=2048 that equals m/E -- every expert's ~8 rows fill exactly one group,
# so there is zero padding; R=12's padding waste (405 groups vs 256) is what
# the M4's bandwidth could swallow but this chip cannot.  In-model, warmed,
# chunk S=256 (m=2048): R=8 1162 tok/s vs R=12 1110 (+4.7%); S=512 (m=4096):
# 1258 vs 1174 (+7.2%).  bench/results/m4-max-64gb, 2026-08-11.
_PREFILL_R = {"Apple M4 Max": 8}


def _device_name() -> str:
    return str(mx.device_info().get("device_name", ""))

_DROP_LEAVES = {"escha_s_in", "escha_s_out", "escha_config"}


@lru_cache(maxsize=None)
def _arange32(n: int) -> mx.array:
    """mx.arange((n,), int32): the same value is rebuilt every layer and every
    step (row_token and the dn-leg identity index), so cache it -- mx arrays
    are immutable, so reusing a constant is bit-identical and drops the
    per-step arange/reshape ops from the lazy graph."""
    return mx.arange(n, dtype=mx.int32)


def _row_token(t: int, top_k: int) -> mx.array:
    """row_token = repeat(arange(t), top_k): depends ONLY on t and top_k, so
    the same value is rebuilt every layer every decode step.  mx arrays are
    immutable, so a module-global cache keyed by t reuses the SAME array
    (bit-identical) and drops the per-step arange/repeat ops."""
    return mx.repeat(_arange32(t), top_k)


class EschaSparseMoeBlock(nn.Module):
    """Drop-in replacement for the mlx-lm Qwen3.5/3.6 MoE block, executing the
    escha trellis-coded routed experts (see the module docstring)."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int,
                 gu: moe.EschaExperts, dn: moe.EschaExperts,
                 gate_w: np.ndarray, shg_w: np.ndarray,
                 shared: dict[str, np.ndarray],
                 group_size: int = quant.DEFAULT_GROUP) -> None:
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self._gu = gu
        self._dn = dn
        self._inter = gu.OC // 2

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.gate.weight = mx.array(gate_w.astype(np.float16))
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
        self.shared_expert_gate.weight = mx.array(shg_w.astype(np.float16))
        # shared expert gate+up: ONE affine-Q8 linear of [2*I, hidden] instead
        # of two [I, hidden] calls -- per-row affine dequant makes the stacked
        # matmul bit-identical to the separate ones (omlx#2238's free regroup;
        # same product shape as the routed experts' gate_up concatenation).
        # One payload read + one launch per layer instead of two.
        gu_w8 = np.concatenate([shared["gate_w8"], shared["up_w8"]], axis=0)
        gu_scale = np.concatenate([shared["gate_scale"], shared["up_scale"]], axis=0)
        self._sh_inter = gu_w8.shape[0] // 2
        self.sh_gu = quant.make_linear(gu_w8, gu_scale, group_size)
        # separate gate/up linears for the mid-S branch: the stacked sh_gu is
        # bit-exact only at m<=16 or m>=1024 rows (split-K shape dependence;
        # measured boundary: diverges at 36..512, bit-exact at <=32 and >=528).
        # At m in [36,512] the separate matmuls are bit-identical by
        # construction, so we branch on the row count in __call__.
        self.sh_gate = quant.make_linear(shared["gate_w8"], shared["gate_scale"], group_size)
        self.sh_up = quant.make_linear(shared["up_w8"], shared["up_scale"], group_size)
        self.sh_down = quant.make_linear(shared["down_w8"], shared["down_scale"], group_size)

        self._mode = os.environ.get(
            "ESCHA_MLX_MOE", "fused" if mx.metal.is_available() else "ops")
        if self._mode not in ("fused", "ops"):
            raise ValueError(f"ESCHA_MLX_MOE must be 'fused' or 'ops', got {self._mode!r}")
        # ESCHA_MLX_BLOCK_R pins rows-per-group (1 = always the per-row kernel);
        # unset = the size-dependent policy in _blocked_R.
        self._fused_had = msl.use_fused_had() and self._mode == "fused"
        # gemv_had fuses the input Hadamard into the GEMV; only safe when both
        # the gate_up (IC) and down (dn.IC) dimensions are 128-block multiples.
        self._gemv_had = (msl.use_gemv_had() and self._mode == "fused"
                          and dn.IC % 256 == 0)
        self._dn_sum = msl.use_dn_sum() and self._mode == "fused" and self._gemv_had
        self._silu_tail = msl.use_silu_tail() and self._mode == "fused"
        _br = os.environ.get("ESCHA_MLX_BLOCK_R")
        self._block_env = int(_br) if _br else None
        # Read once at construction: flipping it per-forward would defeat the
        # compile cache and make A/Bs depend on call order.

    # -- expert path ------------------------------------------------------

    def _rows(self, xf: mx.array, ids: mx.array):
        t = xf.shape[0]
        row_expert = ids.reshape(t * self.top_k).astype(mx.int32)
        row_token = _row_token(t, self.top_k)
        return row_expert, row_token

    def _scaled_had(self, rows: mx.array, row_expert: mx.array,
                    ex: moe.EschaExperts) -> mx.array:
        """f16( H128(rows * rin[e]) * RS ) — the input transform for either leg."""
        if self._fused_had and rows.dtype == mx.float16:
            # One kernel instead of ~5 ops each materialising [m, IC] f32.
            return msl.scaled_had(rows, ex.rin, row_expert, RS)
        xr = rows.astype(mx.float32) * ex.rin[row_expert]
        return (moe.had_blocks(xr) * RS).astype(mx.float16)

    def _input_rows(self, xf: mx.array, row_token: mx.array, row_expert: mx.array,
                    ex: moe.EschaExperts) -> mx.array:
        return self._scaled_had(xf[row_token], row_expert, ex)

    def _output_rows(self, mid: mx.array, row_expert: mx.array,
                     ex: moe.EschaExperts) -> mx.array:
        if self._fused_had and mid.dtype == mx.float32:
            # Keep the transform, output scale gather and f16 cast in one kernel.
            return msl.scaled_had_out(mid, ex.rout, row_expert, RS)
        y = moe.had_blocks(mid) * RS * ex.rout[row_expert]
        return y.astype(mx.float16)

    def _blocked_R(self, m: int) -> int:
        """Rows per group, or 1 to use the plain per-row decode kernel.

        Row-blocking trades STREAM BYTES for ROW WORK: grouping cuts decoded-
        stream reads to (distinct experts)/m but pads every partial group up to
        R, so row work rises to R*(groups)/m.  With m = 8B rows drawn from
        E=256, expected distinct experts is E*(1-(1-1/E)^m) -- 1.13 rows/expert
        at m=64, 1.27 at 128, 1.58 at 256, 1.93 at 384 -- so the useful R grows
        slowly with m and large R is actively harmful at decode row counts.

        Thresholds below m=1024 are measured IN-MODEL (median of 3, whole decode
        step, bench/sweep_block_r.py), not from a kernel microbenchmark:

            m=  64   R=1 best   (R=2 -3.5%)
            m= 128   R=2 +5.9%  (R=3 -3.8%)
            m= 192   R=2 +3.7%  (R=3 +1.9%)
            m= 256   R=2 +13.1% (R=3 +13.1%, R=4 -5.8%)
            m= 384   R=3 +17.7% (R=2 +14.7%)

        Note m=256: the old policy chose R=4 here, which is the WORST of the
        four (-5.8% vs R=1, i.e. 8.4% below R=2).  That threshold came from an
        isolated-kernel table where R=4 at M=256 measured 1.28x; in-model the
        build_groups + padding cost eats it.  Kernel microbenchmarks do not
        settle this -- the whole-step number does.

        Prefill bands, re-measured properly (doc §13.1) after the first attempt
        was swamped by cold-kernel time.  Whole-chunk forward, warmed per shape:

            S=256 (m=2048, ~8 rows/expert)   R=12 +3.8%, R=8 +2.6%, R=4 -0.8%
            S=512 (m=4096, ~16 rows/expert)  R=12 +1.8%, R=8 -0.6%, R=4 -5.3%

        R=12 wins at both, but only by a few percent -- NOT the ~2x that the
        padding arithmetic predicts (at m=2048 a group of 16 is only half full,
        so R=16 issues twice the MACs).  That the MAC count barely matters is
        the evidence that this kernel is not MAC-bound at prefill either.

        M4 Max / 40-core (bench/results/m4-max-64gb, 2026-08-11): the m>=2048
        band flips to R=8 via _PREFILL_R (S=256 +4.7%, S=512 +7.2% vs R=12, in
        separate processes so the module-construction cache is clean).  Lower
        bands measured identical to the shipped policy: m=1024 R=4 best (988
        tok/s vs 959 at R=8), m=512 R=3/R=4 tie (~797), R=2 -6%.
        """
        if self._block_env is not None:
            return self._block_env
        if m >= 2048:
            return _PREFILL_R.get(_device_name(), 12)
        if m >= 1024:       # untested band, inherited
            return 4
        if m >= 320:
            return 3
        if m >= 128:
            return 2
        return 1

    def _gemv(self, xh: mx.array, row_expert: mx.array, ex: moe.EschaExperts,
              groups=None) -> mx.array:
        if self._mode == "fused":
            if groups is not None:
                rows_idx, group_expert, r, order, src_row0, n_valid = groups
                m = xh.shape[0]
                if order is not None:
                    # xs[p] is the p-th row in expert order, so a group's rows
                    # are consecutive and the kernel computes their addresses.
                    # No padding sink row is needed: staging masks on n_valid.
                    mid = msl.moe_gemm_rows(xh[order], ex.code, rows_idx,
                                            group_expert, ex.K, ex.IC, ex.OC,
                                            r, m, sort_idx=(src_row0, n_valid))
                else:
                    xh_pad = mx.concatenate(
                        [xh, mx.zeros((1, xh.shape[1]), dtype=xh.dtype)], axis=0)
                    mid = msl.moe_gemm_rows(xh_pad, ex.code, rows_idx,
                                            group_expert, ex.K, ex.IC, ex.OC, r, m)
                return mid[:m]
            return msl.moe_gemv(xh, ex.code, row_expert, ex.K, ex.IC, ex.OC)
        # ops path: numpy decode per row (test/CPU only — slow)
        code_np = ex.code_numpy()
        re = np.array(row_expert)
        xh_np = np.array(xh)
        out = np.empty((xh.shape[0], ex.OC), dtype=np.float32)
        cache: dict[int, np.ndarray] = {}
        for r, e in enumerate(re):
            e = int(e)
            if e not in cache:
                cache[e] = ref.reconstruct_fast(
                    code_np[e].view(np.uint16).view(np.int16), ex.IC, ex.OC, ex.K
                ).astype(np.float32)
            out[r] = xh_np[r].astype(np.float32) @ cache[e]
        return mx.array(out)

    def _foot_fused(self, xh, row_expert, ex, groups):
        """GEMV + output transform for one leg."""
        mid = self._gemv(xh, row_expert, ex, groups)
        return self._output_rows(mid, row_expert, ex)

    def _gate_silu(self, mid: mx.array, row_expert: mx.array):
        """gu-leg epilogue: output-Hadamard then the silu gate, fused.

        With ESCHA_MLX_SILU_TAIL the transform, scale, cast and the silu/mul
        tail run in ONE kernel returning (s16, h); otherwise the native
        scaled_had_out -> silu chain.  Both yield s16=f16(g*sigmoid(g)) and
        h=s16*gu16[inter:], bit-identical."""
        if self._silu_tail and mid.dtype == mx.float32:
            return msl.scaled_had_out_silu(mid, self._gu.rout, row_expert,
                                           self._gu.OC, ref.RS)
        gu16 = self._output_rows(mid, row_expert, self._gu)
        g = gu16[:, :self._inter].astype(mx.float32)
        s16 = (g * mx.sigmoid(g)).astype(mx.float16)
        h = s16 * gu16[:, self._inter:]
        return s16, h

    def _expert_path(self, xf: mx.array, ids: mx.array, scores: mx.array) -> mx.array:
        t = xf.shape[0]
        row_expert, row_token = self._rows(xf, ids)
        m = row_expert.shape[0]
        r = self._blocked_R(m)
        groups = None
        if r > 1:
            ng = moe.n_groups_bound(m, self.num_experts, r)
            sx = msl.use_sortx()
            g = moe.build_groups(row_expert, self.num_experts, r, ng, with_sorted=sx)
            groups = (g[0], g[1], r) + (tuple(g[2:]) if sx else (None, None, None))
        if self._gemv_had and groups is None:
            # fuse the input Hadamard transform into the per-leg GEMV: one
            # kernel instead of [scaled_had; moe_gemv], bit-identical (the
            # transform runs the same butterfly in threadgroup memory; the
            # GEMV accumulation order is unchanged).  The xf[row_token]
            # gather is folded in too (row_token indexes xf inside the
            # kernel), deleting a per-layer [m, IC] copy.
            mid = msl.moe_gemv_had(xf, self._gu.rin, self._gu.code,
                                   row_expert, row_token, self._gu.K,
                                   self._gu.IC, self._gu.OC, ref.RS)
            s16, h = self._gate_silu(mid, row_expert)
            # fuse the down-leg input Hadamard into its GEMV too (same butter-
            # fly, same accumulation order => bit-identical to scaled_had+gemv).
            # h is already per-row (not xf), so index by row identity.
            mid2 = msl.moe_gemv_had(h, self._dn.rin, self._dn.code,
                                    row_expert, _arange32(m),
                                    self._dn.K, self._dn.IC, self._dn.OC, ref.RS)
            if not self._dn_sum:
                # Without dn_sum the [m, OC] f32 output-Hadamard is needed
                # downstream; with dn_sum it is DEAD (scaled_had_out_sum
                # below fuses the transform into the per-token sum), so skip
                # the kernel entirely -- 40 launches/step that never touched
                # the result.  Bit-identical: the value was unused.
                d16 = self._output_rows(mid2, row_expert, self._dn)
        else:
            xh = self._input_rows(xf, row_token, row_expert, self._gu)
            mid = self._gemv(xh, row_expert, self._gu, groups)
            s16, h = self._gate_silu(mid, row_expert)
            xh2 = self._scaled_had(h, row_expert, self._dn)
            if self._dn_sum:
                # moe_gemm_rows scatters its output back to the ORIGINAL
                # (token-major) row index via rows_idx, so the row-blocked
                # prefill leg meets scaled_had_out_sum's contiguous
                # token-major invariant exactly like the decode leg.  Fusing
                # the output-Hadamard + f16 round + score product + per-token
                # sum into one kernel is bit-identical to the old chain
                # [scaled_had_out; d16.astype(f32); w16.astype(f32); mul;
                # reshape; sum; f16->f32 round] and deletes per layer one
                # launch and both [m, OC] f32 d16/contrib intermediates.
                mid2 = self._gemv(xh2, row_expert, self._dn, groups)
                y = msl.scaled_had_out_sum(
                    mid2, self._dn.rout, row_expert,
                    scores.reshape(t * self.top_k), self.top_k, ref.RS)
                return y.astype(mx.float32)
            d16 = self._foot_fused(xh2, row_expert, self._dn, groups)
        if self._dn_sum and groups is None:
            # Fuse the down-leg output-Hadamard + score-weighted per-token sum
            # into one kernel (bit-identical to scaled_had_out -> mul -> sum).
            # Combines the f16 round-trip too (y is f16, so the astype(f32)
            # below is the only cast).  Rows are token-major by construction.
            # Pass the f32 scores straight to the kernel: it rounds each
            # through f16 internally (same RNE bits as the old .astype(f16))
            # so the per-layer f16 cast kernel + node are dropped entirely.
            y = msl.scaled_had_out_sum(
                mid2, self._dn.rout, row_expert,
                scores.reshape(t * self.top_k), self.top_k, ref.RS)
            return y.astype(mx.float32)
        w16 = scores.reshape(t * self.top_k).astype(mx.float16)
        contrib = d16.astype(mx.float32) * w16.astype(mx.float32)[:, None]

        # Segmented sum, NOT a scatter-add.  `_rows` lays rows out token-major
        # (row_token = repeat(arange(t), top_k)), so the top_k contributions of
        # token i are exactly rows [i*top_k, (i+1)*top_k) -- contiguous -- and
        # reshape+sum adds precisely the same addends as
        # `mx.zeros(...).at[row_token].add(contrib)` did.
        #
        # Why it replaced the scatter: `.at[].add()` on f32 with top_k=8
        # DUPLICATE indices per token is an ATOMIC accumulation, so the
        # summation order varied per run.  Over 40 layers that was enough to
        # flip greedy argmax on near-ties -- decoding the same prompt twice
        # gave different tokens (measured).  A fixed-order reduction makes the
        # whole decode bit-reproducible, which is a hard requirement here: with
        # nondeterministic output you cannot A/B a kernel by diffing text, and
        # every eval number carries an invisible run-to-run term.
        #
        # This changes the summation ORDER versus the old scatter (f32 addition
        # is not associative), so it is not bit-identical to pre-fix output;
        # it is within f32 rounding of it and gated both ways in
        # tests/test_moe_determinism.py.  The token-major invariant above is
        # load-bearing -- it is pinned by test_rows_layout_is_token_major.
        y = contrib.reshape(t, self.top_k, -1).sum(axis=1)
        # The forward contract rounds the routed-expert sum to f16 before
        # adding the shared expert — match that rounding point.
        return y.astype(mx.float16).astype(mx.float32)

    # -- block forward ----------------------------------------------------

    def __call__(self, x: mx.array) -> mx.array:
        b, s, hdim = x.shape
        xf = x.reshape(-1, hdim)
        logits = self.gate(xf).astype(mx.float32)
        ids = mx.argpartition(logits, kth=-self.top_k, axis=-1)[..., -self.top_k:]
        top = mx.take_along_axis(logits, ids, axis=-1)
        scores = mx.softmax(top, axis=-1, precise=True)

        y = self._expert_path(xf, ids, scores)

        # shared expert, gate+up.  Stacked sh_gu (one launch, per-row affine
        # dequant) is bit-exact only at m<=16 or m>=320 rows; at mid row
        # counts the split-K shape dependence diverges from the separate
        # matmuls, so fall back to separate gate/up there (bit-identical by
        # construction).  Decode (m=1) keeps the launch-reduction win.
        m = xf.shape[0]
        if m <= 16 or m >= 1024:
            gu = self.sh_gu(xf).astype(mx.float32)      # [..., 2*I]
        else:
            g = self.sh_gate(xf).astype(mx.float32)
            u = self.sh_up(xf).astype(mx.float32)
            gu = mx.concatenate([g, u], axis=1)
        g = gu[:, :self._sh_inter]
        u = gu[:, self._sh_inter:]
        hh = ((g * mx.sigmoid(g)) * u).astype(mx.float16)
        sh = self.sh_down(hh).astype(mx.float32)
        sgate = mx.sigmoid(self.shared_expert_gate(xf).astype(mx.float32))
        out = (y + sh * sgate).astype(x.dtype)
        return out.reshape(b, s, hdim)


class CheckpointLoader:
    """Streaming consumer for eschamoe qwen3_5_moe exports (contract:
    escha_mlx/models/__init__.py)."""

    def __init__(self, config: dict, group_size: int) -> None:
        from mlx_lm.models import qwen3_5_moe as skel

        self.model = skel.Model(skel.ModelArgs.from_dict(config))
        self.group_size = group_size
        text_args = self.model.language_model.args
        self.n_layers = text_args.num_hidden_layers
        self.top_k = text_args.num_experts_per_tok
        self.hidden_size = text_args.hidden_size
        self.num_experts = text_args.num_experts
        self.layers = self.model.language_model.model.layers

        # Streaming single pass: every tensor is converted to its final (mx)
        # form as soon as its dependency group completes, then the numpy copy
        # is freed.
        self._experts_np: dict[tuple[int, str], dict[str, np.ndarray]] = {}
        self._experts_mx: dict[tuple[int, str], moe.EschaExperts] = {}
        self._int8_np: dict[str, dict[str, np.ndarray]] = {}
        self._shared_np: dict[str, dict[str, np.ndarray]] = {}   # held to block build
        self._mlp_fp16: dict[tuple[int, str], np.ndarray] = {}
        self._base: dict[str, np.ndarray] = {}
        self.n_q8 = 0
        self.dropped = 0

    def _install_q8(self, base_name: str, pair: dict[str, np.ndarray]) -> None:
        w8, scale = pair["weight_int8"], pair["weight_scale"]
        if base_name == "lm_head":
            self.model.language_model.lm_head = quant.make_linear(w8, scale, self.group_size)
        elif base_name == "embed_tokens":
            self.model.language_model.model.embed_tokens = quant.make_embedding(
                w8, scale, self.group_size)
        elif base_name.startswith("layers."):
            rest = base_name[len("layers."):]
            idx, dotted = rest.split(".", 1)
            parent, attr = resolve_module(self.layers[int(idx)], dotted)
            setattr(parent, attr, quant.make_linear(w8, scale, self.group_size))
        else:
            raise ValueError(f"unexpected int8 tensor: {base_name}")
        self.n_q8 += 1

    def consume(self, name: str, w: np.ndarray) -> None:
        if name.startswith("mtp.") or ".visual." in name or name.startswith("visual."):
            self.dropped += 1
            return
        s = strip_lm_prefix(name)
        parts = s.split(".")
        if ".mlp.experts." in s:
            layer = int(parts[1])
            proj, leaf = parts[4], parts[5]
            if leaf in _DROP_LEAVES:
                self.dropped += 1
                return
            group = self._experts_np.setdefault((layer, proj), {})
            group[leaf] = w
            if len(group) == 3:
                self._experts_mx[(layer, proj)] = moe.EschaExperts(
                    group["escha_code"], group["escha_rin"], group["escha_rout"])
                del self._experts_np[(layer, proj)]
            return
        if s.endswith(".weight_int8") or s.endswith(".weight_scale"):
            base_name, leaf = s.rsplit(".", 1)
            if ".shared_expert." in s:
                self._shared_np.setdefault(base_name, {})[leaf] = w
                return
            pair = self._int8_np.setdefault(base_name, {})
            pair[leaf] = w
            if len(pair) == 2:
                self._install_q8(base_name, pair)
                del self._int8_np[base_name]
            return
        if s.endswith(".mlp.gate.weight") or s.endswith(".mlp.shared_expert_gate.weight"):
            self._mlp_fp16[(int(parts[1]), parts[3])] = w
            return
        self._base[name] = w

    def finalize(self) -> list[mx.array]:
        assert not self._experts_np and not self._int8_np, \
            (list(self._experts_np), list(self._int8_np))

        # ---- MoE blocks --------------------------------------------------
        escha_arrays: list[mx.array] = []
        for i in range(self.n_layers):
            gu = self._experts_mx.pop((i, "gate_up_proj"))
            dn = self._experts_mx.pop((i, "down_proj"))
            assert gu.K == 2 and dn.K == 3, (gu.K, dn.K)
            pref = f"layers.{i}.mlp.shared_expert"
            shared = {}
            for p in ("gate", "up", "down"):
                pair = self._shared_np.pop(f"{pref}.{p}_proj")
                shared[f"{p}_w8"] = pair["weight_int8"]
                shared[f"{p}_scale"] = pair["weight_scale"]
            block = EschaSparseMoeBlock(
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                top_k=self.top_k,
                gu=gu, dn=dn,
                gate_w=self._mlp_fp16.pop((i, "gate")),
                shg_w=self._mlp_fp16.pop((i, "shared_expert_gate")),
                shared=shared,
                group_size=self.group_size,
            )
            self.layers[i].mlp = block
            escha_arrays += gu.arrays() + dn.arrays()

        assert not self._experts_mx, \
            f"unconsumed expert tensors: {list(self._experts_mx)[:4]}"

        # ---- fp16 remainder through mlx-lm's own sanitize ----------------
        assert any(k.endswith("conv1d.weight") and v.shape[-1] != 1
                   for k, v in self._base.items()), \
            "conv1d already sanitized? norm (1+w) shift heuristic would not fire"
        sanitized = self.model.sanitize({k: mx.array(v) for k, v in self._base.items()})
        self.model.update(tree_unflatten(list(sanitized.items())))
        self._base.clear()

        # ---- post-load quirks -------------------------------------------
        if use_last_logit():
            self.model.language_model.lm_head = LastPositionHead(
                self.model.language_model.lm_head)
            logger.info("escha_mlx: LM head restricted to the last position "
                        "(ESCHA_MLX_LAST_LOGIT=0 for per-position logits)")
        gdn_cache.install(self.model)

        if use_fuse_attn():
            fuse_attention_layers(self.model, self.n_layers)
            logger.info("escha_mlx: fused attention q/k/v projections")

        if use_fuse_gdn():
            fuse_gdn_layers(self.model, self.n_layers)
            logger.info("escha_mlx: fused GDN in-projections (qkv+z, b+a)")

        logger.info("escha_mlx: %d MoE layers, %d Q8 dense, %d dropped",
                    self.n_layers, self.n_q8, self.dropped)
        return escha_arrays
