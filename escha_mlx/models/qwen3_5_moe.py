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

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten

from .. import envs, gdn_cache, moe, msl, quant, ref
from ..loader import LastPositionHead, resolve_module, strip_lm_prefix, use_last_logit

logger = logging.getLogger(__name__)

MODEL_TYPE = "qwen3_5_moe"

RS = ref.RS

_DROP_LEAVES = {"escha_s_in", "escha_s_out", "escha_config"}


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
        self.sh_gate = quant.make_linear(shared["gate_w8"], shared["gate_scale"], group_size)
        self.sh_up = quant.make_linear(shared["up_w8"], shared["up_scale"], group_size)
        self.sh_down = quant.make_linear(shared["down_w8"], shared["down_scale"], group_size)

        self._mode = envs.ESCHA_MLX_MOE.get(
            default="fused" if mx.metal.is_available() else "ops")
        # ESCHA_MLX_BLOCK_R pins rows-per-group (1 = always the per-row kernel);
        # unset = the size-dependent policy in _blocked_R.
        self._fused_had = msl.use_fused_had() and self._mode == "fused"
        self._block_env = envs.ESCHA_MLX_BLOCK_R.get()
        # Read once at construction: flipping it per-forward would defeat the
        # compile cache and make A/Bs depend on call order.

    # -- expert path ------------------------------------------------------

    def _rows(self, xf: mx.array, ids: mx.array):
        t = xf.shape[0]
        row_expert = ids.reshape(t * self.top_k).astype(mx.int32)
        row_token = mx.repeat(mx.arange(t, dtype=mx.int32), self.top_k)
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
        """
        if self._block_env is not None:
            return self._block_env
        if m >= 2048:
            return 12
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
        xh = self._input_rows(xf, row_token, row_expert, self._gu)
        mid = self._gemv(xh, row_expert, self._gu, groups)
        gu16 = self._output_rows(mid, row_expert, self._gu)
        g = gu16[:, :self._inter].astype(mx.float32)
        s16 = (g * mx.sigmoid(g)).astype(mx.float16)
        h = s16 * gu16[:, self._inter:]
        xh2 = self._scaled_had(h, row_expert, self._dn)
        mid2 = self._gemv(xh2, row_expert, self._dn, groups)
        d16 = self._output_rows(mid2, row_expert, self._dn)
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

        g = self.sh_gate(xf).astype(mx.float32)
        u = self.sh_up(xf).astype(mx.float32)
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

        logger.info("escha_mlx: %d MoE layers, %d Q8 dense, %d dropped",
                    self.n_layers, self.n_q8, self.dropped)
        return escha_arrays
