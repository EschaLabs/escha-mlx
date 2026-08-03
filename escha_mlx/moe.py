"""EschaSparseMoeBlock — drop-in replacement for the mlx-lm Qwen3.5/3.6 MoE
block, executing the escha trellis-coded routed experts.

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

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import msl, quant, ref

RS = ref.RS


def _h128_matrix() -> mx.array:
    i = np.arange(128, dtype=np.uint32)
    par = np.bitwise_count(i[:, None] & i[None, :]).astype(np.int64)
    h = np.where(par % 2 == 0, 1.0, -1.0).astype(np.float32)
    return mx.array(h)


_H128: mx.array | None = None


def had_blocks(x: mx.array) -> mx.array:
    """Unnormalized 128-point WHT (Sylvester order) on 128-blocks of last dim.

    Matmul-by-H formulation: order-exact vs ref.h128 on every backend
    (mx.hadamard_transform is the P4 fast path, gated by G0.1).
    """
    global _H128
    if _H128 is None:
        _H128 = _h128_matrix()
    shape = x.shape
    c = shape[-1]
    y = x.reshape(-1, c // 128, 128) @ _H128
    return y.reshape(shape)


def build_groups(row_expert: mx.array, E: int, R: int, n_groups: int,
                 with_sorted: bool = False):
    """Sort rows by expert and pack them into fixed R-row groups.

    Returns (rows_idx [n_groups*R] i32, group_expert [n_groups] i32).
    Padding slots hold M (the caller appends a zero row at index M); groups past
    the end hold expert -1 and the kernel returns immediately.

    Entirely device-resident: no `mx.eval`, no host sync, static shapes -- so it
    does not break the lazy graph the way a data-dependent shape would.
    """
    m = row_expert.shape[0]
    order = mx.argsort(row_expert).astype(mx.int32)        # stable
    se = row_expert[order]
    counts = mx.zeros((E,), dtype=mx.int32).at[row_expert].add(
        mx.ones((m,), dtype=mx.int32))
    padded = ((counts + (R - 1)) // R) * R
    dst_start = mx.cumsum(padded) - padded
    src_start = mx.cumsum(counts) - counts
    dst = dst_start[se] + (mx.arange(m, dtype=mx.int32) - src_start[se])

    # start at the padding sink M, then add (order - M) so occupied slots land
    # on `order` exactly (each dst is unique, so scatter-add is unambiguous).
    rows_idx = mx.full((n_groups * R,), m, dtype=mx.int32).at[dst].add(order - m)

    eid = mx.arange(E, dtype=mx.int32) + 1
    marker = mx.zeros((n_groups,), dtype=mx.int32).at[dst_start // R].add(
        mx.where(counts > 0, eid, mx.zeros_like(eid)))
    total_groups = mx.sum(padded) // R
    gidx = mx.arange(n_groups, dtype=mx.int32)
    group_expert = mx.where(gidx < total_groups, mx.cummax(marker) - 1,
                            mx.full((n_groups,), -1, dtype=mx.int32))

    if not with_sorted:
        return rows_idx, group_expert

    # --- sorted-x addressing (opt-in; measured a wash, see doc §15.4) --------
    # `order` sorts rows by expert, so in xs = xh[order] every expert owns a
    # CONTIGUOUS run starting at src_start[e].  Group grp is the j-th group of
    # its expert, so its R rows live at xs[src_row0 + 0 .. +R), which lets the
    # kernel compute row addresses instead of loading rows_idx and chasing an
    # arbitrary row per staged element.  n_valid marks how many of the R are
    # real (the tail group of an expert is partly padding).
    ge = mx.maximum(group_expert, 0)                 # -1 groups read slot 0
    j = gidx - (dst_start // R)[ge]                  # index of grp within expert
    src_row0 = src_start[ge] + j * R
    n_valid = mx.clip(counts[ge] - j * R, 0, R).astype(mx.int32)
    n_valid = mx.where(group_expert < 0, mx.zeros_like(n_valid), n_valid)
    return rows_idx, group_expert, order, src_row0.astype(mx.int32), n_valid


def n_groups_bound(m: int, E: int, R: int) -> int:
    """Static upper bound on groups: every expert may waste up to R-1 slots."""
    return (m + E * (R - 1) + R - 1) // R


class EschaExperts:
    """Container for one projection's E-stacked arrays (kept off the module
    parameter tree via leading-underscore attributes on the owner)."""

    def __init__(self, code_i16: np.ndarray, rin: np.ndarray, rout: np.ndarray) -> None:
        e, tk, tn, wpt2 = code_i16.shape
        self.K = wpt2 // 16
        self.E, self.IC, self.OC = e, tk * 16, tn * 16
        self.code = mx.array(msl.code_to_u32(code_i16))     # [E, TK, TN, 8K] u32
        self.rin = mx.array(rin.astype(np.float32))          # [E, IC] f32
        self.rout = mx.array(rout.astype(np.float32))        # [E, OC] f32
        self._code_np = None

    def code_numpy(self) -> np.ndarray:
        if self._code_np is None:
            self._code_np = np.array(self.code)
        return self._code_np

    def arrays(self) -> list[mx.array]:
        return [self.code, self.rin, self.rout]


class EschaSparseMoeBlock(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int,
                 gu: EschaExperts, dn: EschaExperts,
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

        self._mode = os.environ.get(
            "ESCHA_MLX_MOE", "fused" if mx.metal.is_available() else "ops")
        if self._mode not in ("fused", "ops"):
            raise ValueError(f"ESCHA_MLX_MOE must be 'fused' or 'ops', got {self._mode!r}")
        # ESCHA_MLX_BLOCK_R pins rows-per-group (1 = always the per-row kernel);
        # unset = the size-dependent policy in _blocked_R.
        self._fused_had = msl.use_fused_had() and self._mode == "fused"
        _br = os.environ.get("ESCHA_MLX_BLOCK_R")
        self._block_env = int(_br) if _br else None
        # Read once at construction: flipping it per-forward would defeat the
        # compile cache and make A/Bs depend on call order.

    # -- expert path ------------------------------------------------------

    def _rows(self, xf: mx.array, ids: mx.array):
        t = xf.shape[0]
        row_expert = ids.reshape(t * self.top_k).astype(mx.int32)
        row_token = mx.repeat(mx.arange(t, dtype=mx.int32), self.top_k)
        return row_expert, row_token

    def _scaled_had(self, rows: mx.array, row_expert: mx.array,
                    ex: EschaExperts) -> mx.array:
        """f16( H128(rows * rin[e]) * RS ) — the input transform for either leg."""
        if self._fused_had and rows.dtype == mx.float16:
            # One kernel instead of ~5 ops each materialising [m, IC] f32.
            return msl.scaled_had(rows, ex.rin, row_expert, RS)
        xr = rows.astype(mx.float32) * ex.rin[row_expert]
        return (had_blocks(xr) * RS).astype(mx.float16)

    def _input_rows(self, xf: mx.array, row_token: mx.array, row_expert: mx.array,
                    ex: EschaExperts) -> mx.array:
        return self._scaled_had(xf[row_token], row_expert, ex)

    def _output_rows(self, mid: mx.array, row_expert: mx.array, ex: EschaExperts) -> mx.array:
        y = had_blocks(mid) * RS * ex.rout[row_expert]
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

    def _gemv(self, xh: mx.array, row_expert: mx.array, ex: EschaExperts,
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
            ng = n_groups_bound(m, self.num_experts, r)
            sx = msl.use_sortx()
            g = build_groups(row_expert, self.num_experts, r, ng, with_sorted=sx)
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
