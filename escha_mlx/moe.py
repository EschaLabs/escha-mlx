"""Architecture-agnostic expert toolkit for escha MoE runtimes.

Shared by every MoE architecture plugin (escha_mlx/models/): the E-stacked
expert container, the 128-block Hadamard transform, and the row-grouping
machinery that feeds the row-blocked GEMM kernel. Routing conventions —
top-k, score normalization, shared experts — are per-architecture and live
in the plugin's MoE block, not here.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from . import msl, ref

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
