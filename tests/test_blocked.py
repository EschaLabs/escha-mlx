"""Gates for the row-blocked trellis GEMM (msl.moe_gemm_rows + moe.build_groups).

The blocked kernel changes only the loop nest AROUND the accumulation, never the
order of the kt/j chain for a fixed (row, out-channel).  So the requirement is
BIT-IDENTICAL f32 output vs the per-row kernel -- not "close".  Anything weaker
would let a reassociation bug hide behind a tolerance.

Note the explicit `splits=1` below: the default GEMV path now applies split-K at
low row counts, which DOES reassociate the sum by design (see
msl._moe_gemv_splitk_source and tests/test_splitk.py).  The invariant this file
gates holds against the sequential kernel, so that is what it compares to;
without the pin these tests would fail for the wrong reason.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from escha_mlx import moe, msl

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def _groups_ref(row_expert: np.ndarray, E: int, R: int, n_groups: int):
    """Independent NumPy model of build_groups."""
    m = len(row_expert)
    order = np.argsort(row_expert, kind="stable")
    counts = np.bincount(row_expert, minlength=E)
    padded = ((counts + R - 1) // R) * R
    dst_start = np.cumsum(padded) - padded
    rows_idx = np.full(n_groups * R, m, dtype=np.int32)
    group_expert = np.full(n_groups, -1, dtype=np.int32)
    p = 0
    for e in range(E):
        c = counts[e]
        if c == 0:
            continue
        rows_idx[dst_start[e]:dst_start[e] + c] = order[p:p + c]
        for g in range(padded[e] // R):
            group_expert[dst_start[e] // R + g] = e
        p += c
    return rows_idx, group_expert


@pytest.mark.parametrize("m,E,R", [(64, 16, 4), (2048, 256, 16), (517, 32, 8), (8, 256, 4)])
def test_build_groups_matches_reference(m, E, R):
    rng = np.random.default_rng(0)
    re_np = rng.integers(0, E, size=m).astype(np.int32)
    ng = moe.n_groups_bound(m, E, R)
    ri, ge = moe.build_groups(mx.array(re_np), E, R, ng)[:2]
    mx.eval(ri, ge)
    ri_ref, ge_ref = _groups_ref(re_np, E, R, ng)

    assert np.array_equal(np.array(ge), ge_ref), "group_expert mismatch"
    assert np.array_equal(np.array(ri), ri_ref), "rows_idx mismatch"

    # every real row appears exactly once, and in a group whose expert matches it
    ri_a, ge_a = np.array(ri), np.array(ge)
    real = ri_a[ri_a < m]
    assert sorted(real.tolist()) == list(range(m)), "rows lost or duplicated"
    for g in range(ng):
        blk = ri_a[g * R:(g + 1) * R]
        occ = blk[blk < m]
        if len(occ):
            assert (re_np[occ] == ge_a[g]).all(), f"group {g} mixes experts"


@metal
@pytest.mark.parametrize("K,IC,OC", [(2, 2048, 1024), (3, 512, 2048)])
@pytest.mark.parametrize("m,R", [(64, 4), (2048, 16), (300, 8)])
def test_blocked_bit_identical_to_per_row(K, IC, OC, m, R):
    E = 256
    rng = np.random.default_rng(K * 100 + m)
    tk, tn, wpt = IC // 16, OC // 16, 8 * K
    code = mx.array(rng.integers(0, 2**32, size=(E, tk, tn, wpt), dtype=np.uint64
                                 ).astype(np.uint32))
    xh = mx.array(rng.standard_normal((m, IC)).astype(np.float16))
    re = mx.array(rng.integers(0, E, size=m).astype(np.int32))
    mx.eval(code, xh, re)

    ref = msl.moe_gemv(xh, code, re, K, IC, OC, splits=1)
    mx.eval(ref)

    ng = moe.n_groups_bound(m, E, R)
    rows_idx, grp_e = moe.build_groups(re, E, R, ng)[:2]
    xh_pad = mx.concatenate([xh, mx.zeros((1, IC), dtype=xh.dtype)], axis=0)
    got = msl.moe_gemm_rows(xh_pad, code, rows_idx, grp_e, K, IC, OC, R, m)[:m]
    mx.eval(got)

    a, b = np.array(ref), np.array(got)
    assert a.shape == b.shape
    assert np.array_equal(a, b), (
        f"blocked kernel not bit-identical: max|d|={np.abs(a-b).max()}, "
        f"n_diff={(a != b).sum()}/{a.size}")


@metal
@pytest.mark.parametrize("kb", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("K,IC,OC", [(2, 2048, 1024), (3, 512, 2048)])
def test_kt_block_is_bit_identical(kb, K, IC, OC):
    """Staging KB kt-tiles per barrier pair must not move a single bit.

    KT_BLOCK only changes how many kt tiles are staged between barriers; the
    kt/j accumulation order inside is untouched, so every output bit must match
    KB=1. If a future edit reorders the inner loop this fails immediately.
    """
    E, R, m = 64, 8, 96
    rng = np.random.default_rng(K * 31 + kb)
    tk, tn, wpt = IC // 16, OC // 16, 8 * K
    assert tk % kb == 0
    code = mx.array(rng.integers(0, 2**32, size=(E, tk, tn, wpt),
                                 dtype=np.uint64).astype(np.uint32))
    xh = mx.array(rng.standard_normal((m + 1, IC)).astype(np.float16))
    re = mx.array(rng.integers(0, E, size=m).astype(np.int32))
    mx.eval(code, xh, re)
    ng = moe.n_groups_bound(m, E, R)
    ri, ge = moe.build_groups(re, E, R, ng)[:2]

    outs = {}
    for k in (1, kb):
        kern = msl._moe_gemm_rows_kernel(K, False, R, k)
        inputs = [xh, code.reshape(-1), ri, ge]
        (mid,) = kern(
            inputs=inputs,
            template=[("TK", tk), ("TN", tn), ("IC", IC), ("OC", OC)],
            grid=(256 * (OC // 128), ng, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(m + 1, OC)],
            output_dtypes=[mx.float32],
        )
        mx.eval(mid)
        outs[k] = np.array(mid)[:m]
    assert np.array_equal(outs[1], outs[kb]), (
        f"KT_BLOCK={kb} diverged from KB=1: "
        f"n_diff={(outs[1] != outs[kb]).sum()}/{outs[1].size}")


@metal
def test_blocked_handles_extreme_skew():
    """All rows on ONE expert, and one row per expert -- the padding edge cases."""
    K, IC, OC, E, R = 2, 2048, 1024, 256, 16
    rng = np.random.default_rng(7)
    tk, tn, wpt = IC // 16, OC // 16, 8 * K
    code = mx.array(rng.integers(0, 2**32, size=(E, tk, tn, wpt), dtype=np.uint64
                                 ).astype(np.uint32))
    for name, re_np in (
        ("all-one-expert", np.full(64, 3, dtype=np.int32)),
        ("one-per-expert", np.arange(E, dtype=np.int32)),
        ("single-row", np.array([17], dtype=np.int32)),
    ):
        m = len(re_np)
        xh = mx.array(rng.standard_normal((m, IC)).astype(np.float16))
        re = mx.array(re_np)
        ref = msl.moe_gemv(xh, code, re, K, IC, OC, splits=1)
        ng = moe.n_groups_bound(m, E, R)
        ri, ge = moe.build_groups(re, E, R, ng)[:2]
        xh_pad = mx.concatenate([xh, mx.zeros((1, IC), dtype=xh.dtype)], axis=0)
        got = msl.moe_gemm_rows(xh_pad, code, ri, ge, K, IC, OC, R, m)[:m]
        mx.eval(ref, got)
        assert np.array_equal(np.array(ref), np.array(got)), f"{name} mismatch"


@metal
@pytest.mark.parametrize("K,IC,OC", [(2, 2048, 1024), (3, 512, 2048)])
@pytest.mark.parametrize("m,R", [(64, 4), (300, 8), (2048, 12)])
def test_sorted_x_is_bit_identical(K, IC, OC, m, R):
    """Pre-sorting x changes WHERE rows are read from, never which or in what order.

    With xs = xh[order] a group's rows are consecutive, so the kernel computes
    `src_row0 + rr` instead of loading `rows_idx[grp*R+rr]`. The staged values
    are the same values in the same kt/j order -- padding contributes 0 either
    way -- so the f32 output must be bit-identical, not merely close.
    """
    E = 256
    rng = np.random.default_rng(K * 977 + m + R)
    tk, tn, wpt = IC // 16, OC // 16, 8 * K
    code = mx.array(rng.integers(0, 2**32, size=(E, tk, tn, wpt),
                                 dtype=np.uint64).astype(np.uint32))
    xh = mx.array(rng.standard_normal((m, IC)).astype(np.float16))
    re = mx.array(rng.integers(0, E, size=m).astype(np.int32))
    mx.eval(code, xh, re)

    ng = moe.n_groups_bound(m, E, R)
    rows_idx, grp_e, order, src_row0, n_valid = moe.build_groups(re, E, R, ng, with_sorted=True)

    xh_pad = mx.concatenate([xh, mx.zeros((1, IC), dtype=xh.dtype)], axis=0)
    plain = msl.moe_gemm_rows(xh_pad, code, rows_idx, grp_e, K, IC, OC, R, m)[:m]
    sortd = msl.moe_gemm_rows(xh[order], code, rows_idx, grp_e, K, IC, OC, R, m,
                              sort_idx=(src_row0, n_valid))[:m]
    mx.eval(plain, sortd)

    a, b = np.array(plain), np.array(sortd)
    assert np.array_equal(a, b), (
        f"sorted-x diverged: n_diff={(a != b).sum()}/{a.size}, "
        f"max|d|={np.abs(a - b).max()}")


@metal
def test_sorted_x_addressing_is_consistent():
    """src_row0/n_valid must describe exactly the rows rows_idx names.

    This is the invariant the kernel relies on; if build_groups ever computes
    the sorted-run offsets differently from the padded layout, the GEMM would
    silently read the wrong rows.
    """
    E, R = 64, 6
    for m in (1, 37, 200, 1000):
        rng = np.random.default_rng(m)
        re_np = rng.integers(0, E, size=m).astype(np.int32)
        ng = moe.n_groups_bound(m, E, R)
        ri, ge, order, s0, nv = moe.build_groups(mx.array(re_np), E, R, ng, with_sorted=True)
        mx.eval(ri, ge, order, s0, nv)
        ri, ge, order, s0, nv = (np.array(x) for x in (ri, ge, order, s0, nv))
        for g in range(ng):
            if ge[g] < 0:
                assert nv[g] == 0, f"unused group {g} claims {nv[g]} valid rows"
                continue
            for r in range(R):
                slot = ri[g * R + r]
                if r < nv[g]:
                    assert slot < m, f"group {g} row {r} marked valid but padded"
                    assert order[s0[g] + r] == slot, (
                        f"group {g} row {r}: sorted position {s0[g] + r} holds "
                        f"row {order[s0[g] + r]}, rows_idx says {slot}")
                else:
                    assert slot == m, f"group {g} row {r} beyond n_valid is real"


@metal
@pytest.mark.parametrize("kb", [2, 4, 8])
@pytest.mark.parametrize("K,IC,OC", [(2, 2048, 1024), (3, 512, 2048)])
def test_code_prefetch_is_bit_identical(kb, K, IC, OC):
    """Prefetching KB code tiles must not move a bit.

    The two word offsets a lane needs depend on `lane` only, never on kt, so
    hoisting the fetch changes WHEN the words arrive, not which words or the
    order they are combined in. Anything else is a bug.
    """
    E, R, m = 64, 8, 96
    rng = np.random.default_rng(K * 613 + kb)
    tk, tn, wpt = IC // 16, OC // 16, 8 * K
    assert tk % kb == 0
    code = mx.array(rng.integers(0, 2**32, size=(E, tk, tn, wpt),
                                 dtype=np.uint64).astype(np.uint32))
    xh = mx.array(rng.standard_normal((m + 1, IC)).astype(np.float16))
    re = mx.array(rng.integers(0, E, size=m).astype(np.int32))
    mx.eval(code, xh, re)
    ng = moe.n_groups_bound(m, E, R)
    ri, ge = moe.build_groups(re, E, R, ng)[:2]

    outs = {}
    for pf in (False, True):
        kern = msl._moe_gemm_rows_kernel(K, False, R, kb, False, pf)
        (mid,) = kern(
            inputs=[xh, code.reshape(-1), ri, ge],
            template=[("TK", tk), ("TN", tn), ("IC", IC), ("OC", OC)],
            grid=(256 * (OC // 128), ng, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(m + 1, OC)],
            output_dtypes=[mx.float32],
        )
        mx.eval(mid)
        outs[pf] = np.array(mid)[:m]
    a, b = outs[False], outs[True]
    assert np.array_equal(a, b), (
        f"code prefetch diverged: n_diff={(a != b).sum()}/{a.size}, "
        f"max|d|={np.abs(a - b).max()}")
