"""Split-K GEMV: deterministic, and within f32 rounding of the sequential kernel.

Split-K is the ONE place in this runtime that is deliberately not bit-identical
to its reference: partitioning the kt chain reassociates an f32 sum, and f32
addition is not associative. What must still hold, and is gated here:

  1. run-to-run BIT-REPRODUCIBILITY (the partition is fixed by S, each split
     accumulates in a fixed kt order, and the cross-split reduction is a
     fixed-order mx.sum rather than an atomic),
  2. agreement with the sequential kernel at f32 rounding level,
  3. agreement with an exact f64 reference at least as good as the sequential
     kernel's -- reassociation must not make the answer worse,
  4. the split policy only ever picks an S that divides TK.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from escha_mlx import msl

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

CASES = [(2, 2048, 1024), (3, 512, 2048)]


def _seq(kern, xh, code, re, K, IC, OC, lut):
    m = xh.shape[0]
    inputs = [xh, code.reshape(-1), re] + ([msl._lut_array()] if lut else [])
    (mid,) = kern(
        inputs=inputs,
        template=[("TK", IC // 16), ("TN", OC // 16), ("IC", IC), ("OC", OC)],
        grid=(256 * (OC // 128), m, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, OC)],
        output_dtypes=[mx.float32],
    )
    return mid


def _splitk(K, lut, shuffle, S, xh, code, re, IC, OC):
    m = xh.shape[0]
    kern = msl._moe_gemv_splitk_kernel(K, lut, shuffle, S)
    inputs = [xh, code.reshape(-1), re] + ([msl._lut_array()] if lut else [])
    (part,) = kern(
        inputs=inputs,
        template=[("TK", IC // 16), ("TN", OC // 16), ("IC", IC), ("OC", OC),
                  ("M", m)],
        grid=(256 * (OC // 128), m, S),
        threadgroup=(256, 1, 1),
        output_shapes=[(S, m, OC)],
        output_dtypes=[mx.float32],
    )
    return part.sum(axis=0)


def _data(K, IC, OC, m, seed):
    E = 64
    rng = np.random.default_rng(seed)
    tk, tn, wpt = IC // 16, OC // 16, 8 * K
    code = mx.array(rng.integers(0, 2**32, size=(E, tk, tn, wpt),
                                 dtype=np.uint64).astype(np.uint32))
    xh = mx.array(rng.standard_normal((m, IC)).astype(np.float16))
    re = mx.array(rng.integers(0, E, size=m).astype(np.int32))
    mx.eval(code, xh, re)
    return code, xh, re


@metal
@pytest.mark.parametrize("K,IC,OC", CASES)
@pytest.mark.parametrize("m", [1, 8, 16, 64])
@pytest.mark.parametrize("S", [2, 4, 8])
def test_splitk_close_to_sequential(K, IC, OC, m, S):
    tk = IC // 16
    if tk % S:
        pytest.skip(f"S={S} does not divide TK={tk}")
    code, xh, re = _data(K, IC, OC, m, K * 91 + m * 7 + S)
    seq = _seq(msl._moe_gemv_kernel(K, False, True, True), xh, code, re,
               K, IC, OC, False)
    sk = _splitk(K, False, True, S, xh, code, re, IC, OC)
    mx.eval(seq, sk)
    a, b = np.array(seq), np.array(sk)
    scale = max(np.abs(a).max(), 1e-6)
    assert np.abs(a - b).max() <= 2e-5 * scale, (
        f"split-K drifted from sequential by {np.abs(a-b).max():.3e} "
        f"(scale {scale:.3e})")


@metal
@pytest.mark.parametrize("K,IC,OC", CASES)
@pytest.mark.parametrize("S", [2, 4])
def test_splitk_is_bit_reproducible(K, IC, OC, S):
    """The actual guarantee: same inputs -> same bits, every run."""
    code, xh, re = _data(K, IC, OC, 8, 404 + S)
    ref = None
    for _ in range(16):
        out = _splitk(K, False, True, S, xh, code, re, IC, OC)
        mx.eval(out)
        a = np.array(out)
        if ref is None:
            ref = a
        else:
            assert np.array_equal(a.view(np.uint32), ref.view(np.uint32)), (
                "split-K is not bit-reproducible — the cross-split reduction "
                "must be a fixed-order sum, never an atomic")


@metal
@pytest.mark.parametrize("K,IC,OC", CASES)
def test_splitk_no_worse_than_sequential_vs_exact(K, IC, OC):
    """Reassociation must not degrade accuracy against an exact f64 reference.

    Both kernels sum the same products; split-K sums them as S partial trees,
    which is typically slightly MORE accurate, never materially worse.
    """
    m, S = 8, 4
    if (IC // 16) % S:
        pytest.skip("S does not divide TK")
    code, xh, re = _data(K, IC, OC, m, 1234)
    seq = _seq(msl._moe_gemv_kernel(K, False, True, True), xh, code, re,
               K, IC, OC, False)
    sk = _splitk(K, False, True, S, xh, code, re, IC, OC)
    mx.eval(seq, sk)

    # Exact reference: decode the weights and accumulate in f64.
    from escha_mlx import ref as refmod
    code_np = np.array(code)
    re_np = np.array(re)
    xh_np = np.array(xh).astype(np.float64)
    exact = np.zeros((m, OC), dtype=np.float64)
    cache = {}
    for r in range(m):
        e = int(re_np[r])
        if e not in cache:
            cache[e] = refmod.reconstruct_fast(
                code_np[e].view(np.uint16).view(np.int16), IC, OC, K).astype(np.float64)
        exact[r] = xh_np[r] @ cache[e]

    e_seq = np.abs(np.array(seq).astype(np.float64) - exact).max()
    e_sk = np.abs(np.array(sk).astype(np.float64) - exact).max()
    assert e_sk <= e_seq * 2.0 + 1e-4, (
        f"split-K error {e_sk:.3e} materially worse than sequential {e_seq:.3e}")


def test_split_defaults_off():
    """Split-K measured a regression on M4; the default must be the sequential
    kernel. (The kernel is kept for wider GPUs — see msl.split_k_for.)"""
    import os
    from escha_mlx import msl
    saved = os.environ.pop("ESCHA_MLX_SPLITK", None)
    try:
        for m in (1, 8, 16, 32, 64, 128):
            assert msl.split_k_for(m, 1024, 128) == 1
            assert msl.split_k_for(m, 2048, 32) == 1
    finally:
        if saved is not None:
            os.environ["ESCHA_MLX_SPLITK"] = saved


def test_split_policy_only_divides_tk():
    """A policy S that does not divide TK would silently drop kt iterations."""
    import os
    saved = os.environ.get("ESCHA_MLX_SPLITK")
    os.environ["ESCHA_MLX_SPLITK"] = "auto"
    try:
        for oc in (1024, 2048):
            for tk in (32, 128):
                for m in (1, 8, 16, 64, 128, 512):
                    s = msl.split_k_for(m, oc, tk)
                    assert s >= 1
                    assert tk % s == 0, (m, oc, tk, s)
                    # never split past the target parallelism
                    assert (oc // 128) * m * s <= max(
                        msl._SPLIT_TG_TARGET, (oc // 128) * m)
        # explicit pin is honoured, and a non-dividing pin degrades to 1
        os.environ["ESCHA_MLX_SPLITK"] = "4"
        assert msl.split_k_for(8, 1024, 128) == 4
        os.environ["ESCHA_MLX_SPLITK"] = "5"
        assert msl.split_k_for(8, 1024, 128) == 1
        os.environ["ESCHA_MLX_SPLITK"] = "1"
        assert msl.split_k_for(8, 1024, 128) == 1
    finally:
        os.environ.pop("ESCHA_MLX_SPLITK", None)
        if saved is not None:
            os.environ["ESCHA_MLX_SPLITK"] = saved
