"""Gate the barrier-free per-row GEMV against the staged one.

`_moe_gemv_direct_source` removes both threadgroup barriers and the shared-memory
staging of code and x.  It changes only WHERE operands are read from, never the
kt/j accumulation order, so the f32 output must be BIT-IDENTICAL.

The existing golden tests (test_metal.py) run whichever variant is default, so
this file is what keeps the *other* variant honest -- both are exercised here.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from escha_mlx import msl

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def _run(kern, xh, code, re, K, IC, OC, lut):
    m = xh.shape[0]
    tk, tn = IC // 16, OC // 16
    inputs = [xh, code.reshape(-1), re] + ([msl._lut_array()] if lut else [])
    (mid,) = kern(
        inputs=inputs,
        template=[("TK", tk), ("TN", tn), ("IC", IC), ("OC", OC)],
        grid=(256 * (OC // 128), m, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, OC)],
        output_dtypes=[mx.float32],
    )
    return mid


@metal
@pytest.mark.parametrize("K,IC,OC", [(2, 2048, 1024), (3, 512, 2048)])
@pytest.mark.parametrize("m", [1, 8, 16, 64, 257])
@pytest.mark.parametrize("lut", [False, True])
def test_shuffle_fetch_matches_load(K, IC, OC, m, lut):
    """Shuffle-broadcast changes only WHERE a code word is read from.

    Lane L loads word L and the two words each lane needs arrive via
    simd_shuffle instead of two redundant device loads. Same values, same
    kt/j accumulation order => bit-identical, not merely close.
    """
    E = 256
    rng = np.random.default_rng(K * 7717 + m + int(lut))
    tk, tn, wpt = IC // 16, OC // 16, 8 * K
    code = mx.array(rng.integers(0, 2**32, size=(E, tk, tn, wpt),
                                 dtype=np.uint64).astype(np.uint32))
    xh = mx.array(rng.standard_normal((m, IC)).astype(np.float16))
    re = mx.array(rng.integers(0, E, size=m).astype(np.int32))
    mx.eval(code, xh, re)

    loads = _run(msl._moe_gemv_kernel(K, lut, True, False), xh, code, re, K, IC, OC, lut)
    shuf = _run(msl._moe_gemv_kernel(K, lut, True, True), xh, code, re, K, IC, OC, lut)
    mx.eval(loads, shuf)

    a, b = np.array(loads), np.array(shuf)
    assert np.array_equal(a, b), (
        f"shuffle fetch diverged: n_diff={(a != b).sum()}/{a.size}, "
        f"max|d|={np.abs(a - b).max()}")


@metal
@pytest.mark.parametrize("K,IC,OC", [(2, 2048, 1024), (3, 512, 2048)])
@pytest.mark.parametrize("m", [1, 8, 16, 64, 257])
@pytest.mark.parametrize("lut", [False, True])
def test_direct_matches_staged(K, IC, OC, m, lut):
    E = 256
    rng = np.random.default_rng(K * 1000 + m + int(lut))
    tk, tn, wpt = IC // 16, OC // 16, 8 * K
    code = mx.array(rng.integers(0, 2**32, size=(E, tk, tn, wpt),
                                 dtype=np.uint64).astype(np.uint32))
    xh = mx.array(rng.standard_normal((m, IC)).astype(np.float16))
    re = mx.array(rng.integers(0, E, size=m).astype(np.int32))
    mx.eval(code, xh, re)

    staged = _run(msl._moe_gemv_kernel(K, lut, False), xh, code, re, K, IC, OC, lut)
    direct = _run(msl._moe_gemv_kernel(K, lut, True), xh, code, re, K, IC, OC, lut)
    mx.eval(staged, direct)

    a, b = np.array(staged), np.array(direct)
    assert np.array_equal(a, b), (
        f"direct GEMV diverged from staged: n_diff={(a != b).sum()}/{a.size}, "
        f"max|d|={np.abs(a - b).max()}")


@metal
def test_direct_matches_golden():
    """The default path must still reproduce the committed goldens."""
    from pathlib import Path
    d = Path(__file__).parent / "data" / "codec"
    if not (d / "packed_gu_e0_k2.i16").exists():
        pytest.skip("goldens not present")
    # decode-side golden is covered by test_metal.py; here we assert the direct
    # kernel reproduces the staged kernel on the REAL packed stream, not random.
    packed = np.fromfile(d / "packed_gu_e0_k2.i16", dtype=np.int16)
    K, IC, OC = 2, 2048, 1024
    tk, tn, wpt = IC // 16, OC // 16, 8 * K
    need = tk * tn * wpt * 2
    if packed.size < need:
        pytest.skip("golden stream shorter than one expert")
    code1 = mx.array(msl.code_to_u32(packed[:need].reshape(tk, tn, wpt * 2))[None])
    rng = np.random.default_rng(0)
    xh = mx.array(rng.standard_normal((8, IC)).astype(np.float16))
    re = mx.zeros((8,), dtype=mx.int32)
    mx.eval(code1, xh, re)
    a = _run(msl._moe_gemv_kernel(K, False, False), xh, code1, re, K, IC, OC, False)
    b = _run(msl._moe_gemv_kernel(K, False, True), xh, code1, re, K, IC, OC, False)
    mx.eval(a, b)
    assert np.array_equal(np.array(a), np.array(b))
