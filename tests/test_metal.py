"""Metal-only kernel tests (Apple Silicon). The correctness gates for the MSL.

Run on the Mac:  pytest tests/test_metal.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import needs_metal

pytestmark = needs_metal


def _set_lut(monkeypatch, lut: bool) -> None:
    from escha_mlx import msl
    monkeypatch.setenv("ESCHA_MLX_LUT", "1" if lut else "0")
    msl._lut_array.cache_clear()


@pytest.mark.parametrize("lut", [False, True], ids=["hash", "lut"])
@pytest.mark.parametrize("K", [2, 3])
def test_decode_tiles_bit_exact(K, lut, k2_golden, k3_golden, monkeypatch):
    import mlx.core as mx
    from escha_mlx import msl
    _set_lut(monkeypatch, lut)
    packed, expected = k2_golden if K == 2 else k3_golden
    ic, oc = expected.shape
    code = mx.array(msl.code_to_u32(packed))
    got = np.array(msl.decode_tiles(code, K, ic, oc))
    if not np.array_equal(got.view(np.uint16), expected.view(np.uint16)):
        n = (got.view(np.uint16) != expected.view(np.uint16)).sum()
        d = np.abs(got.astype(np.float32) - expected.astype(np.float32)).max()
        pytest.fail(f"decode_tiles K={K} lut={lut}: {n} mismatches, max abs {d}")


def _expert_stack(packed: np.ndarray, E: int, rng) -> np.ndarray:
    """E distinct expert streams: expert 0 = the golden, rest = shuffled tiles."""
    stack = [packed]
    flat = packed.reshape(-1, packed.shape[-1])
    for _ in range(E - 1):
        stack.append(flat[rng.permutation(len(flat))].reshape(packed.shape))
    return np.stack(stack)


@pytest.mark.parametrize("lut", [False, True], ids=["hash", "lut"])
@pytest.mark.parametrize("K", [2, 3])
def test_moe_gemv_expert_offsets(K, lut, k2_golden, k3_golden, monkeypatch):
    """gemv vs per-row reference matmul with a MIXED row_expert (0 and E-1
    included) — the expert-stride indexing is otherwise untested anywhere."""
    import mlx.core as mx
    from escha_mlx import msl, ref
    _set_lut(monkeypatch, lut)

    packed = (k2_golden if K == 2 else k3_golden)[0]
    ic, oc = (k2_golden if K == 2 else k3_golden)[1].shape
    rng = np.random.default_rng(K)
    E = 4
    codes = _expert_stack(packed, E, rng)
    w_ref = [ref.reconstruct_fast(codes[e], ic, oc, K).astype(np.float32)
             for e in range(E)]

    m = 8
    xh = (rng.standard_normal((m, ic)) * 0.05).astype(np.float16)
    row_expert = np.array([0, E - 1, 1, 2, E - 1, 0, 2, 1], dtype=np.int32)
    code_mx = mx.array(codes.reshape(E, ic // 16, oc // 16, -1).view(np.uint16).view(np.uint32))
    mid = np.array(msl.moe_gemv(mx.array(xh), code_mx, mx.array(row_expert), K, ic, oc))
    for r in range(m):
        want = xh[r].astype(np.float32) @ w_ref[int(row_expert[r])]
        d = np.abs(mid[r] - want).max()
        denom = max(np.abs(want).max(), 1e-6)
        assert d / denom < 2e-3, (r, int(row_expert[r]), d, denom)


def test_gemv_ship_matches_reference(k2_golden, monkeypatch):
    """The shipped direct GEMV (KB staging removed) must match the reference
    matmul within the codec tolerance for every K/LUT/shuffle combination."""
    import mlx.core as mx
    from escha_mlx import msl, ref
    _set_lut(monkeypatch, False)

    packed, expected = k2_golden
    ic, oc = expected.shape
    rng = np.random.default_rng(7)
    E = 4
    codes = _expert_stack(packed, E, rng)
    w_ref = [ref.reconstruct_fast(codes[e], ic, oc, 2).astype(np.float32)
             for e in range(E)]
    m = 8
    xh = (rng.standard_normal((m, ic)) * 0.05).astype(np.float16)
    # mixed experts incl. the staggered last expert; some rows repeat an expert
    row_expert = np.array([0, E - 1, 1, 2, 1, 0, 2, E - 1], dtype=np.int32)
    code_mx = mx.array(codes.reshape(E, ic // 16, oc // 16, -1)
                       .view(np.uint16).view(np.uint32))

    got = np.array(msl.moe_gemv(mx.array(xh), code_mx,
                                mx.array(row_expert), 2, ic, oc))
    for r in range(m):
        want = xh[r].astype(np.float32) @ w_ref[int(row_expert[r])]
        d = np.abs(got[r] - want).max()
        denom = max(np.abs(want).max(), 1e-6)
        assert d / denom < 2e-3, (r, int(row_expert[r]), d, denom)


@pytest.mark.parametrize("lut", [False, True], ids=["hash", "lut"])
@pytest.mark.parametrize("K,m", [(2, 8), (2, 64), (3, 8), (3, 64)])
def test_gemv_had_fused_bit_identical(K, m, lut, monkeypatch):
    """[scaled_had; moe_gemv] == moe_gemv_had, bit-for-bit (fused transform
    runs the same butterfly in threadgroup memory; GEMV order unchanged)."""
    import mlx.core as mx
    from escha_mlx import msl, ref
    _set_lut(monkeypatch, lut)
    rng = np.random.default_rng(3)
    E, IC, OC = 64, (2048 if K == 2 else 512), (1024 if K == 2 else 2048)
    nw = 32 if K == 2 else 48
    code = rng.integers(-32768, 32768, size=(E, IC // 16, OC // 16, nw),
                        dtype=np.int16)
    code_mx = mx.array(msl.code_to_u32(code.reshape(E, IC // 16, OC // 16, -1)))
    rin = mx.array(rng.random((E, IC), dtype=np.float32))
    x = mx.random.normal((m, IC), dtype=mx.float16)
    re = mx.array((mx.arange(m) * 13 % E).astype(mx.int32))
    xh = msl.scaled_had(x, rin, re, ref.RS)
    ref_mid = mx.array(msl.moe_gemv(xh, code_mx, re, K, IC, OC))
    got = msl.moe_gemv_had(x, rin, code_mx, re, mx.arange(m, dtype=mx.int32), K, IC, OC, ref.RS)
    mx.eval(ref_mid, got)
    assert np.array_equal(np.array(got).view(np.uint32),
                          np.array(ref_mid).view(np.uint32))


def test_gemv_hash_equals_lut(k2_golden, monkeypatch):
    """The production gemv hash decode must equal the LUT variant bit-for-bit
    (deterministic same-order f32 accumulation on both sides)."""
    import mlx.core as mx
    from escha_mlx import msl
    packed, expected = k2_golden
    ic, oc = expected.shape
    rng = np.random.default_rng(0)
    xh = mx.array((rng.standard_normal((8, ic)) * 0.05).astype(np.float16))
    code = mx.array(msl.code_to_u32(packed)).reshape(1, ic // 16, oc // 16, 16)
    re = mx.zeros(8, dtype=mx.int32)
    _set_lut(monkeypatch, False)
    a = np.array(msl.moe_gemv(xh, code, re, 2, ic, oc))
    _set_lut(monkeypatch, True)
    b = np.array(msl.moe_gemv(xh, code, re, 2, ic, oc))
    assert np.array_equal(a, b), \
        "gemv hash decode != LUT on this GPU/compiler — set ESCHA_MLX_LUT=1 and report"


def test_decode_hash_equals_lut(monkeypatch):
    """G0.1 core on random streams (not just the golden's value distribution)."""
    import mlx.core as mx
    from escha_mlx import msl
    rng = np.random.default_rng(42)
    code = rng.integers(-32768, 32768, size=(8, 8, 32), dtype=np.int16)  # K=2
    c = mx.array(msl.code_to_u32(code))
    _set_lut(monkeypatch, False)
    a = np.array(msl.decode_tiles(c, 2, 128, 128))
    _set_lut(monkeypatch, True)
    b = np.array(msl.decode_tiles(c, 2, 128, 128))
    assert np.array_equal(a.view(np.uint16), b.view(np.uint16)), \
        "hash decode != LUT on this GPU/compiler — set ESCHA_MLX_LUT=1 and report"


def test_native_hadamard_transform_order():
    """The production native transform must preserve the reference ordering."""
    import mlx.core as mx
    from escha_mlx import ref
    x = np.random.default_rng(1).standard_normal((2, 128)).astype(np.float32)
    want = ref.h128(x)
    got = np.array(mx.hadamard_transform(mx.array(x), scale=1.0))
    assert np.abs(got - want).max() < 1e-3 * np.abs(want).max()
