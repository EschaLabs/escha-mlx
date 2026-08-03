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


def test_mx_hadamard_transform_order():
    """Informational (P4 fast path): does mx.hadamard_transform match Sylvester
    order? The production path is the matmul in moe.had_blocks either way —
    a mismatch here is a note for the perf campaign, NOT a failure."""
    import mlx.core as mx
    from escha_mlx import ref
    x = np.random.default_rng(1).standard_normal((2, 128)).astype(np.float32)
    want = ref.h128(x)
    try:
        got = np.array(mx.hadamard_transform(mx.array(x), scale=1.0))
    except Exception as e:  # pragma: no cover
        print(f"mx.hadamard_transform unavailable: {e!r}")
        return
    ok = np.abs(got - want).max() < 1e-3 * np.abs(want).max()
    print(f"mx.hadamard_transform Sylvester-order match: {ok} "
          f"({'fast path usable in P4' if ok else 'keep matmul path'})")
