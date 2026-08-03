"""Pure-NumPy decode reference vs the committed goldens (runs anywhere)."""
from __future__ import annotations

import numpy as np

from escha_mlx import ref


def test_lane_positions_bijection():
    seen = set()
    for lane in range(32):
        for r, c in ref.lane_positions(lane):
            seen.add((r, c))
    assert len(seen) == 256


def test_k2_reconstruct_bit_exact(k2_golden):
    packed, expected = k2_golden
    got = ref.reconstruct(packed, 2048, 1024, K=2)
    assert np.array_equal(got.view(np.uint16), expected.view(np.uint16))


def test_k3_reconstruct_bit_exact(k3_golden):
    packed, expected = k3_golden
    got = ref.reconstruct(packed, 512, 2048, K=3)
    assert np.array_equal(got.view(np.uint16), expected.view(np.uint16))


def test_fast_reconstruct_matches(k2_golden, k3_golden):
    p2, e2 = k2_golden
    p3, e3 = k3_golden
    assert np.array_equal(ref.reconstruct_fast(p2, 2048, 1024, 2).view(np.uint16),
                          e2.view(np.uint16))
    assert np.array_equal(ref.reconstruct_fast(p3, 512, 2048, 3).view(np.uint16),
                          e3.view(np.uint16))


def test_fast_reconstruct_synthetic_random():
    rng = np.random.default_rng(7)
    for K, ic, oc in ((2, 64, 32), (3, 32, 64)):
        code = rng.integers(-32768, 32768, size=(ic // 16, oc // 16, 16 * K), dtype=np.int16)
        slow = ref.reconstruct(code, ic, oc, K)
        fast = ref.reconstruct_fast(code, ic, oc, K)
        assert np.array_equal(slow.view(np.uint16), fast.view(np.uint16))


def test_h128_involution():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 256)).astype(np.float32)
    y = ref.h128(ref.h128(x)) / 128.0
    assert np.allclose(y, x, atol=1e-4)
