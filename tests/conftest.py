from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from escha_mlx import envs

HERE = Path(__file__).parent
CODEC = HERE / "data" / "codec"    # format-level goldens (arch-independent)
ARCH = HERE / "data"               # per-architecture goldens: data/<model_type>/
DEFAULT_CKPT = envs.ESCHA_MODEL.get()
DENSE_CKPT = envs.ESCHA_DENSE_MODEL.get()


def has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except Exception:
        return False


def has_metal() -> bool:
    if not has_mlx():
        return False
    import mlx.core as mx
    return mx.metal.is_available()


needs_mlx = pytest.mark.skipif(not has_mlx(), reason="mlx not installed")
needs_metal = pytest.mark.skipif(not has_metal(), reason="Metal not available")
needs_ckpt = pytest.mark.skipif(
    not DEFAULT_CKPT or not Path(DEFAULT_CKPT).exists(),
    reason=(
        f"checkpoint not found: {DEFAULT_CKPT}"
        if DEFAULT_CKPT else "ESCHA_MODEL is not set"
    ),
)
needs_dense_ckpt = pytest.mark.skipif(
    not DENSE_CKPT or not Path(DENSE_CKPT).exists(),
    reason=(
        f"dense checkpoint not found: {DENSE_CKPT}"
        if DENSE_CKPT else "ESCHA_DENSE_MODEL is not set"
    ),
)
# Opt-in: reads real coded weights and peaks around 8 GB RSS. Not something a
# plain `pytest tests/` on a laptop should start without being asked.
needs_slow = pytest.mark.skipif(
    not envs.ESCHA_MLX_SLOW_TESTS.get(),
    reason="slow/heavy test; set ESCHA_MLX_SLOW_TESTS=1")


@pytest.fixture(scope="session")
def k2_golden():
    packed = np.fromfile(CODEC / "packed_gu_e0_k2.i16", dtype=np.int16).reshape(128, 64, 32)
    expected = np.fromfile(CODEC / "expected_gu_e0_k2.f16", dtype=np.float16).reshape(2048, 1024)
    return packed, expected


@pytest.fixture(scope="session")
def k3_golden():
    packed = np.fromfile(CODEC / "packed_down_e0_k3.i16", dtype=np.int16).reshape(32, 128, 48)
    expected = np.fromfile(CODEC / "expected_down_e0_k3.f16", dtype=np.float16).reshape(512, 2048)
    return packed, expected


@pytest.fixture(scope="session")
def w8a16_golden():
    w8 = np.fromfile(CODEC / "w8a16_w8_2048x2048.i8", dtype=np.int8).reshape(2048, 2048)
    scale = np.fromfile(CODEC / "w8a16_scale_2048.f32", dtype=np.float32)
    x = np.fromfile(CODEC / "w8a16_x_8x2048.f16", dtype=np.float16).reshape(8, 2048)
    want = np.fromfile(CODEC / "w8a16_expected_8x2048.f16", dtype=np.float16).reshape(8, 2048)
    return w8, scale, x, want


@pytest.fixture(scope="session", params=["k2", "k3"])
def dense_linear_golden(request):
    """One 128x128 corner of a shipped dense linear + its reference output.

    Real coded data, not synthetic: the top-left 8x8 tiles of a shipped
    projection, which decode to exactly the top-left 128x128 of that weight
    (tiles are independent, and 128 is one Hadamard block on each side, so the
    corner is a self-contained linear).  `deploy` is the reference output
    shipped for `x` — produced outside this package, so the tolerance in
    tests/test_dense_linear.py is a genuine cross-runtime gate rather than a
    self-comparison.  k2 is from a self_attn.k_proj, k3 from an mlp.down_proj.
    """
    tag = request.param
    d = ARCH / "qwen3_5"
    def rd(sfx, dt, shape):
        return np.fromfile(d / f"lin_{tag}_{sfx}", dtype=dt).reshape(shape)
    K = 2 if tag == "k2" else 3
    return {
        "K": K,
        "code": rd("code.i16", np.int16, (8, 8, 16 * K)),
        "rin": rd("rin.f16", np.float16, (128,)),
        "rout": rd("rout.f16", np.float16, (128,)),
        "s_in": rd("s_in.f32", np.float32, (128,)),
        "s_out": rd("s_out.f32", np.float32, (128,)),
        "bias": rd("bias.f16", np.float16, (128,)),
        "x": rd("x.f16", np.float16, (8, 128)),
        "deploy": rd("deploy.f16", np.float16, (8, 128)),
    }


@pytest.fixture(scope="session")
def moeblk_golden():
    d = ARCH / "qwen3_5_moe"
    x = np.fromfile(d / "moeblk_x.f16", dtype=np.float16).reshape(8, 2048)
    out = np.fromfile(d / "moeblk_out.f16", dtype=np.float16).reshape(8, 2048)
    ids = np.fromfile(d / "moeblk_ids.i64", dtype=np.int64).reshape(8, 8)
    scores = np.fromfile(d / "moeblk_scores.f32", dtype=np.float32).reshape(8, 8)
    return x, out, ids, scores
