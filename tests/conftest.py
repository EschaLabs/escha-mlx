from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
DATA = HERE / "data"               # the committed reference goldens
DEFAULT_CKPT = os.environ.get(
    "ESCHA_MODEL", str(Path.home() / "Desktop" / "escha-release-2026-07-16"))


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
    not Path(DEFAULT_CKPT).exists(), reason=f"checkpoint not found: {DEFAULT_CKPT}")


@pytest.fixture(scope="session")
def k2_golden():
    packed = np.fromfile(DATA / "packed_gu_e0_k2.i16", dtype=np.int16).reshape(128, 64, 32)
    expected = np.fromfile(DATA / "expected_gu_e0_k2.f16", dtype=np.float16).reshape(2048, 1024)
    return packed, expected


@pytest.fixture(scope="session")
def k3_golden():
    packed = np.fromfile(DATA / "packed_down_e0_k3.i16", dtype=np.int16).reshape(32, 128, 48)
    expected = np.fromfile(DATA / "expected_down_e0_k3.f16", dtype=np.float16).reshape(512, 2048)
    return packed, expected


@pytest.fixture(scope="session")
def w8a16_golden():
    w8 = np.fromfile(DATA / "w8a16_w8_2048x2048.i8", dtype=np.int8).reshape(2048, 2048)
    scale = np.fromfile(DATA / "w8a16_scale_2048.f32", dtype=np.float32)
    x = np.fromfile(DATA / "w8a16_x_8x2048.f16", dtype=np.float16).reshape(8, 2048)
    want = np.fromfile(DATA / "w8a16_expected_8x2048.f16", dtype=np.float16).reshape(8, 2048)
    return w8, scale, x, want


@pytest.fixture(scope="session")
def moeblk_golden():
    x = np.fromfile(DATA / "moeblk_x.f16", dtype=np.float16).reshape(8, 2048)
    out = np.fromfile(DATA / "moeblk_out.f16", dtype=np.float16).reshape(8, 2048)
    ids = np.fromfile(DATA / "moeblk_ids.i64", dtype=np.int64).reshape(8, 8)
    scores = np.fromfile(DATA / "moeblk_scores.f32", dtype=np.float32).reshape(8, 8)
    return x, out, ids, scores
