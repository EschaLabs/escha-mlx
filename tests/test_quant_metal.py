from __future__ import annotations

import numpy as np
import pytest

from .conftest import needs_metal


pytestmark = needs_metal


@pytest.mark.parametrize("rows", [1, 3, 8])
def test_q8_head_is_deterministic_close_and_keeps_top1(monkeypatch, rows):
    import mlx.core as mx

    from escha_mlx import quant

    rng = np.random.default_rng(1200 + rows)
    n, k = 263, 512
    w8 = rng.integers(-128, 128, size=(n, k), dtype=np.int8)
    scale = (rng.random(n) * 0.01 + 1e-3).astype(np.float16)
    packed, scales, biases = quant.pack_q8(w8, scale, 128)
    standard = quant.EschaQ8Linear(packed, scales, biases, 128)
    head = quant.EschaQ8Head(packed, scales, biases, 128)
    x = mx.array(rng.standard_normal((1, rows, k)).astype(np.float16))

    monkeypatch.setenv("ESCHA_MLX_Q8_HEAD", "1")
    reference = standard(x)
    got = head(x)
    again = head(x)
    mx.eval(reference, got, again)

    delta = mx.abs(reference.astype(mx.float32) - got.astype(mx.float32))
    relative = float(delta.max()) / max(float(mx.abs(reference).mean()), 1e-6)
    assert relative < 5e-3
    assert bool((got.view(mx.uint16) == again.view(mx.uint16)).all())
    assert np.array_equal(
        np.array(mx.argmax(got, axis=-1)),
        np.array(mx.argmax(reference, axis=-1)),
    )


def test_q8_head_flag_and_large_rows_use_mlx_exactly(monkeypatch):
    import mlx.core as mx

    from escha_mlx import quant

    rng = np.random.default_rng(1230)
    n, k = 264, 512
    w8 = rng.integers(-128, 128, size=(n, k), dtype=np.int8)
    scale = (rng.random(n) * 0.01 + 1e-3).astype(np.float16)
    packed, scales, biases = quant.pack_q8(w8, scale, 128)
    standard = quant.EschaQ8Linear(packed, scales, biases, 128)
    head = quant.EschaQ8Head(packed, scales, biases, 128)

    for rows, enabled in ((8, False), (9, True)):
        monkeypatch.setenv("ESCHA_MLX_Q8_HEAD", "1" if enabled else "0")
        x = mx.array(rng.standard_normal((rows, k)).astype(np.float16))
        reference, got = standard(x), head(x)
        mx.eval(reference, got)
        assert bool((got.view(mx.uint16) == reference.view(mx.uint16)).all())
