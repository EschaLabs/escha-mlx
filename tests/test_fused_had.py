"""Fused gather+scale+Hadamard+scale+cast kernel.

Replaces ~5 MLX ops, each materialising an [m, IC] f32 tensor, with one kernel
that keeps everything in threadgroup memory. Measured 15.8% of prefill and 11.3%
of decode was the rin stage alone (doc §16.2).

It is NOT bit-identical to the op chain: the butterfly sums 128 terms in a
different order than the matmul-by-H. It IS the same f32 arithmetic — no
precision is dropped — and it is held to the same tolerance the matmul itself is
held to against ref.h128, plus reproducibility, which is the property this
runtime actually guarantees.
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import needs_metal

pytestmark = needs_metal


def _case(m, IC, E, seed):
    import mlx.core as mx
    rng = np.random.default_rng(seed)
    rows = mx.array(rng.standard_normal((m, IC)).astype(np.float16))
    rin = mx.array(rng.standard_normal((E, IC)).astype(np.float32))
    re = mx.array(rng.integers(0, E, size=m).astype(np.int32))
    mx.eval(rows, rin, re)
    return rows, rin, re


@pytest.mark.parametrize("m,IC,E", [(8, 512, 256), (96, 2048, 64),
                                    (300, 1024, 128), (2048, 2048, 256)])
def test_fused_matches_op_chain(m, IC, E):
    import mlx.core as mx
    from escha_mlx import moe, msl, ref

    rows, rin, re = _case(m, IC, E, m * 7 + IC)
    got = msl.scaled_had(rows, rin, re, ref.RS)
    want = (moe.had_blocks(rows.astype(mx.float32) * rin[re])
            * ref.RS).astype(mx.float16)
    mx.eval(got, want)
    a = np.array(got).astype(np.float32)
    b = np.array(want).astype(np.float32)
    scale = max(np.abs(b).max(), 1e-6)
    assert np.abs(a - b).max() <= 2e-3 * scale, np.abs(a - b).max() / scale
    # and the f16 outputs should agree almost everywhere
    nd = int((np.array(got).view(np.uint16) != np.array(want).view(np.uint16)).sum())
    assert nd / a.size < 0.01, f"{100*nd/a.size:.2f}% of f16 elements differ"


@pytest.mark.parametrize("m,IC,E", [(96, 2048, 64), (300, 1024, 128)])
def test_fused_is_bit_reproducible(m, IC, E):
    """Determinism is the guarantee; reassociation is allowed, drift is not."""
    import mlx.core as mx
    from escha_mlx import msl, ref

    rows, rin, re = _case(m, IC, E, 4242)
    ref_out = None
    for _ in range(16):
        out = msl.scaled_had(rows, rin, re, ref.RS)
        mx.eval(out)
        a = np.array(out)
        if ref_out is None:
            ref_out = a
        else:
            assert np.array_equal(a.view(np.uint16), ref_out.view(np.uint16))


def test_fused_matches_numpy_reference():
    """Independent check against ref.h128, not just against the op chain."""
    import mlx.core as mx
    from escha_mlx import msl, ref

    m, IC, E = 32, 512, 16
    rows, rin, re = _case(m, IC, E, 11)
    got = np.array(msl.scaled_had(rows, rin, re, ref.RS)).astype(np.float32)
    x = np.array(rows).astype(np.float32) * np.array(rin)[np.array(re)]
    want = (ref.h128(x) * ref.RS).astype(np.float16).astype(np.float32)
    scale = max(np.abs(want).max(), 1e-6)
    assert np.abs(got - want).max() <= 2e-3 * scale


def test_toggle():
    import os
    from escha_mlx import msl
    saved = os.environ.get("ESCHA_MLX_FUSED_HAD")
    try:
        os.environ.pop("ESCHA_MLX_FUSED_HAD", None)
        assert msl.use_fused_had() is True
        os.environ["ESCHA_MLX_FUSED_HAD"] = "0"
        assert msl.use_fused_had() is False
    finally:
        os.environ.pop("ESCHA_MLX_FUSED_HAD", None)
        if saved is not None:
            os.environ["ESCHA_MLX_FUSED_HAD"] = saved
