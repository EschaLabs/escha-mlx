"""Fused gather+scale+Hadamard+scale+cast kernel.

Replaces ~5 MLX ops, each materialising an [m, IC] f32 tensor, with one kernel
that keeps everything in threadgroup memory. Measured 15.8% of prefill and 11.3%
of decode was the rin stage alone (doc §16.2).

The fused and native op-chain paths use the same radix-2 butterfly order. The
fused path must therefore be bit-identical at its final f16 output; the separate
NumPy test keeps the butterfly's mathematical result tied to ref.h128.
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


def _output_case(m, OC, E, seed):
    import mlx.core as mx
    rng = np.random.default_rng(seed)
    mid = mx.array(rng.standard_normal((m, OC)).astype(np.float32))
    rout = mx.array(rng.standard_normal((E, OC)).astype(np.float32))
    re = mx.array(rng.integers(0, E, size=m).astype(np.int32))
    mx.eval(mid, rout, re)
    return mid, rout, re


@pytest.mark.parametrize("m,IC,E", [(8, 512, 256), (96, 2048, 64),
                                    (300, 1024, 128), (2048, 2048, 256)])
def test_fused_matches_native_chain_bit_exact(m, IC, E):
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
    # The fused kernel and MLX native transform use the same butterfly order.
    nd = int((np.array(got).view(np.uint16) != np.array(want).view(np.uint16)).sum())
    assert nd == 0, f"{nd}/{a.size} f16 elements differ"


@pytest.mark.parametrize("m,IC,E", [(96, 2048, 64), (300, 1024, 128)])
def test_fused_is_bit_reproducible(m, IC, E):
    """Repeated fused evaluations must produce identical output bits."""
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


@pytest.mark.parametrize("m,OC,E", [(8, 1024, 256), (96, 2048, 64),
                                    (300, 1024, 128), (2048, 2048, 256)])
def test_fused_output_matches_native_chain_bit_exact(m, OC, E):
    import mlx.core as mx
    from escha_mlx import moe, msl, ref

    mid, rout, re = _output_case(m, OC, E, m * 11 + OC)
    got = msl.scaled_had_out(mid, rout, re, ref.RS)
    want = (moe.had_blocks(mid) * ref.RS * rout[re]).astype(mx.float16)
    mx.eval(got, want)
    a = np.array(got)
    b = np.array(want)
    nd = int((a.view(np.uint16) != b.view(np.uint16)).sum())
    assert nd == 0, f"{nd}/{a.size} f16 elements differ"


@pytest.mark.parametrize("m,OC,E", [(96, 1024, 64), (300, 2048, 128)])
def test_fused_output_is_bit_reproducible(m, OC, E):
    import mlx.core as mx
    from escha_mlx import msl, ref

    mid, rout, re = _output_case(m, OC, E, 8675309)
    ref_out = None
    for _ in range(16):
        out = msl.scaled_had_out(mid, rout, re, ref.RS)
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
