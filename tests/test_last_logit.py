"""LastPositionHead: skip the LM head on positions nobody reads.

mlx-lm applies lm_head to the whole sequence, so a prefill chunk computes
S x [2048 x 248320] and generation uses one row. This wrapper restricts it to
the final position.

The gates here pin the two things that make it safe:
  * the surviving row is EXACTLY what the full head would have produced at
    position -1 (bit-identical, not close),
  * single-token input (decode) is untouched, so the decode path is unchanged.

And the one thing that makes it dangerous: per-position outputs are gone. That
is a deliberate trade for a generation runtime, gated by ESCHA_MLX_LAST_LOGIT,
and asserted here so nobody discovers it from a wrong eval score.
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import needs_mlx

pytestmark = needs_mlx

B, S, H, V = 2, 7, 32, 19


def _head():
    import mlx.core as mx
    import mlx.nn as nn

    rng = np.random.default_rng(0)
    lin = nn.Linear(H, V, bias=False)
    lin.weight = mx.array(rng.standard_normal((V, H)).astype(np.float16))
    return lin


def test_last_row_is_bit_identical_to_full_head():
    import mlx.core as mx
    from escha_mlx.loader import LastPositionHead

    inner = _head()
    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal((B, S, H)).astype(np.float16))
    mx.eval(x)

    full = inner(x)
    last = LastPositionHead(inner)(x)
    mx.eval(full, last)

    assert last.shape == (B, 1, V)
    a = np.array(full[:, -1, :])
    b = np.array(last[:, 0, :])
    assert np.array_equal(a.view(np.uint16), b.view(np.uint16)), (
        "the retained row must match the full head exactly")


def test_single_token_is_untouched():
    """Decode passes S=1 and must keep the identical shape and values."""
    import mlx.core as mx
    from escha_mlx.loader import LastPositionHead

    inner = _head()
    rng = np.random.default_rng(2)
    x = mx.array(rng.standard_normal((B, 1, H)).astype(np.float16))
    mx.eval(x)
    full, last = inner(x), LastPositionHead(inner)(x)
    mx.eval(full, last)
    assert last.shape == full.shape == (B, 1, V)
    assert np.array_equal(np.array(full).view(np.uint16),
                          np.array(last).view(np.uint16))


def test_two_dim_input_passes_through():
    """Not every caller passes [B, S, H]; a 2-D input must not be sliced."""
    import mlx.core as mx
    from escha_mlx.loader import LastPositionHead

    inner = _head()
    x = mx.array(np.random.default_rng(3).standard_normal((5, H)).astype(np.float16))
    mx.eval(x)
    out = LastPositionHead(inner)(x)
    mx.eval(out)
    assert out.shape == (5, V)


def test_env_toggle():
    import os
    from escha_mlx.loader import use_last_logit

    saved = os.environ.get("ESCHA_MLX_LAST_LOGIT")
    try:
        os.environ.pop("ESCHA_MLX_LAST_LOGIT", None)
        assert use_last_logit() is True            # default on
        os.environ["ESCHA_MLX_LAST_LOGIT"] = "0"
        assert use_last_logit() is False
        os.environ["ESCHA_MLX_LAST_LOGIT"] = "1"
        assert use_last_logit() is True
    finally:
        os.environ.pop("ESCHA_MLX_LAST_LOGIT", None)
        if saved is not None:
            os.environ["ESCHA_MLX_LAST_LOGIT"] = saved


def test_per_position_logits_are_deliberately_lost():
    """Document the trade: multi-token output is [B,1,V], not [B,S,V].

    Anything scoring per position (loglikelihood eval, speculative
    verification) must set ESCHA_MLX_LAST_LOGIT=0. Asserted so the limitation
    is discovered here rather than in a silently wrong eval number.
    """
    import mlx.core as mx
    from escha_mlx.loader import LastPositionHead

    inner = _head()
    x = mx.array(np.zeros((1, 4, H), dtype=np.float16))
    mx.eval(x)
    out = LastPositionHead(inner)(x)
    assert out.shape[1] == 1 and inner(x).shape[1] == 4
