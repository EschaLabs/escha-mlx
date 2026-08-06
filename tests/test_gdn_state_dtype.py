"""GDNStateCache: cast the recurrent slot only, and nothing else.

These are the cheap structural gates. The numerical question -- how far an f16
recurrent state drifts from f32 over a long decode -- cannot be answered without
the real model and lives in bench/sweep_gdn_state.py.

What matters structurally:
  * the first recurrence starts from zeros in Metal registers and emits the
    configured storage dtype directly, so no full-sized f32 state is allocated,
  * slot 1 (the 2.1 MB/layer recurrent state) is cast; slot 0 (the ~50 KB conv
    state) is NOT -- it is not the f32-accumulated quantity and casting it buys
    nothing while risking the conv window,
  * requesting f32 is a genuine no-op: `install` leaves mlx-lm's own ArraysCache
    in place rather than routing through an identity cast, so the revert path is
    exactly the original code,
  * subclass-producing methods (extract) keep the dtype -- ArraysCache.extract
    hardcodes its own class, which would silently revert to f32 on any server
    path that splits a batch.
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import needs_metal, needs_mlx

pytestmark = needs_mlx


def test_state_dtype_env_parsing():
    import os
    import mlx.core as mx
    from escha_mlx import gdn_cache

    saved = os.environ.get("ESCHA_MLX_GDN_STATE")
    try:
        os.environ.pop("ESCHA_MLX_GDN_STATE", None)
        assert gdn_cache.state_dtype() == mx.float16       # measured default
        for v, want in [("fp16", mx.float16), ("float16", mx.float16),
                        ("bf16", mx.bfloat16), ("fp32", mx.float32),
                        ("FP16", mx.float16)]:
            os.environ["ESCHA_MLX_GDN_STATE"] = v
            assert gdn_cache.state_dtype() == want, v
        os.environ["ESCHA_MLX_GDN_STATE"] = "int8"
        with pytest.raises(ValueError):
            gdn_cache.state_dtype()
    finally:
        os.environ.pop("ESCHA_MLX_GDN_STATE", None)
        if saved is not None:
            os.environ["ESCHA_MLX_GDN_STATE"] = saved


@pytest.mark.parametrize("dt_name", ["fp16", "bf16"])
def test_only_recurrent_slot_is_cast(dt_name):
    import mlx.core as mx
    from escha_mlx import gdn_cache

    dt = gdn_cache._DTYPES[dt_name]
    c = gdn_cache.GDNStateCache(size=2, dtype=dt)
    conv = mx.zeros((1, 3, 512), dtype=mx.float32)
    state = mx.zeros((1, 32, 128, 128), dtype=mx.float32)
    c[0] = conv
    c[1] = state
    assert c[0].dtype == mx.float32, "conv state must not be cast"
    assert c[1].dtype == dt


def test_f32_is_a_true_noop():
    """Requesting f32 must not replace make_cache at all."""
    import mlx.core as mx
    from escha_mlx import gdn_cache

    class FakeLM:
        class model:
            layers = []
        def make_cache(self):
            return "ORIGINAL"

    class FakeModel:
        language_model = FakeLM()
        def make_cache(self):
            return "ORIGINAL"

    m = FakeModel()
    dt = gdn_cache.install(m, dtype=mx.float32)
    assert dt == mx.float32
    # install() overrides by assigning an INSTANCE attribute; the no-op path
    # must leave none behind. (Comparing bound methods by identity would not
    # work -- attribute access builds a fresh binding every time.)
    assert "make_cache" not in m.__dict__
    assert "make_cache" not in m.language_model.__dict__
    assert m.make_cache() == "ORIGINAL"


def test_extract_preserves_dtype():
    import mlx.core as mx
    from escha_mlx import gdn_cache

    c = gdn_cache.GDNStateCache(size=2, dtype=mx.float16)
    c[0] = mx.zeros((4, 3, 64), dtype=mx.float32)
    c[1] = mx.zeros((4, 8, 16, 16), dtype=mx.float32)
    sub = c.extract(2)
    assert isinstance(sub, gdn_cache.GDNStateCache)
    assert sub.gdn_dtype == mx.float16
    assert sub[1].dtype == mx.float16
    assert sub[1].shape == (1, 8, 16, 16)
    # a subsequent write through the extracted cache must still cast
    sub[1] = mx.zeros((1, 8, 16, 16), dtype=mx.float32)
    assert sub[1].dtype == mx.float16


def test_cast_is_idempotent_and_value_preserving_within_range():
    """Casting must be a plain dtype change, not a rescale."""
    import mlx.core as mx
    from escha_mlx import gdn_cache

    rng = np.random.default_rng(0)
    v = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
    c = gdn_cache.GDNStateCache(size=2, dtype=mx.float16)
    c[1] = mx.array(v)
    got = np.array(c[1].astype(mx.float32))
    assert np.abs(got - v).max() <= 1e-3 * max(np.abs(v).max(), 1e-6)
    # storing the already-cast array again changes nothing
    before = np.array(c[1])
    c[1] = c[1]
    assert np.array_equal(np.array(c[1]), before)


def test_nbytes_reflects_the_saving():
    """The whole point is bytes; assert the cache actually reports fewer."""
    import mlx.core as mx
    from escha_mlx import gdn_cache
    from mlx_lm.models.cache import ArraysCache

    shape = (1, 32, 128, 128)
    ref = ArraysCache(size=2)
    ref[1] = mx.zeros(shape, dtype=mx.float32)
    half = gdn_cache.GDNStateCache(size=2, dtype=mx.float16)
    half[1] = mx.zeros(shape, dtype=mx.float32)
    assert half.nbytes * 2 == ref.nbytes


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("dt_name", ["fp16", "bf16"])
@needs_metal
def test_zero_state_kernel_matches_upstream(masked, dt_name):
    """Register-zero initialization must be bit-identical to a zero buffer."""
    import mlx.core as mx
    from escha_mlx import gdn_cache
    from mlx_lm.models import gated_delta

    rng = np.random.default_rng(7)
    B, T, Hk, Hv, Dk, Dv = 2, 3, 1, 2, 32, 8

    def f16(shape):
        return mx.array(rng.standard_normal(shape).astype(np.float16))

    q = f16((B, T, Hk, Dk))
    k = f16((B, T, Hk, Dk))
    v = f16((B, T, Hv, Dv))
    g = mx.array(rng.uniform(0.8, 1.0, (B, T, Hv)).astype(np.float32))
    beta = mx.array(rng.uniform(0.0, 1.0, (B, T, Hv)).astype(np.float16))
    mask = (
        mx.array([[True, True, False], [True, False, False]])
        if masked else None
    )
    state_type = gdn_cache._DTYPES[dt_name]

    got_y, got_state = gdn_cache.gated_delta_zero_kernel(
        q, k, v, g, beta, state_type, mask)
    want_y, want_state = gated_delta.gated_delta_kernel(
        q, k, v, g, beta,
        mx.zeros((B, Hv, Dv, Dk), dtype=state_type), mask)
    mx.eval(got_y, got_state, want_y, want_state)

    assert got_state.dtype == state_type
    assert np.array_equal(np.array(got_y), np.array(want_y))
    # NumPy cannot represent MLX bfloat16 directly; f32 conversion is exact.
    assert np.array_equal(
        np.array(got_state.astype(mx.float32)),
        np.array(want_state.astype(mx.float32)),
    )


def test_cache_survives_deepcopy():
    """mlx-lm deepcopies the WHOLE prompt cache on every batch split.

    `BatchGenerator.split()` -> `_copy()` -> `copy.deepcopy(self.prompt_cache)`
    runs whenever continuous batching splits a batch — i.e. constantly, under
    exactly the concurrency this runtime exists for. An `mlx.core.Dtype` stored
    on the cache is not picklable, so holding one here took the OpenAI server
    down with `TypeError: cannot pickle 'mlx.core.Dtype' object` on the first
    split. The dtype is therefore stored by NAME and resolved on access.

    This shipped in 0.1.0 because the suite never exercised a server path. The
    assertion is cheap; the outage was not.
    """
    import copy
    import mlx.core as mx
    from escha_mlx import gdn_cache

    for dt in (mx.float16, mx.bfloat16):
        c = gdn_cache.GDNStateCache(size=2, dtype=dt)
        c[0] = mx.zeros((2, 3, 64), dtype=mx.float32)
        c[1] = mx.zeros((2, 8, 16, 16), dtype=mx.float32)
        mx.eval(c[0], c[1])

        d = copy.deepcopy(c)
        assert isinstance(d, gdn_cache.GDNStateCache)
        assert d.gdn_dtype == dt
        assert d[1].dtype == dt and d[1].shape == (2, 8, 16, 16)
        assert d.cache[1] is not c.cache[1], "deepcopy must not alias the state"
        # a write through the copy must still cast
        d[1] = mx.zeros((2, 8, 16, 16), dtype=mx.float32)
        assert d[1].dtype == dt


def test_cache_list_deepcopy_like_the_server():
    """The server deepcopies a LIST mixing GDNStateCache and KVCache."""
    import copy
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    from escha_mlx import gdn_cache

    caches = []
    for i in range(4):
        if i % 2:
            caches.append(KVCache())
        else:
            c = gdn_cache.GDNStateCache(size=2, dtype=mx.float16)
            c[1] = mx.zeros((1, 4, 8, 8), dtype=mx.float32)
            mx.eval(c[1])
            caches.append(c)
    copied = copy.deepcopy(caches)
    assert len(copied) == 4
    assert copied[0].gdn_dtype == mx.float16
    assert copied[0][1].dtype == mx.float16
