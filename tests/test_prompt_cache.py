"""Prompt-cache safety for the hybrid (GDN + attention) cache lists.

Two invariants, both enforced in escha_mlx code rather than assumed of
mlx-lm:

* ``GDNStateCache.is_trimmable()`` is pinned False (a recurrent state resumes
  only at its exact stored boundary), so the server's supersequence-reuse
  gate can never trim-and-reuse a longer hybrid entry.
* ``LRUPromptCache.fetch_nearest_cache`` never returns an empty remainder
  once ``server._install_exact_hit_guard`` runs.  Upstream answers an exact
  key hit with ``(cache, [])``, which crashes the batched serving path
  (``seq[-1]`` on an empty segment list) and would hand the sequential path
  an empty prompt.  The key is ``prompt + generated`` of a finished request,
  so ordinary continuation traffic reaches it.

The tests drive ``LRUPromptCache`` directly with real cache objects -- no
checkpoint, no Metal, no server process.
"""
from __future__ import annotations

import pytest

from .conftest import needs_mlx

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.cache import KVCache, can_trim_prompt_cache  # noqa: E402
from mlx_lm.models.cache import LRUPromptCache  # noqa: E402

from escha_mlx.gdn_cache import GDNStateCache  # noqa: E402
from escha_mlx.server import _install_exact_hit_guard  # noqa: E402

MODEL_KEY = ("model", None, None)


def _filled_kv(n_tokens: int) -> KVCache:
    kv = KVCache()
    keys = mx.zeros((1, 2, n_tokens, 8), dtype=mx.float16)
    kv.update_and_fetch(keys, keys)
    return kv


def _hybrid_cache(n_tokens: int) -> list:
    """One GDN layer + one attention layer, as both escha models produce."""
    gdn = GDNStateCache(size=2, dtype=mx.float16)
    gdn[0] = mx.zeros((1, 4, 8), dtype=mx.float16)          # conv state
    gdn[1] = mx.zeros((1, 2, 8, 8), dtype=mx.float32)       # recurrent state
    return [gdn, _filled_kv(n_tokens)]


@pytest.fixture()
def guarded_lru() -> LRUPromptCache:
    _install_exact_hit_guard()
    return LRUPromptCache(max_size=10)


@needs_mlx
class TestTrimmability:
    def test_gdn_state_cache_is_not_trimmable(self):
        assert GDNStateCache(size=2, dtype=mx.float16).is_trimmable() is False

    def test_hybrid_cache_list_cannot_trim(self):
        assert can_trim_prompt_cache(_hybrid_cache(4)) is False

    def test_is_trimmable_is_pinned_not_inherited(self):
        # The method must live on the class itself: the invariant may not
        # silently change with the mlx-lm base class.
        assert "is_trimmable" in GDNStateCache.__dict__


@needs_mlx
class TestExactHitGuard:
    def test_install_is_idempotent(self):
        _install_exact_hit_guard()
        first = LRUPromptCache.fetch_nearest_cache
        _install_exact_hit_guard()
        assert LRUPromptCache.fetch_nearest_cache is first

    def test_exact_hit_hybrid_no_prefix_goes_cold(self, guarded_lru):
        tokens = list(range(32))
        guarded_lru.insert_cache(MODEL_KEY, tokens, _hybrid_cache(32))
        cache, rest = guarded_lru.fetch_nearest_cache(MODEL_KEY, tokens)
        assert cache is None
        assert rest == tokens

    def test_exact_hit_hybrid_reuses_stored_shorter_prefix(self, guarded_lru):
        tokens = list(range(32))
        guarded_lru.insert_cache(MODEL_KEY, tokens[:16], _hybrid_cache(16))
        guarded_lru.insert_cache(MODEL_KEY, tokens, _hybrid_cache(32))
        cache, rest = guarded_lru.fetch_nearest_cache(MODEL_KEY, tokens)
        assert cache is not None
        assert rest == tokens[16:]          # never empty, boundary respected
        assert isinstance(cache[0], GDNStateCache)

    def test_exact_hit_hybrid_reuses_len_minus_one_key(self, guarded_lru):
        tokens = list(range(32))
        guarded_lru.insert_cache(MODEL_KEY, tokens[:-1], _hybrid_cache(31))
        guarded_lru.insert_cache(MODEL_KEY, tokens, _hybrid_cache(32))
        cache, rest = guarded_lru.fetch_nearest_cache(MODEL_KEY, tokens)
        assert cache is not None
        assert rest == tokens[-1:]
        assert cache[1].offset == 31        # the len-1 entry, untouched

    def test_exact_hit_trimmable_trims_one_and_refeeds_last(self, guarded_lru):
        tokens = list(range(16))
        guarded_lru.insert_cache(MODEL_KEY, tokens, [_filled_kv(16)])
        cache, rest = guarded_lru.fetch_nearest_cache(MODEL_KEY, tokens)
        assert rest == tokens[-1:]
        assert cache[0].offset == 15        # trimmed copy...
        stored, stored_rest = guarded_lru.fetch_nearest_cache(
            MODEL_KEY, tokens + [99])
        assert stored[0].offset == 16       # ...stored entry untouched
        assert stored_rest == [99]

    def test_single_token_exact_hit_goes_cold(self, guarded_lru):
        guarded_lru.insert_cache(MODEL_KEY, [7], _hybrid_cache(1))
        cache, rest = guarded_lru.fetch_nearest_cache(MODEL_KEY, [7])
        assert cache is None
        assert rest == [7]

    def test_supersequence_of_hybrid_reuses_at_stored_boundary(self, guarded_lru):
        tokens = list(range(32))
        guarded_lru.insert_cache(MODEL_KEY, tokens, _hybrid_cache(32))
        cache, rest = guarded_lru.fetch_nearest_cache(MODEL_KEY, tokens + [1, 2])
        # Reuse at exactly the stored boundary is the one legal hybrid reuse.
        assert cache is not None
        assert rest == [1, 2]

    def test_miss_passes_through(self, guarded_lru):
        tokens = list(range(8))
        cache, rest = guarded_lru.fetch_nearest_cache(MODEL_KEY, tokens)
        assert cache is None
        assert rest == tokens

    def test_fetch_preserves_gdn_dtype(self, guarded_lru):
        tokens = list(range(32))
        guarded_lru.insert_cache(MODEL_KEY, tokens, _hybrid_cache(32))
        cache, rest = guarded_lru.fetch_nearest_cache(MODEL_KEY, tokens + [5])
        assert isinstance(cache[0], GDNStateCache)
        assert cache[0].gdn_dtype == mx.float16
