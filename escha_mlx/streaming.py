"""Cheap streaming detokenizers for the serving / dispatch path.

mlx-lm's `BPEStreamingDetokenizer` and `SPMStreamingDetokenizer` rebuild a
token-id -> text map by iterating the ENTIRE tokenizer vocab in `__init__`
(measured ~0.1 s for a 248k-vocab Qwen3 checkpoint, per `.detokenizer`
access).  `stream_generate` / the served endpoint create a fresh detokenizer
per request, so that vocab rescan lands directly in every time-to-first-token
and per-request dispatch cost.

These subclasses build the (read-only) token map once and reuse it across all
instances; each instance still gets its own mutable streaming buffer via
`reset()`, so semantics are identical to the stock classes — only the
one-time map construction is hoisted out of per-request construction.
"""
from __future__ import annotations

from functools import partial

import mlx_lm.tokenizer_utils as tu


class CachedBPEStreamingDetokenizer(tu.BPEStreamingDetokenizer):
    """BPE streaming detokenizer whose tokenmap is built once per tokenizer."""

    def __init__(self, tokenizer):
        tm = getattr(tokenizer, "_escha_tokenmap", None)
        if tm is None:
            vocab = tokenizer.vocab
            tm = [None] * len(vocab)
            for value, tokenid in vocab.items():
                tm[tokenid] = value
            tokenizer._escha_tokenmap = tm
        self.clean_spaces = tokenizer.clean_up_tokenization_spaces
        self.tokenmap = tm
        self.reset()
        self.make_byte_decoder()


class CachedSPMStreamingDetokenizer(tu.SPMStreamingDetokenizer):
    """SPM streaming detokenizer whose tokenmap is built once per tokenizer.

    Preserves the `trim_space` option of the stock partial-constructed class.
    """

    def __init__(self, tokenizer, trim_space=True):
        self.trim_space = trim_space
        self._sep = "\u2581".encode()
        tm = getattr(tokenizer, "_escha_tokenmap_spm", None)
        if tm is None:
            vocab = tokenizer.vocab
            tm = [""] * (max(vocab.values()) + 1)
            for value, tokenid in vocab.items():
                if value.startswith("<0x"):
                    tm[tokenid] = bytes([int(value[3:5], 16)])
                else:
                    tm[tokenid] = value.encode()
            tokenizer._escha_tokenmap_spm = tm
        self.tokenmap = tm
        self.reset()


def install_fast_detokenizer(tokenizer) -> None:
    """Swap the wrapper's detokenizer class for a cached one and prime it.

    Mutates the mlx-lm TokenizerWrapper in place: `_detokenizer_class` becomes
    the cached subclass and the token map is built eagerly (once, at load
    time — outside any measured dispatch window) so that every subsequent
    `.detokenizer` construction is ~free.

    Returns the wrapper unchanged (identity-preserving for callers).  Only the
    BPE/SPM classes carry the vocab-scan cost; the Naive detokenizer is left
    untouched.
    """
    dc = tokenizer._detokenizer_class
    base = dc.func if isinstance(dc, partial) else dc
    if base is tu.BPEStreamingDetokenizer:
        tokenizer._detokenizer_class = CachedBPEStreamingDetokenizer
    elif base is tu.SPMStreamingDetokenizer:
        trim_space = (
            dc.keywords.get("trim_space", True) if isinstance(dc, partial) else True
        )
        if trim_space:
            tokenizer._detokenizer_class = CachedSPMStreamingDetokenizer
        else:
            tokenizer._detokenizer_class = partial(
                CachedSPMStreamingDetokenizer, trim_space=False
            )
    else:
        return
    # Prime: build the token map once now so per-request construction is cheap.
    tokenizer.detokenizer


_ORIGINAL_INFER_THINKING = None


def _cached_infer_thinking(tokenizer):
    """Cache the mlx-lm thinking-token inference per tokenizer.

    `TokenizerWrapper.__init__` calls `_infer_thinking(tokenizer)` on every
    construction, which scans the FULL vocab via `tokenizer.get_vocab()` -- a
    ~51 ms rebuild of a 248k-entry dict for a Qwen3 checkpoint (measured on
    M4 Max).  Any path that re-wraps a tokenizer per request (e.g. the served
    endpoint, or a mlx-lm caller that gets a bare tokenizer) pays that scan in
    before the first token.  The result is a pure function of the (immutable)
    vocab, so caching it on the tokenizer is exact -- identical return, no
    output change.
    """
    cached = getattr(tokenizer, "_escha_think", None)
    if cached is None:
        cached = _ORIGINAL_INFER_THINKING(tokenizer)
        tokenizer._escha_think = cached
    return cached


def install_cached_thinking(tokenizer=None):
    """Monkeypatch `mlx_lm.tokenizer_utils._infer_thinking` with the per-
    tokenizer cached version.  Idempotent; installed once from loader.load so
    subsequent TokenizerWrapper constructions skip the vocab scan.  If a
    tokenizer is given, the cache is primed immediately (so the very first
    dispatch does not pay the one-time vocab scan)."""
    global _ORIGINAL_INFER_THINKING
    import mlx_lm.tokenizer_utils as tu

    if _ORIGINAL_INFER_THINKING is None:
        _ORIGINAL_INFER_THINKING = tu._infer_thinking
        tu._infer_thinking = _cached_infer_thinking
    if tokenizer is not None:
        _cached_infer_thinking(tokenizer)

