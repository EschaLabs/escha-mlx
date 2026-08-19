"""Verify install_cached_thinking caches mlx-lm's per-tokenizer thinking-token
inference so repeated TokenizerWrapper construction skips the full-vocab scan.

The cached value must be IDENTICAL to the original _infer_thinking result
(measured ~51 ms on a 248k-vocab tokenizer otherwise), the cache must be
per-tokenizer (no cross-tokenizer bleed), and install must be idempotent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx_lm.tokenizer_utils as tu

sys.path.insert(0, str(Path(__file__).parent.parent))

from escha_mlx import streaming


def _fake_tokenizer(vocab):
    class T:
        def __init__(self, v):
            self._v = v
            self.vocab = v  # the shape of the real property this scans
            self.clean_up_tokenization_spaces = True

        def get_vocab(self):
            return self._v

        def encode(self, s, add_special_tokens=False):
            return [0]  # multi-token thinking-mode path never taken in tests

    return T(vocab)


def test_cached_thinking_matches_original_and_caches():
    # Normalise any prior install from another test/import order.
    streaming._ORIGINAL_INFER_THINKING = None
    orig = tu._infer_thinking

    vocab = {" thinking": 1, " response": 2, "a": 3, "b": 4}
    tok = _fake_tokenizer(vocab)

    # The real (original) inference on this tokenizer.
    expected = orig(tok)

    streaming.install_cached_thinking(tok)

    # After install + prime, the cached path must equal the original result.
    got = tu._infer_thinking(tok)
    assert got == expected, (got, expected)

    # Per-tokenizer cache: a second tokenizer must be re-inferred (no bleed).
    tok2 = _fake_tokenizer({" other": 5, "</think>": 2, "x": 6})
    expected2 = orig(tok2)
    assert tu._infer_thinking(tok2) == expected2
    # And the first tokenizer still serves from its own cache.
    assert tu._infer_thinking(tok) == expected

    # Idempotent install.
    streaming.install_cached_thinking(tok)
    assert tu._infer_thinking(tok) == expected

    # Un-prime for other tests in this process.
    streaming._ORIGINAL_INFER_THINKING = None
    tu._infer_thinking = orig
