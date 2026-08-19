"""The folded-first-token generate_step is token-identical to mlx_lm's stock.

`folded_generate_step` produces the first generated token from the final
prefill chunk's last-position logits instead of running an extra single-token
forward pass.  Because logits at position len(prompt)-1 are a pure causal
function of the preceding positions, the two paths emit the same greedy tokens.
Note: logprob vectors may differ numerically in the default mode — the final
prompt token is computed as the last row of an S-token prefill kernel here vs.
a single-token (S=1) decode kernel in stock; different shapes can use different
reduction orders in MLX.

With exact_first_logprobs=True the prefix (S-1 tokens) is pre-filled and the
last prompt token uses the same S=1 decode as stock, giving bit-identical
logprobs at the cost of the TTFT win.  test_exact_logprobs_match_stock verifies
this with np.array_equal.
"""
import os
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib
from mlx_lm.sample_utils import make_sampler

from escha_mlx.generation import folded_generate_step, install_folded_generate_step
from escha_mlx.loader import load


@pytest.fixture(scope="module")
def model_tokenizer():
    install_folded_generate_step()
    model_path = os.environ.get("ESCHA_MODEL", "~/models/escha-w2")
    m, t = load(Path(model_path).expanduser())
    yield m, t


def _tokens(step, model, tokenizer, prompt, mt, **kw):
    p = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    sampler = make_sampler(temp=0.0)
    return [tok for tok, _lp in step(p, model, max_tokens=mt, sampler=sampler, **kw)]


def _tokens_and_logprobs(step, model, tokenizer, prompt, mt, **kw):
    p = mx.array(tokenizer.encode(prompt, add_special_tokens=True))
    sampler = make_sampler(temp=0.0)
    toks, lps = [], []
    for tok, lp in step(p, model, max_tokens=mt, sampler=sampler, **kw):
        toks.append(tok)
        lps.append(np.array(lp))
    return toks, lps


@pytest.mark.parametrize(
    "prompt,mt",
    [
        ("The capital of France is", 4),
        ("What is 17 multiplied by 23?", 12),
        ("Hello", 3),
        # max_tokens=1 exercises the non-pipelined single-token branch (no
        # decode is primed ahead of the first yield).
        ("The capital of France is", 1),
        ("Hello", 2),  # exactly one pipelined decode after the folded first token
        ("The quick brown fox jumps over the lazy dog and keeps running far away.", 10),
        ("Explain the theory of general relativity in a few sentences please.", 8),
    ],
)
def test_folded_matches_stock(model_tokenizer, prompt, mt):
    m, t = model_tokenizer
    import escha_mlx.generation as G
    stock = G.installed_generate_step.__dict__["_stock"]
    folded = _tokens(folded_generate_step, m, t, prompt, mt)
    reference = _tokens(stock, m, t, prompt, mt)
    assert folded == reference, (prompt, folded, reference)


@pytest.mark.parametrize(
    "prompt,mt",
    [
        ("The capital of France is", 4),
        ("Hello", 3),
        ("Hello", 1),  # single-token prompt: no prefix, direct S=1 decode
        ("Hello", 2),
    ],
)
def test_exact_logprobs_match_stock(model_tokenizer, prompt, mt):
    """With exact_first_logprobs=True every yielded (token, logprob) pair must
    be bit-for-bit identical to the stock generator (same S=1 kernel shape for
    the first token; identical _step for all subsequent tokens)."""
    m, t = model_tokenizer
    import escha_mlx.generation as G
    stock = G.installed_generate_step.__dict__["_stock"]
    folded_toks, folded_lps = _tokens_and_logprobs(
        folded_generate_step, m, t, prompt, mt, exact_first_logprobs=True)
    stock_toks, stock_lps = _tokens_and_logprobs(stock, m, t, prompt, mt)
    assert folded_toks == stock_toks, (prompt, folded_toks, stock_toks)
    for i, (fl, sl) in enumerate(zip(folded_lps, stock_lps)):
        assert np.array_equal(fl.view(np.uint32), sl.view(np.uint32)), \
            f"logprob mismatch at token {i} for prompt {prompt!r}"
