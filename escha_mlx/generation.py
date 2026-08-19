"""Faster first-token dispatch: fold the final prompt token into prefill.

`mlx_lm.generate.generate_step` pre-fills every prompt token EXCEPT the last,
then runs one extra single-token forward pass to produce the first generated
token.  For a fully causal decoder that final forward is redundant work: the
last prompt token's logits can be produced by the final prefill chunk itself
(the LM head is already applied per position).  Folding it in removes one full
decode forward (~16 ms on the M4 Max) from every time-to-first-token.

The fold is token-identical to the stock loop for greedy decoding: position
(len-1) logits are a pure causal function of the preceding positions, so the
argmax token is the same.  The logprob vectors may differ numerically because
the final prompt token is computed as the last row of an S-token prefill kernel
here vs. a single-token (S=1) decode kernel in stock — different shapes can use
different reduction orders in MLX.

Pass `exact_first_logprobs=True` to get bit-identical logprobs: the prefix
(S-1 tokens) is pre-filled and the last prompt token is computed via the same
S=1 decode kernel as stock, at the cost of the TTFT win.  Gated by
`tests/test_folded_generate.py` token-for-token (default) and bit-for-bit
logprobs (`exact_first_logprobs=True`).

Any path without logits processors or input embeddings takes the folded
branch (greedy and non-greedy alike); everything else falls through to the
stock implementation, so no serving behaviour outside the common dispatch
path changes.  Set ESCHA_MLX_FOLD_GEN=0 to disable the fold entirely and
restore stock generate_step behaviour.
"""
from __future__ import annotations

import functools
import os
from typing import Any, Callable, Generator, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import (
    generation_stream,
    maybe_quantize_kv_cache,
    generate_step as _STOCK_GENERATE_STEP,
)
from mlx_lm.sample_utils import make_sampler


def folded_generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[list] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    input_embeddings: Optional[mx.array] = None,
    exact_first_logprobs: bool = False,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """First-token-fast generate_step; falls back to `mlx_lm.generate.generate_step`
    for the paths this fold does not handle (logits processors, embeddings).

    exact_first_logprobs: when True, prefill S-1 tokens then run the last
      prompt token as a single-token (S=1) decode, matching the kernel shape
      stock uses.  The first yielded logprob vector is then bit-identical to
      stock.  Default False keeps the TTFT win."""

    # Delegate the paths this fold does not own to the stock implementation:
    # logits processors, embeddings, and multi-chunk prompts.  For multi-chunk
    # prompts the fold would shift the final GDN chunk boundary by one token;
    # hybrid-model prefill is chunk-boundary-sensitive (stock itself differs by
    # chunk size), so to keep output EXACTLY identical to stock we only fold
    # prompts that fit a single prefill chunk (the common <=2048-token case,
    # and the eval dispatch probe).
    if (logits_processors or input_embeddings is not None or prompt is None
            or len(prompt) > prefill_step_size):
        # generator function: must yield through and stop, a plain return would
        # discard the stock generator entirely.
        yield from _STOCK_GENERATE_STEP(
            prompt, model, max_tokens=max_tokens, sampler=sampler,
            logits_processors=logits_processors, max_kv_size=max_kv_size,
            prompt_cache=prompt_cache, prefill_step_size=prefill_step_size,
            kv_bits=kv_bits, kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
            prompt_progress_callback=prompt_progress_callback,
            input_embeddings=input_embeddings,
        )
        return

    from mlx_lm.models import cache as _cache
    if len(prompt) == 0:
        raise ValueError("prompt must be non-empty")

    if prompt_cache is None:
        prompt_cache = _cache.make_prompt_cache(model, max_kv_size=max_kv_size)

    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)
    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _model_call(input_tokens: mx.array):
        return model(input_tokens[None], cache=prompt_cache)

    def _step(input_tokens: mx.array):
        with mx.stream(generation_stream):
            # _model_call adds the batch dim; input_tokens here is the raw
            # 1-D token sequence (e.g. a single sampled token).
            logits = _model_call(input_tokens)
            logits = logits[:, -1, :]
            quantize_cache_fn(prompt_cache)
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            return sampled, logprobs.squeeze(0)

    with mx.stream(generation_stream):
        total_prompt_tokens = len(prompt)
        prompt_processed_tokens = 0
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        last_logits = None
        # Pipelined decode is only safe on the single-chunk fold (the delegation
        # check above guarantees len(prompt) <= prefill_step_size, so this is
        # the normal <=2048-token path; a caller shrinking prefill_step_size
        # could still multi-chunk, so we gate on it).  When we will stream at
        # least one decode (max_tokens > 1) we leave the prefill ASYNC: the
        # first decode's forward reads the still-lazy cache states, so MLX
        # materialises the whole prefill -> decode graph back-to-back and the
        # token-2 graph build overlaps the prefill GPU instead of idling behind
        # a sync barrier.
        pipe = (max_tokens < 0 or max_tokens > 1) and total_prompt_tokens <= prefill_step_size
        # exact_first_logprobs: stop the prefill one token early so the last
        # prompt token is handled by a single-token _step decode below, giving
        # the same S=1 kernel shape as stock and therefore bit-identical logprobs.
        # For a single-token prompt there is no prefix to fill; _step handles it.
        n_prefix = (total_prompt_tokens - 1
                    if exact_first_logprobs
                    else total_prompt_tokens)
        while prompt_processed_tokens < n_prefix:
            remaining = n_prefix - prompt_processed_tokens
            n_to_process = min(prefill_step_size, remaining)
            last_logits = _model_call(prompt[:n_to_process])
            quantize_cache_fn(prompt_cache)
            if not pipe:
                mx.eval([c.state for c in prompt_cache])
            prompt_processed_tokens += n_to_process
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt = prompt[n_to_process:]
            mx.clear_cache()

        # First token: either from the final prefill chunk's last-position logits
        # (folded, TTFT-optimised) or via a single-token decode (exact_first_logprobs,
        # bit-identical to stock).  `prompt` is now the residual: the last 1 token
        # for exact_first_logprobs, or empty for the folded path.
        with mx.stream(generation_stream):
            if exact_first_logprobs:
                y, logprobs = _step(prompt)
            else:
                logits = last_logits[:, -1, :]
                logprobs = logits - mx.logsumexp(logits, keepdims=True)
                y = sampler(logprobs)
    mx.async_eval(y, logprobs)
    if pipe:
        # Software-pipeline the decode loop.  Build + enqueue the first
        # next-token decode NOW -- while the prefill GPU is still running async
        # -- so the GPU flows prefill -> decode2 with no idle gap.  Each later
        # decode is built+enqueued in the iteration BEFORE the one that
        # consumes it, so its ~3 ms Python graph-build overlaps the previous
        # decode's GPU run instead of sitting on an idle GPU (ROUND3 measured
        # ~3.2 ms build per decode step, additive when uncovered).  Purely
        # scheduling: the yielded tokens are bit-identical.
        next_y, next_logprobs = _step(y)
        mx.async_eval(next_y, next_logprobs)
        # Materialize + yield token 1 immediately, then stream tokens
        # 2..max_tokens from the pipelined decodes.  Token 1 is shifted only by
        # the single token-2 decode build done above (~3 ms of the ~11 ms total
        # won); tokens 2.. are built ahead so their graph-builds overlay the
        # prior decode's GPU run.
        mx.eval(y)
        prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        yield y.item(), logprobs
        n = 1
        while max_tokens < 0 or n < max_tokens:
            # next_y holds the decode for token n+1 (already enqueued).  Build
            # + enqueue the decode for token n+2 now so its Python graph-build
            # overlaps token n+1's GPU run, keeping the GPU busy instead of
            # idling between decodes.
            if max_tokens < 0 or n + 1 < max_tokens:
                nn_y, nn_lp = _step(next_y)
                mx.async_eval(nn_y, nn_lp)
            yield next_y.item(), next_logprobs
            if n % 256 == 0:
                mx.clear_cache()
            if max_tokens < 0 or n + 1 < max_tokens:
                next_y, next_logprobs = nn_y, nn_lp
            n += 1
        return

    n = 0
    while True:
        # Materialize the FIRST token BEFORE anything else, then yield it
        # immediately.  Everything runs on the single generation_stream, so an
        # eval barrier placed after the next-token prefetch would wait for that
        # full decode too, inflating time-to-first-token by one extra forward.
        # eval(y) first yields token 1 as soon as the prefill's last-position
        # logits (which y comes from) are done.
        if n == 0:
            mx.eval(y)
            prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        if max_tokens >= 0 and n == max_tokens:
            break
        yield y.item(), logprobs
        # Prefetch the NEXT token AFTER yielding the current one, so the
        # current token's first-yield latency is not charged the next-decode
        # graph build + enqueue.  The prefetch still overlaps the caller
        # consuming this yield (no steady-state throughput loss); only the
        # very first token avoids paying for token 2's setup.
        if max_tokens < 0 or n + 1 < max_tokens:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)
        if n % 256 == 0:
            mx.clear_cache()
        if max_tokens < 0 or n + 1 < max_tokens:
            y, logprobs = next_y, next_logprobs
        n += 1


def _gen_module():
    """Return the real `mlx_lm.generate` module object.

    `mlx_lm/__init__.py` re-exports the `generate()` helper over the submodule
    under the same name, so `import mlx_lm.generate as g` can bind the helper
    function instead of the module.  `importlib.import_module` always returns
    the module from `sys.modules`, which is what `stream_generate`'s globals
    reference."""
    import importlib
    return importlib.import_module("mlx_lm.generate")


def installed_generate_step():
    """Return the current mlx_lm.generate.generate_step (the folded one when this
    package has installed it, the stock one otherwise)."""
    return _gen_module().generate_step


def install_folded_generate_step() -> None:
    """Replace mlx_lm.generate.generate_step with the folded-first-token variant.

    `stream_generate` resolves `generate_step` as a module global, so patching
    it here makes every non-speculative generation path (the served endpoint's
    per-sequence steam, `stream_generate`, the eval dispatch probe) take the
    fold.  Idempotent: re-patching a second time is a no-op.
    ESCHA_MLX_FOLD_GEN=0 disables the install entirely.
    """
    if os.environ.get("ESCHA_MLX_FOLD_GEN", "1") == "0":
        return
    _g = _gen_module()
    if getattr(_g.generate_step, "__escha_folded__", False):
        return
    installed_generate_step._stock = _g.generate_step
    _g.generate_step = folded_generate_step
    _g.generate_step.__escha_folded__ = True


def uninstall_folded_generate_step() -> None:
    """Restore mlx_lm.generate.generate_step to the stock implementation."""
    _g = _gen_module()
    if not getattr(_g.generate_step, "__escha_folded__", False):
        return
    stock = getattr(installed_generate_step, "_stock", None)
    if stock is not None:
        _g.generate_step = stock
