"""Continuous batching adapter for the native Qwen3.8 MTP head.

Each request accepts its own linear prefix or path through the topk=3/depth=3
proposal tree. BatchKVCache remains physically rectangular by right-aligning
the independently committed histories; GDN/conv state is selected from each
request's own final verification node. No request is shortened to the minimum
acceptance of its peers.
"""
from __future__ import annotations

import copy
from typing import Callable, List

import mlx.core as mx

from mlx_lm.generate import (
    BatchResponse,
    BatchGenerator as _BatchGenerator,
    GenerationBatch as _GenerationBatch,
    PromptProcessingBatch as _PromptProcessingBatch,
    _make_cache,
)
from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache

from .mtp import (
    _accepted_tree_paths,
    _propose_tree,
    _target_forward,
    _target_hidden,
    _target_tree_forward,
)


def _right_pad(rows: List[List[int]], width: int) -> mx.array:
    return mx.array([row + [0] * (width - len(row)) for row in rows])


def _reject_rotating_cache(cache) -> None:
    """Fail before tree verification reaches a cache layout it cannot commit."""
    if any(isinstance(item, BatchRotatingKVCache) for item in cache):
        raise ValueError(
            "native MTP does not support rotating KV caches; omit max_kv_size"
        )


def _batch_path_marks(cache):
    """Remember the rectangular attention boundary before verification."""
    return [
        (
            item._idx,
            mx.array(item.offset.tolist()),
            mx.array(item.left_padding.tolist()),
        )
        if item.is_trimmable()
        else None
        for item in cache
    ]


def _commit_batch_paths(cache, marks, chains, trace):
    """Right-align independently selected verification paths in BatchKVCache."""
    from . import gdn_cache

    lengths = [len(chain) for chain in chains]
    max_length = max(lengths)
    padding = [max_length - length for length in lengths]
    B = len(chains)
    for item, mark in zip(cache, marks):
        if not item.is_trimmable():
            continue
        prefix, old_offset, old_left_padding = mark
        key_rows, value_rows = [], []
        for b, (chain, pad) in enumerate(zip(chains, padding)):
            old_keys = item.keys[b : b + 1, :, :prefix, :]
            old_values = item.values[b : b + 1, :, :prefix, :]
            tree_keys = mx.take(
                item.keys[b : b + 1, :, prefix : item._idx, :],
                mx.array(chain, dtype=mx.int32),
                axis=2,
            )
            tree_values = mx.take(
                item.values[b : b + 1, :, prefix : item._idx, :],
                mx.array(chain, dtype=mx.int32),
                axis=2,
            )
            key_pad = mx.zeros(
                (1, old_keys.shape[1], pad, old_keys.shape[3]),
                dtype=old_keys.dtype,
            )
            value_pad = mx.zeros(
                (1, old_values.shape[1], pad, old_values.shape[3]),
                dtype=old_values.dtype,
            )
            key_rows.append(mx.concatenate([key_pad, old_keys, tree_keys], axis=2))
            value_rows.append(
                mx.concatenate([value_pad, old_values, tree_values], axis=2)
            )
        item.keys = mx.concatenate(key_rows, axis=0)
        item.values = mx.concatenate(value_rows, axis=0)
        item._idx = prefix + max_length
        item.offset = old_offset + mx.array(lengths)
        item.left_padding = old_left_padding + mx.array(padding)
    gdn_cache.commit_tree_states(trace, [chain[-1] for chain in chains])


def _right_pad_hidden(rows):
    width = max(row.shape[1] for row in rows)
    padded = [
        mx.pad(row, [(0, 0), (0, width - row.shape[1]), (0, 0)])
        for row in rows
    ]
    return mx.concatenate(padded, axis=0), width


class MTPGenerationBatch(_GenerationBatch):
    """A real batched MTP decode stage compatible with mlx-lm's scheduler."""

    def __init__(
        self,
        *args,
        mtp_prefill_step_size: int = 256,
        mtp_cache=None,
        **kwargs,
    ):
        self.mtp_prefill_step_size = mtp_prefill_step_size
        model = args[0] if args else kwargs.get("model")
        self.mtp_head = getattr(model, "mtp", None)
        if self.mtp_head is None:
            raise ValueError("batched MTP requires a model loaded with load_mtp=True")
        self.mtp_cache = mtp_cache or []
        self._mtp_hidden = None
        self._mtp_logits = None
        self._mtp_bootstrapped = False
        self._last_target_trace = []
        self._last_target_chains = []
        super().__init__(*args, **kwargs)

    @classmethod
    def empty(
        cls,
        model,
        fallback_sampler: Callable,
        *,
        mtp_prefill_step_size: int = 256,
    ):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            uids=[],
            inputs=mx.array([], dtype=mx.uint32),
            prompt_cache=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            state_machines=[],
            mtp_prefill_step_size=mtp_prefill_step_size,
        )

    def _sample(self, logits: mx.array, prefixes: mx.array, tree_parents=None):
        """Apply each request's processors/sampler at every proposed position."""
        B, T, _ = logits.shape
        if not any(self.logits_processors) and not any(self.samplers):
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            samples = self.fallback_sampler(logprobs.reshape(-1, logits.shape[-1]))
            return samples.reshape(B, T), logprobs

        sample_rows, logprob_rows = [], []
        for b in range(B):
            row_samples, row_logprobs = [], []
            sampler = self.samplers[b] or self.fallback_sampler
            for t in range(T):
                row_logits = logits[b : b + 1, t]
                if self.logits_processors[b]:
                    if tree_parents is not None:
                        path, node = [], t
                        parents = tree_parents[b]
                        while node >= 0:
                            path.append(node)
                            node = parents[node]
                        suffix = mx.take(
                            prefixes[b], mx.array(path[::-1], dtype=mx.int32)
                        )
                    else:
                        suffix = prefixes[b, : t + 1]
                    context = mx.concatenate(
                        [self._token_context[b].tokens, suffix]
                    )
                    for processor in self.logits_processors[b]:
                        row_logits = processor(context, row_logits)
                row_logprobs_t = row_logits - mx.logsumexp(
                    row_logits, axis=-1, keepdims=True
                )
                row_samples.append(sampler(row_logprobs_t).reshape(()))
                row_logprobs.append(row_logprobs_t[0])
            sample_rows.append(mx.stack(row_samples))
            logprob_rows.append(mx.stack(row_logprobs))
        return mx.stack(sample_rows), mx.stack(logprob_rows)

    def _prefill_mtp_head(self, last_prompt_tokens: mx.array) -> None:
        """Build the head cache from target prompt hiddens, including cache hits.

        mlx-lm's prompt cache currently stores target states only.  Replaying the
        target body here keeps that public cache format unchanged and makes MTP
        work for both fresh and cached prompts.  The replay is chunked and its
        cache is released immediately after the head has consumed the hiddens.
        """
        last = last_prompt_tokens.tolist()
        prompts = [list(tokens) + [token] for tokens, token in zip(self.tokens, last)]
        target_rows = [prompt[:-1] for prompt in prompts]
        mtp_rows = [prompt[1:] for prompt in prompts]
        lengths = [len(row) for row in target_rows]
        width = max(lengths, default=0)
        B = len(prompts)

        self.mtp_cache = [BatchKVCache([0] * B)]
        if width == 0:
            return

        padding = [width - length for length in lengths]
        target_ids = _right_pad(target_rows, width)
        mtp_ids = _right_pad(mtp_rows, width)
        scratch_cache = _make_cache(self.model, [0] * B, None)
        for item in scratch_cache:
            item.prepare(lengths=lengths, right_padding=padding)
        for item in self.mtp_cache:
            item.prepare(lengths=lengths, right_padding=padding)

        step = self.mtp_prefill_step_size
        for start in range(0, width, step):
            end = min(start + step, width)
            hidden = _target_hidden(
                self.model, target_ids[:, start:end], scratch_cache
            )
            self.mtp_head.forward_hidden(
                mtp_ids[:, start:end], hidden, self.mtp_cache
            )
            mx.eval(
                [item.state for item in scratch_cache],
                [item.state for item in self.mtp_cache],
            )
            mx.clear_cache()

        for item in self.mtp_cache:
            item.finalize()
        mx.eval([item.state for item in self.mtp_cache])

    def _bootstrap(self):
        """Consume the last prompt token and seed the first MTP proposal."""
        self._current_tokens = self._next_tokens
        self._current_logprobs = self._next_logprobs
        inputs = self._current_tokens
        if not self.mtp_cache:
            self._prefill_mtp_head(inputs)

        hidden, logits = _target_forward(self.model, inputs[:, None], self.prompt_cache)
        sampled, logprobs = self._sample(logits[:, -1:, :], inputs[:, None])
        sampled = sampled[:, 0].astype(mx.uint32)
        self._next_tokens = sampled
        self._next_logprobs = list(logprobs[:, 0])

        self._mtp_hidden = self.mtp_head.forward_hidden(
            sampled[:, None], hidden[:, -1:], self.mtp_cache
        )[:, -1:]
        self._mtp_logits = self.mtp_head.logits(self._mtp_hidden)[:, -1, :]
        mx.async_eval(
            self._next_tokens,
            self._next_logprobs,
            self._mtp_hidden,
            self._mtp_logits,
            [item.state for item in self.mtp_cache],
        )

        mx.eval(inputs, self._current_logprobs)
        for stored, token in zip(self.tokens, inputs.tolist()):
            stored.append(token)
        for context, token in zip(self._token_context, inputs):
            context.update_and_fetch(token[None])
        self._mtp_bootstrapped = True
        return inputs.tolist(), self._current_logprobs

    def _replay_mtp_rows(self, token_rows, hidden_rows):
        """Rebuild independently accepted MTP suffixes in one padded batch."""
        lengths = [len(row) for row in token_rows]
        width = max(lengths)
        replay_ids = _right_pad(token_rows, width).astype(mx.uint32)
        replay_hidden, hidden_width = _right_pad_hidden(hidden_rows)
        if hidden_width != width:
            raise RuntimeError("MTP replay token/hidden widths diverged")
        padding = [width - length for length in lengths]
        for item in self.mtp_cache:
            item.prepare(lengths=lengths, right_padding=padding)
        all_hidden = self.mtp_head.forward_hidden(
            replay_ids, replay_hidden, self.mtp_cache
        )
        last_hidden = mx.stack(
            [all_hidden[b, length - 1] for b, length in enumerate(lengths)]
        )[:, None]
        for item in self.mtp_cache:
            item.finalize()
        self._mtp_hidden = last_hidden
        self._mtp_logits = self.mtp_head.logits(last_hidden)[:, -1, :]

    def _step(self):
        if not self._mtp_bootstrapped:
            return self._bootstrap()

        # The previous trace is needed only while ``next()`` turns its accepted
        # burst into responses (notably to extract an exact cache at an early
        # stop).  Release it before building the next verification graph.  A
        # GDN trace is several times the recurrent-cache size because it keeps
        # every tree node; retaining two rounds at once pushes an MTP B=8 server
        # over the Metal working-set limit when the prompt LRU is populated.
        self._last_target_trace = []
        self._last_target_chains = []
        self._current_tokens = self._next_tokens
        self._current_logprobs = self._next_logprobs
        current = self._current_tokens.astype(mx.uint32)
        remaining = [
            limit - used for limit, used in zip(self.max_tokens, self._num_tokens)
        ]
        if max(remaining) > 1:
            return self._tree_step(current)

        # Every request has exactly one pending token left. Commit those tokens
        # to the target cache without paying for an unused vocabulary head or
        # another proposal tree.
        hidden = _target_hidden(self.model, current[:, None], self.prompt_cache)
        mx.eval(hidden, [item.state for item in self.prompt_cache])
        return (
            [[token] for token in current.tolist()],
            [[logprobs] for logprobs in self._current_logprobs],
        )

    def _tree_step(self, current):
        """One fixed topk-3/depth-3/M=4 proposal round for the batch."""
        tree = _propose_tree(
            self.mtp_head,
            self._mtp_hidden,
            self._mtp_logits,
            self.mtp_cache,
        )
        verify_ids = mx.concatenate([current[:, None], tree.tokens], axis=1)
        marks = _batch_path_marks(self.prompt_cache)
        verify_hidden, verify_logits, target_trace = _target_tree_forward(
            self.model, verify_ids, tree.parents, tree.depths, self.prompt_cache
        )
        needs_tree_paths = any(self.logits_processors) or any(self.samplers)
        parent_rows = tree.parents.tolist() if needs_tree_paths else None
        target_tokens, target_logprobs = self._sample(
            verify_logits, verify_ids, parent_rows
        )
        mx.eval(
            verify_hidden,
            target_tokens,
            target_logprobs,
            [item.state for item in self.prompt_cache],
        )

        verify_rows, target_rows, parent_rows, chains = _accepted_tree_paths(
            verify_ids,
            target_tokens,
            parent_rows if parent_rows is not None else tree.parents,
        )

        recurrent = [item for item in self.prompt_cache if not item.is_trimmable()]
        if len(target_trace) != len(recurrent):
            raise RuntimeError("incomplete GDN trace during batched tree verification")
        _commit_batch_paths(self.prompt_cache, marks, chains, target_trace)
        self._last_target_trace = target_trace
        self._last_target_chains = chains

        replay_token_rows, replay_hidden_rows = [], []
        next_tokens, next_logprobs = [], []
        token_rows, logprob_rows = [], []
        for b, chain in enumerate(chains):
            accepted = [verify_rows[b][node] for node in chain[1:]]
            last_node = chain[-1]
            next_token = target_rows[b][last_node]
            replay_token_rows.append(accepted + [next_token])
            replay_hidden_rows.append(mx.take(
                verify_hidden[b : b + 1], mx.array(chain, dtype=mx.int32), axis=1
            ))
            next_tokens.append(next_token)
            next_logprobs.append(target_logprobs[b, last_node])
            token_rows.append([int(current[b].item()), *accepted])
            logprob_rows.append(
                [self._current_logprobs[b]]
                + [target_logprobs[b, parent] for parent in chain[:-1]]
            )

        self._replay_mtp_rows(replay_token_rows, replay_hidden_rows)
        self._next_tokens = mx.array(next_tokens, dtype=mx.uint32)
        self._next_logprobs = next_logprobs
        mx.async_eval(
            self._next_tokens,
            self._next_logprobs,
            self._mtp_hidden,
            self._mtp_logits,
            [item.state for item in self.mtp_cache],
        )
        return token_rows, logprob_rows

    def _finished_target_cache(self, batch_idx: int, position: int, width: int):
        """Extract a cache ending at an early stop inside an accepted burst."""
        extracted = self.extract_cache(batch_idx)
        extra = width - position - 1
        if extra == 0:
            return extracted

        trace_idx = 0
        trace_position = (
            self._last_target_chains[batch_idx][position]
            if self._last_target_chains
            else position
        )
        for source, destination in zip(self.prompt_cache, extracted):
            if source.is_trimmable():
                destination.trim(extra)
                continue
            trace_cache, conv_states, recurrent_states = self._last_target_trace[
                trace_idx
            ]
            if trace_cache is not source:
                raise RuntimeError("GDN trace/cache order changed during extraction")
            destination.state = [
                conv_states[batch_idx : batch_idx + 1, trace_position],
                recurrent_states[batch_idx : batch_idx + 1, trace_position],
            ]
            trace_idx += 1
        return extracted

    def next(self):
        if not self.uids:
            return []

        token_rows, logprob_rows = self._step()
        responses = []
        keep = []
        for batch_idx in range(len(self.uids)):
            width = len(token_rows[batch_idx])
            finished = False
            for position, (token, logprobs) in enumerate(
                zip(token_rows[batch_idx], logprob_rows[batch_idx])
            ):
                self.tokens[batch_idx].append(token)
                self._token_context[batch_idx].update_and_fetch(
                    mx.array([token], dtype=mx.int32)
                )
                self._num_tokens[batch_idx] += 1
                finish_reason = None
                if self._num_tokens[batch_idx] >= self.max_tokens[batch_idx]:
                    finish_reason = "length"
                (
                    self._matcher_states[batch_idx],
                    match_sequence,
                    current_state,
                ) = self.state_machines[batch_idx].match(
                    self._matcher_states[batch_idx], token
                )
                if match_sequence is not None and current_state is None:
                    finish_reason = "stop"

                if finish_reason is not None:
                    responses.append(
                        self.Response(
                            uid=self.uids[batch_idx],
                            token=token,
                            logprobs=logprobs,
                            finish_reason=finish_reason,
                            current_state=current_state,
                            match_sequence=match_sequence,
                            prompt_cache=self._finished_target_cache(
                                batch_idx, position, width
                            ),
                            all_tokens=self.tokens[batch_idx],
                        )
                    )
                    finished = True
                    break

                responses.append(
                    self.Response(
                        uid=self.uids[batch_idx],
                        token=token,
                        logprobs=logprobs,
                        finish_reason=None,
                        current_state=current_state,
                        match_sequence=match_sequence,
                        prompt_cache=None,
                        all_tokens=None,
                    )
                )
            if not finished:
                keep.append(batch_idx)

        if len(keep) < len(self.uids):
            self.filter(keep)
        return responses

    def filter(self, keep):
        super().filter(keep)
        if keep:
            for item in self.mtp_cache:
                item.filter(keep)
            self._mtp_hidden = self._mtp_hidden[keep]
            self._mtp_logits = self._mtp_logits[keep]
        else:
            self.mtp_cache.clear()
            self._mtp_hidden = None
            self._mtp_logits = None

    def extend(self, batch):
        was_empty = not self.uids
        super().extend(batch)
        if was_empty:
            self.mtp_cache = batch.mtp_cache
            self._mtp_hidden = batch._mtp_hidden
            self._mtp_logits = batch._mtp_logits
            self._mtp_bootstrapped = batch._mtp_bootstrapped
        else:
            for current, incoming in zip(self.mtp_cache, batch.mtp_cache):
                current.extend(incoming)
            self._mtp_hidden = mx.concatenate([self._mtp_hidden, batch._mtp_hidden])
            self._mtp_logits = mx.concatenate([self._mtp_logits, batch._mtp_logits])


class MTPPromptProcessingBatch(_PromptProcessingBatch):
    """Prompt stage which hands completed requests to MTPGenerationBatch."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        _reject_rotating_cache(self.prompt_cache)
        self.mtp_head = getattr(self.model, "mtp", None)
        if self.mtp_head is None:
            raise ValueError("batched MTP requires a model loaded with load_mtp=True")
        self.mtp_cache = [BatchKVCache([0] * len(self.uids))]
        self._mtp_pending_hidden = [None] * len(self.uids)
        # Fresh requests can construct the shifted MTP cache while the target
        # prompt is already being evaluated. Existing target-only prompt caches
        # have no hidden-state sidecar, so those requests retain the replay
        # fallback in MTPGenerationBatch._prefill_mtp_head.
        target_has_history = any(
            item.is_trimmable() and getattr(item, "_idx", 0) > 0
            for item in self.prompt_cache
        )
        self._mtp_online_prefill = not target_has_history and not any(self.tokens)

    def _append_mtp(self, token_rows, hidden_rows):
        """Append independently sized shifted pairs to the batched MTP cache."""
        lengths = [len(row) for row in token_rows]
        width = max(lengths, default=0)
        if width == 0:
            return
        padding = [width - length for length in lengths]
        input_ids = _right_pad(token_rows, width).astype(mx.uint32)
        hidden = mx.concatenate(
            [
                mx.pad(row, [(0, 0), (0, width - row.shape[1]), (0, 0)])
                for row in hidden_rows
            ],
            axis=0,
        )
        for item in self.mtp_cache:
            item.prepare(lengths=lengths, right_padding=padding)
        self.mtp_head.forward_hidden(input_ids, hidden, self.mtp_cache)
        mx.eval([item.state for item in self.mtp_cache])
        if max(padding) > 0:
            for item in self.mtp_cache:
                item.finalize()
            mx.eval([item.state for item in self.mtp_cache])

    def _consume_target_hidden(self, token_rows, hidden, lengths):
        """Pair target H(x_t) with x_(t+1), retaining the final unpaired H."""
        pair_tokens, pair_hidden = [], []
        for b, (row, length) in enumerate(zip(token_rows, lengths)):
            row_hidden = hidden[b : b + 1, :length]
            if length == 0:
                pair_tokens.append([])
                pair_hidden.append(row_hidden)
                continue
            pending = self._mtp_pending_hidden[b]
            if pending is None:
                pair_tokens.append(row[1:length])
                pair_hidden.append(row_hidden[:, :-1])
            else:
                pair_tokens.append(row[:length])
                pair_hidden.append(
                    mx.concatenate([pending, row_hidden[:, :-1]], axis=1)
                )
            self._mtp_pending_hidden[b] = row_hidden[:, -1:]
        self._append_mtp(pair_tokens, pair_hidden)

    def prompt(self, tokens):
        """Prefill target and native MTP caches in the same target-body pass."""
        if len(self.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")
        if not tokens:
            return

        token_rows = [list(row) for row in tokens]
        for stored, row in zip(self.tokens, token_rows):
            stored += row
        lengths = [len(row) for row in token_rows]
        max_length = max(lengths)
        padding = [max_length - length for length in lengths]
        if max(padding) > 0:
            inputs = _right_pad(token_rows, max_length)
            for item in self.prompt_cache:
                item.prepare(lengths=lengths, right_padding=padding)
        else:
            inputs = mx.array(token_rows)

        start = 0
        while start < max_length:
            width = min(self.prefill_step_size, max_length - start)
            chunk_lengths = [max(0, min(length - start, width)) for length in lengths]
            hidden = _target_hidden(
                self.model, inputs[:, start : start + width], self.prompt_cache
            )
            chunk_rows = [
                row[start : start + length]
                for row, length in zip(token_rows, chunk_lengths)
            ]
            if self._mtp_online_prefill:
                self._consume_target_hidden(chunk_rows, hidden, chunk_lengths)
            mx.eval(hidden, [item.state for item in self.prompt_cache])
            mx.clear_cache()
            start += width

        if max(padding) > 0:
            for item in self.prompt_cache:
                item.finalize()
            mx.eval([item.state for item in self.prompt_cache])
            mx.clear_cache()

    @classmethod
    def empty(
        cls,
        model,
        fallback_sampler,
        prefill_step_size: int = 2048,
    ):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            prefill_step_size=prefill_step_size,
            uids=[],
            caches=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            state_machines=[],
        )

    def _copy(self):
        copied = super()._copy()
        copied.mtp_head = self.mtp_head
        copied.mtp_cache = copy.deepcopy(self.mtp_cache)
        copied._mtp_pending_hidden = list(self._mtp_pending_hidden)
        copied._mtp_online_prefill = self._mtp_online_prefill
        return copied

    def filter(self, keep):
        if self._mtp_online_prefill:
            if keep:
                for item in self.mtp_cache:
                    item.filter(keep)
                self._mtp_pending_hidden = [
                    self._mtp_pending_hidden[index] for index in keep
                ]
            else:
                self.mtp_cache.clear()
                self._mtp_pending_hidden = []
        super().filter(keep)

    def extend(self, batch):
        online = self._mtp_online_prefill and batch._mtp_online_prefill
        was_empty = not self.uids
        if online:
            if was_empty:
                mtp_cache = batch.mtp_cache
                pending = list(batch._mtp_pending_hidden)
            else:
                for current, incoming in zip(self.mtp_cache, batch.mtp_cache):
                    current.extend(incoming)
                mtp_cache = self.mtp_cache
                pending = self._mtp_pending_hidden + batch._mtp_pending_hidden
        super().extend(batch)
        self._mtp_online_prefill = online
        if online:
            self.mtp_cache = mtp_cache
            self._mtp_pending_hidden = pending
        else:
            self.mtp_cache = []
            self._mtp_pending_hidden = []

    def generate(self, tokens):
        if any(len(row) > 1 for row in tokens):
            self.prompt([row[:-1] for row in tokens])
        last_token = mx.array([row[-1] for row in tokens])
        mtp_cache = None
        if self._mtp_online_prefill:
            pair_tokens, pair_hidden = [], []
            for token, pending in zip(last_token.tolist(), self._mtp_pending_hidden):
                if pending is None:
                    pair_tokens.append([])
                    pair_hidden.append(
                        mx.zeros((1, 0, self.mtp_head.hidden_size), dtype=mx.float16)
                    )
                else:
                    pair_tokens.append([token])
                    pair_hidden.append(pending)
            self._append_mtp(pair_tokens, pair_hidden)
            mtp_cache = self.mtp_cache
        generation = MTPGenerationBatch(
            self.model,
            self.uids,
            last_token,
            self.prompt_cache,
            self.tokens,
            self.samplers,
            self.fallback_sampler,
            self.logits_processors,
            self.state_machines,
            self.max_tokens,
            mtp_prefill_step_size=self.prefill_step_size,
            mtp_cache=mtp_cache,
        )
        self.uids = []
        self.prompt_cache = []
        self.tokens = []
        self.samplers = []
        self.logits_processors = []
        self.max_tokens = []
        self.mtp_cache = []
        self._mtp_pending_hidden = []
        return generation


class MTPBatchGenerator(_BatchGenerator):
    """mlx-lm BatchGenerator with native MTP in its decode stage."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.max_kv_size is not None:
            self.close()
            raise ValueError(
                "native MTP does not support max_kv_size; use a growing KV cache"
            )
        self._prompt_batch = MTPPromptProcessingBatch.empty(
            self.model,
            self.sampler,
            prefill_step_size=self.prefill_step_size,
        )
        self._generation_batch = MTPGenerationBatch.empty(
            self.model,
            self.sampler,
            mtp_prefill_step_size=self.prefill_step_size,
        )

    def _make_batch(self, n: int):
        uids, caches, tokens = [], [], []
        samplers, processors, max_tokens, state_machines = [], [], [], []
        for _ in range(n):
            sequence = self._unprocessed_sequences.popleft()
            uids.append(sequence[0])
            caches.append(sequence[3])
            tokens.append(sequence[4])
            samplers.append(sequence[5])
            processors.append(sequence[6])
            max_tokens.append(sequence[2])
            state_machines.append(sequence[7])
            self._currently_processing.append(
                [sequence[1], 0, sum(len(segment) for segment in sequence[1])]
            )
        return MTPPromptProcessingBatch(
            model=self.model,
            uids=uids,
            caches=caches,
            tokens=tokens,
            prefill_step_size=self.prefill_step_size,
            samplers=samplers,
            fallback_sampler=self.sampler,
            logits_processors=processors,
            state_machines=state_machines,
            max_tokens=max_tokens,
        )

    @property
    def prompt_cache_nbytes(self):
        base = super().prompt_cache_nbytes
        prompt_mtp = sum(item.nbytes for item in self._prompt_batch.mtp_cache)
        generation_mtp = sum(
            item.nbytes for item in self._generation_batch.mtp_cache
        )
        return base + prompt_mtp + generation_mtp


def batch_generate(
    model,
    tokenizer,
    prompts: List[List[int]],
    *,
    prompt_caches=None,
    max_tokens=128,
    return_prompt_caches: bool = False,
    **kwargs,
) -> BatchResponse:
    """Generate a static batch through the same MTP path used by the server."""
    generator = MTPBatchGenerator(
        model,
        stop_tokens=[[token] for token in tokenizer.eos_token_ids],
        **kwargs,
    )
    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * len(prompts)
    uids = generator.insert(prompts, max_tokens, caches=prompt_caches)
    output = {uid: [] for uid in uids}
    output_caches = {}
    with generator.stats() as stats:
        while responses := generator.next_generated():
            for response in responses:
                if response.finish_reason != "stop":
                    output[response.uid].append(response.token)
                if response.finish_reason is not None and return_prompt_caches:
                    output_caches[response.uid] = response.prompt_cache
    generator.close()
    texts = [tokenizer.decode(output[uid]) for uid in uids]
    caches = (
        [output_caches[uid] for uid in uids] if return_prompt_caches else None
    )
    return BatchResponse(texts, stats, caches)
