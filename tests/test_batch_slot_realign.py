"""Slot-realign guard for GenerationBatch.filter (see server.py).

Upstream reindexes per-request samplers/logits_processors only when
``any(...)`` is truthy, so an all-falsy processor list keeps a stale length
when a request finishes and the next ``extend()`` appends positionally —
misaligning processors (``logit_bias``, ``repetition_penalty``) with uids.
Exercised against the real mlx-lm GenerationBatch, built structurally with
no model forward.
"""
from __future__ import annotations

import pytest

from .conftest import needs_mlx

mx = pytest.importorskip("mlx.core")

from escha_mlx.server import _install_slot_realign_guard  # noqa: E402


def _bare_batch(n: int, processors):
    from mlx_lm.generate import GenerationBatch
    b = GenerationBatch.__new__(GenerationBatch)
    b.uids = list(range(n))
    b.prompt_cache = []
    b.tokens = [[i] for i in range(n)]
    b.samplers = [None] * n
    b.logits_processors = processors
    b.max_tokens = [8] * n
    b.state_machines = [object()] * n
    b._current_tokens = None
    b._current_logprobs = None
    b._next_tokens = mx.zeros((n,), dtype=mx.int32)
    b._next_logprobs = [None] * n
    b._token_context = [None] * n
    b._num_tokens = [0] * n
    b._matcher_states = [None] * n
    return b


@needs_mlx
class TestSlotRealignGuard:
    def test_filter_realigns_all_falsy_processor_list(self):
        _install_slot_realign_guard()
        b = _bare_batch(2, [[], []])
        b.filter([1])
        assert len(b.uids) == 1
        assert len(b.logits_processors) == 1      # stale length without guard
        assert len(b.samplers) == 1

    def test_extend_after_filter_keeps_alignment(self):
        _install_slot_realign_guard()
        b = _bare_batch(2, [[], []])
        b.filter([1])
        proc = object()
        other = _bare_batch(1, [[proc]])
        b.extend(other)
        # the new request's processor must sit at ITS index, not one past it
        assert len(b.logits_processors) == len(b.uids) == 2
        assert b.logits_processors[-1] == [proc]

    def test_truthy_lists_still_reindex_normally(self):
        _install_slot_realign_guard()
        p0, p1 = object(), object()
        b = _bare_batch(2, [[p0], [p1]])
        b.filter([1])
        assert b.logits_processors == [[p1]]

    def test_install_is_idempotent(self):
        from mlx_lm.generate import GenerationBatch
        _install_slot_realign_guard()
        first = GenerationBatch.filter
        _install_slot_realign_guard()
        assert GenerationBatch.filter is first
