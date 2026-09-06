from __future__ import annotations

import json

import numpy as np
import pytest

from .conftest import needs_metal, needs_mlx


TINY_MTP_CONFIG = {
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 256,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "tie_word_embeddings": False,
    },
}


def _target():
    from mlx_lm.models import qwen3_5

    return qwen3_5.Model(qwen3_5.ModelArgs.from_dict(TINY_MTP_CONFIG))


def _write_mtp(path, target):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file

    from escha_mlx.mtp import MTPHead, _NORM_WEIGHTS

    source = MTPHead(TINY_MTP_CONFIG, target)
    weights = {}
    for name, value in tree_flatten(source.parameters()):
        value = np.array(value).astype(np.float16)
        if name in _NORM_WEIGHTS:
            value = (value.astype(np.float32) - 1.0).astype(np.float16)
        weights["mtp." + name] = value
    mtp_dir = path / "mtp"
    mtp_dir.mkdir()
    (mtp_dir / "config.json").write_text(json.dumps(TINY_MTP_CONFIG))
    save_file(weights, str(mtp_dir / "model.safetensors"))
    mx.clear_cache()
    return weights


@needs_mlx
def test_mtp_loads_exact_15_tensors_and_shares_vocab(tmp_path):
    from mlx.utils import tree_flatten

    from escha_mlx.mtp import _NORM_WEIGHTS, load_mtp

    target = _target()
    weights = _write_mtp(tmp_path, target)
    mtp_head = load_mtp(tmp_path, target)

    params = dict(tree_flatten(mtp_head.parameters()))
    assert len(params) == len(weights) == 15
    assert mtp_head._embed_tokens is target.language_model.model.embed_tokens
    assert mtp_head._lm_head is target.language_model.lm_head
    for name, value in params.items():
        expected = weights["mtp." + name]
        if name in _NORM_WEIGHTS:
            expected = (expected.astype(np.float32) + 1.0).astype(np.float16)
        assert np.array_equal(np.array(value), expected), name


@needs_mlx
def test_mtp_forward_advances_single_attention_cache(tmp_path):
    import mlx.core as mx

    from escha_mlx.mtp import load_mtp

    target = _target()
    _write_mtp(tmp_path, target)
    mtp_head = load_mtp(tmp_path, target)
    cache = mtp_head.make_cache()
    tokens = mx.array([[1, 2, 3]], dtype=mx.uint32)
    hidden = mx.zeros((1, 3, 128), dtype=mx.float16)
    logits = mtp_head(tokens, hidden, cache)
    mx.eval(logits, [c.state for c in cache])

    assert logits.shape == (1, 3, 256)
    assert cache[0].offset == 3
    assert np.isfinite(np.array(logits)).all()


@needs_mlx
def test_mtp_generation_matches_target_greedy(tmp_path):
    import mlx.core as mx

    from escha_mlx.mtp import load_mtp, mtp_generate_step

    mx.random.seed(7)
    target = _target()
    target.eval()
    _write_mtp(tmp_path, target)
    mtp_head = load_mtp(tmp_path, target)
    prompt = mx.array([1, 7, 9, 4], dtype=mx.uint32)

    # Reference uses one full-prompt forward followed by one-token decode, the
    # same target-cache boundary used by MTP bootstrap.
    cache = target.make_cache()
    hidden = target.language_model.model(prompt[None], cache=cache)
    logits = target.language_model.lm_head(hidden[:, -1:])
    token = mx.argmax(logits[:, -1], axis=-1).astype(mx.uint32)
    reference = [int(token.item())]
    for _ in range(5):
        hidden = target.language_model.model(token[None], cache=cache)
        logits = target.language_model.lm_head(hidden)
        token = mx.argmax(logits[:, -1], axis=-1).astype(mx.uint32)
        mx.eval(token, [c.state for c in cache])
        reference.append(int(token.item()))

    generated = list(
        mtp_generate_step(
            prompt,
            target,
            mtp_head=mtp_head,
            max_tokens=6,
            prefill_step_size=16,
        )
    )
    assert [token for token, _, _ in generated] == reference
    assert len(generated) == 6


@needs_mlx
def test_mtp_single_token_prompt_matches_target_greedy(tmp_path):
    """An empty shifted prefill must leave the MTP KV cache unallocated safely."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    from escha_mlx.mtp import load_mtp, mtp_generate_step

    mx.random.seed(9)
    target = _target()
    target.eval()
    _write_mtp(tmp_path, target)
    head = load_mtp(tmp_path, target)
    prompt = mx.array([7], dtype=mx.uint32)

    reference = [
        int(token)
        for token, _ in generate_step(prompt, target, max_tokens=4)
    ]
    generated = [
        token
        for token, _, _ in mtp_generate_step(
            prompt, target, mtp_head=head, max_tokens=4
        )
    ]
    assert generated == reference


@needs_mlx
def test_mtp_continuous_batch_matches_target_greedy(tmp_path):
    """Batched MTP keeps every request on the target model's token path."""
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator

    from escha_mlx.mtp import load_mtp
    from escha_mlx.mtp_batch import MTPBatchGenerator

    mx.random.seed(13)
    target = _target()
    target.eval()
    _write_mtp(tmp_path, target)
    target.mtp = load_mtp(tmp_path, target)
    prompts = [[1, 7, 9, 4], [3], [8, 5, 6]]
    limits = [7, 5, 6]

    def collect(generator):
        uids = generator.insert(prompts, max_tokens=limits)
        output = {uid: [] for uid in uids}
        while responses := generator.next_generated():
            for response in responses:
                output[response.uid].append(response.token)
        generator.close()
        return [output[uid] for uid in uids]

    baseline = collect(
        BatchGenerator(
            target,
            completion_batch_size=3,
            prefill_batch_size=3,
            prefill_step_size=16,
        )
    )
    speculative = collect(
        MTPBatchGenerator(
            target,
            completion_batch_size=3,
            prefill_batch_size=3,
            prefill_step_size=16,
        )
    )
    assert speculative == baseline
    assert list(map(len, speculative)) == limits


@needs_mlx
def test_mtp_batch_rejects_rotating_cache_before_generation(tmp_path):
    from escha_mlx.mtp import load_mtp
    from escha_mlx.mtp_batch import MTPBatchGenerator

    target = _target()
    target.eval()
    _write_mtp(tmp_path, target)
    target.mtp = load_mtp(tmp_path, target)

    with pytest.raises(ValueError, match="does not support max_kv_size"):
        MTPBatchGenerator(target, max_kv_size=4)


@needs_mlx
def test_mtp_cache_accounting_includes_online_prompt_cache(tmp_path):
    from mlx_lm.generate import BatchGenerator

    from escha_mlx.mtp import load_mtp
    from escha_mlx.mtp_batch import MTPBatchGenerator

    target = _target()
    target.eval()
    _write_mtp(tmp_path, target)
    target.mtp = load_mtp(tmp_path, target)
    generator = MTPBatchGenerator(target, prefill_step_size=2)
    generator.insert([[1, 2, 3, 4, 5, 6, 7, 8]], max_tokens=[3])
    generator.next()

    target_bytes = BatchGenerator.prompt_cache_nbytes.fget(generator)
    prompt_mtp_bytes = sum(item.nbytes for item in generator._prompt_batch.mtp_cache)
    generation_mtp_bytes = sum(
        item.nbytes for item in generator._generation_batch.mtp_cache
    )
    assert prompt_mtp_bytes > 0
    assert generator.prompt_cache_nbytes == (
        target_bytes + prompt_mtp_bytes + generation_mtp_bytes
    )
    generator.close()


@needs_mlx
def test_fresh_batch_prefill_does_not_replay_target_body(tmp_path, monkeypatch):
    import mlx.core as mx

    from escha_mlx.mtp import load_mtp
    from escha_mlx.mtp_batch import MTPBatchGenerator, MTPGenerationBatch

    mx.random.seed(15)
    target = _target()
    target.eval()
    _write_mtp(tmp_path, target)
    target.mtp = load_mtp(tmp_path, target)

    def reject_replay(self, last_prompt_tokens):
        raise AssertionError("fresh prompt unexpectedly replayed the target body")

    monkeypatch.setattr(MTPGenerationBatch, "_prefill_mtp_head", reject_replay)
    generator = MTPBatchGenerator(
        target, completion_batch_size=2, prefill_batch_size=2,
        prefill_step_size=2,
    )
    generator.insert([[1, 7, 9, 4], [3]], max_tokens=[3, 3])
    assert generator.next_generated()
    generator.close()


@needs_mlx
def test_cached_batch_prompt_uses_replay_fallback(tmp_path, monkeypatch):
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator

    from escha_mlx.mtp import load_mtp
    from escha_mlx.mtp_batch import MTPBatchGenerator, MTPGenerationBatch

    mx.random.seed(16)
    target = _target()
    target.eval()
    _write_mtp(tmp_path, target)
    target.mtp = load_mtp(tmp_path, target)
    prompt, prefix, suffix = [1, 7, 9, 4], [1, 7], [9, 4]

    reference = BatchGenerator(target)
    (reference_uid,) = reference.insert([prompt], max_tokens=[5])
    expected = []
    while responses := reference.next_generated():
        expected.extend(r.token for r in responses if r.uid == reference_uid)
    reference.close()

    cache = target.make_cache()
    target.language_model.model(mx.array([prefix]), cache=cache)
    mx.eval([item.state for item in cache])
    replay_calls = 0
    original = MTPGenerationBatch._prefill_mtp_head

    def count_replay(self, last_prompt_tokens):
        nonlocal replay_calls
        replay_calls += 1
        return original(self, last_prompt_tokens)

    monkeypatch.setattr(MTPGenerationBatch, "_prefill_mtp_head", count_replay)
    generator = MTPBatchGenerator(target)
    (uid,) = generator.insert(
        [suffix], caches=[cache], all_tokens=[prefix], max_tokens=[5]
    )
    actual = []
    while responses := generator.next_generated():
        actual.extend(r.token for r in responses if r.uid == uid)
    generator.close()

    assert replay_calls == 1
    assert actual == expected


@needs_mlx
def test_mtp_continuous_batch_accepts_late_request(tmp_path):
    """A request can join an already-decoding MTP batch."""
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator

    from escha_mlx.mtp import load_mtp
    from escha_mlx.mtp_batch import MTPBatchGenerator

    mx.random.seed(17)
    target = _target()
    target.eval()
    _write_mtp(tmp_path, target)
    target.mtp = load_mtp(tmp_path, target)
    prompts = [[1, 4, 7], [2, 9], [5, 3, 8, 6]]
    limits = [6, 6, 6]

    reference = BatchGenerator(
        target, completion_batch_size=3, prefill_batch_size=3, prefill_step_size=16
    )
    reference_ids = reference.insert(prompts, max_tokens=limits)
    expected = {uid: [] for uid in reference_ids}
    while responses := reference.next_generated():
        for response in responses:
            expected[response.uid].append(response.token)
    reference.close()

    generator = MTPBatchGenerator(
        target,
        completion_batch_size=3,
        prefill_batch_size=2,
        prefill_step_size=16,
    )
    first_ids = generator.insert(prompts[:2], max_tokens=limits[:2])
    got = {uid: [] for uid in first_ids}
    for response in generator.next_generated():
        got[response.uid].append(response.token)
    (late_id,) = generator.insert([prompts[2]], max_tokens=[limits[2]])
    got[late_id] = []
    while responses := generator.next_generated():
        for response in responses:
            got[response.uid].append(response.token)
    generator.close()

    assert [got[uid] for uid in [*first_ids, late_id]] == [
        expected[uid] for uid in reference_ids
    ]


@needs_mlx
def test_mtp_batch_releases_previous_trace_before_next_round(monkeypatch):
    """Only one large GDN tree trace may remain live during verification."""
    import mlx.core as mx

    from escha_mlx.mtp_batch import MTPGenerationBatch

    batch = MTPGenerationBatch.__new__(MTPGenerationBatch)
    batch._mtp_bootstrapped = True
    batch._last_target_trace = [object()]
    batch._last_target_chains = [[0, 1]]
    batch._next_tokens = mx.array([3], dtype=mx.uint32)
    batch._next_logprobs = [mx.array([0.0])]
    batch.max_tokens = [4]
    batch._num_tokens = [0]

    def tree_step(self, current):
        assert self._last_target_trace == []
        assert self._last_target_chains == []
        return [[int(current[0].item())]], [[self._current_logprobs[0]]]

    monkeypatch.setattr(MTPGenerationBatch, "_tree_step", tree_step)
    tokens, _ = batch._step()
    assert tokens == [[3]]


@needs_mlx
def test_mtp_batch_early_stop_extracts_exact_cache_prefix(tmp_path):
    """EOS inside an accepted burst must not save later speculative states."""
    import mlx.core as mx

    from escha_mlx.mtp import load_mtp
    from escha_mlx.mtp_batch import MTPBatchGenerator

    mx.random.seed(19)
    target = _target()
    target.eval()
    _write_mtp(tmp_path, target)
    target.mtp = load_mtp(tmp_path, target)
    # A zero target head makes the tree contain token 0 and lets the stop
    # machine cut an accepted burst back to its first position.
    target.language_model.lm_head.weight = mx.zeros_like(
        target.language_model.lm_head.weight
    )
    prompts = [[1, 7, 9, 4], [3, 2]]
    generator = MTPBatchGenerator(
        target,
        stop_tokens=[[0]],
        completion_batch_size=2,
        prefill_batch_size=2,
        prefill_step_size=16,
    )
    uids = generator.insert(prompts, max_tokens=[8, 8])
    responses = generator.next_generated()
    generator.close()

    assert len(responses) == 2
    by_uid = {response.uid: response for response in responses}
    for uid, prompt in zip(uids, prompts):
        response = by_uid[uid]
        assert response.token == 0
        assert response.finish_reason == "stop"
        assert response.all_tokens == prompt + [0]
        attention_caches = [c for c in response.prompt_cache if c.is_trimmable()]
        assert attention_caches
        assert all(c.offset == len(prompt) + 1 for c in attention_caches)


def test_server_mtp_args_are_consumed():
    from escha_mlx.server import _consume_mtp_args

    argv, enabled = _consume_mtp_args(
        ["server", "--model", "m", "--mtp", "--port", "9"]
    )
    assert argv == ["server", "--model", "m", "--port", "9"]
    assert enabled is True

    argv, enabled = _consume_mtp_args(["server", "--port", "9"])
    assert argv == ["server", "--port", "9"]
    assert enabled is False


def test_server_mtp_prompt_cache_has_default_and_explicit_byte_caps():
    from escha_mlx.server import (
        _MTP_DEFAULT_PROMPT_CACHE_BYTES,
        _append_default_option,
        _install_mtp_prompt_cache_limit,
    )

    class FakeLRU:
        def __init__(self, *, max_size, max_bytes):
            self.max_size = max_size
            self.max_bytes = max_bytes

    class FakeServer:
        LRUPromptCache = FakeLRU

        @staticmethod
        def _parse_size(value):
            units = {"MB": 1_000_000, "GB": 1_000_000_000}
            for suffix, scale in units.items():
                if value.endswith(suffix):
                    return int(float(value[: -len(suffix)]) * scale)
            return int(value)

    defaults = ["server", "--model", "m", "--decode-concurrency=4"]
    assert not _append_default_option(defaults, "--decode-concurrency", "2")
    assert _append_default_option(defaults, "--prompt-concurrency", "2")
    assert defaults[-2:] == ["--prompt-concurrency", "2"]

    argv = ["server", "--model", "m"]
    budget = _install_mtp_prompt_cache_limit(FakeServer, argv)
    cache = FakeServer.LRUPromptCache(10)
    assert budget == _MTP_DEFAULT_PROMPT_CACHE_BYTES
    assert argv[-2:] == ["--prompt-cache-bytes", str(budget)]
    assert cache.max_size == 10
    assert cache.max_bytes == budget

    class ExplicitServer:
        LRUPromptCache = FakeLRU
        _parse_size = staticmethod(FakeServer._parse_size)

    argv = ["server", "--prompt-cache-bytes=1GB"]
    budget = _install_mtp_prompt_cache_limit(ExplicitServer, argv)
    cache = ExplicitServer.LRUPromptCache(3)
    assert budget == 1_000_000_000
    assert argv == ["server", "--prompt-cache-bytes=1GB"]
    assert cache.max_bytes == budget


@needs_mlx
def test_tree_logits_processors_receive_only_ancestor_tokens():
    import mlx.core as mx

    from escha_mlx.mtp_batch import MTPGenerationBatch

    contexts = []

    def processor(context, logits):
        contexts.append(context.tolist())
        return logits

    batch = MTPGenerationBatch.__new__(MTPGenerationBatch)
    batch.logits_processors = [[processor]]
    batch.samplers = [None]
    batch.fallback_sampler = lambda logprobs: mx.argmax(logprobs, axis=-1)
    batch._token_context = [
        type("Context", (), {"tokens": mx.array([1, 2], dtype=mx.uint32)})()
    ]
    prefixes = mx.array([[10, 20, 30, 40]], dtype=mx.uint32)
    parents = [[-1, 0, 0, 1]]

    batch._sample(mx.zeros((1, 4, 8)), prefixes, parents)

    assert contexts == [
        [1, 2, 10],
        [1, 2, 10, 20],
        [1, 2, 10, 30],
        [1, 2, 10, 20, 40],
    ]


@needs_mlx
def test_tree_commit_matches_independent_ar_path():
    import mlx.core as mx

    from escha_mlx.mtp import (
        _commit_tree_cache,
        _mark_cache,
        _target_tree_forward,
    )

    mx.random.seed(11)
    target = _target()
    target.eval()
    prompt = mx.array([[1, 2, 3]], dtype=mx.uint32)
    candidates = mx.array([[4, 5, 6, 7]], dtype=mx.uint32)
    traced_cache = target.make_cache()
    reference_cache = target.make_cache()

    target.language_model.model(prompt, cache=traced_cache)
    target.language_model.model(prompt, cache=reference_cache)
    mark = _mark_cache(traced_cache)
    parents = mx.array([[-1, 0, 0, 1]], dtype=mx.int32)
    depths = mx.array([[0, 1, 1, 2]], dtype=mx.int32)
    traced_hidden, _, trace = _target_tree_forward(
        target, candidates, parents, depths, traced_cache
    )
    mx.eval(traced_hidden, [entry[1:] for entry in trace],
            [c.state for c in traced_cache])

    # Commit the accepted branch 4 -> 5 -> 7, discarding sibling token 6.
    chain = [0, 1, 3]
    _commit_tree_cache(traced_cache, mark, chain, trace)
    reference_hidden = target.language_model.model(
        mx.take(candidates[0], mx.array(chain, dtype=mx.int32))[None],
        cache=reference_cache,
    )
    next_token = mx.array([[8]], dtype=mx.uint32)
    traced_next = target.language_model.model(next_token, cache=traced_cache)
    reference_next = target.language_model.model(next_token, cache=reference_cache)
    mx.eval(
        reference_hidden,
        traced_next,
        reference_next,
        [c.state for c in traced_cache],
        [c.state for c in reference_cache],
    )

    assert np.allclose(
        np.array(traced_hidden[:, 3]), np.array(reference_hidden[:, -1]),
        rtol=2e-3, atol=5e-3,
    )
    assert np.allclose(
        np.array(traced_next), np.array(reference_next), rtol=2e-3, atol=5e-3
    )


@needs_mlx
def test_m4_tree_has_three_topological_draft_nodes(tmp_path):
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache

    from escha_mlx.mtp import _propose_tree, load_mtp

    mx.random.seed(29)
    target = _target()
    target.eval()
    _write_mtp(tmp_path, target)
    head = load_mtp(tmp_path, target)
    cache = [BatchKVCache([0, 0])]
    hidden = mx.zeros((2, 1, 128), dtype=mx.float16)
    seed_hidden = head.forward_hidden(
        mx.array([[7], [11]], dtype=mx.uint32), hidden, cache
    )
    seed_logits = head.logits(seed_hidden)[:, -1]
    tree = _propose_tree(head, seed_hidden, seed_logits, cache)
    mx.eval(tree.tokens, tree.parents, tree.depths)

    assert tree.tokens.shape == (2, 3)
    assert tree.parents.shape == tree.depths.shape == (2, 4)
    assert cache[0].keys.shape[0] == 2
    assert cache[0].offset.tolist() == [1, 1]
    for parents, depths in zip(tree.parents.tolist(), tree.depths.tolist()):
        assert parents[0] == -1 and depths[0] == 0
        assert all(0 <= parents[node] < node for node in range(1, 4))
        assert all(depths[node] == depths[parents[node]] + 1 for node in range(1, 4))
        assert max(depths) <= 3


@needs_mlx
def test_gpu_tree_masks_match_ancestry_and_prefix_padding():
    import mlx.core as mx

    from escha_mlx.mtp import _incremental_tree_mask, _tree_mask

    parents = mx.array(
        [
            [-1, 0, 0, 1, 1, 2, 3, 3],
            [-1, 0, 0, 0, 2, 2, 4, 5],
        ],
        dtype=mx.int32,
    )
    padding = mx.array([1, 3], dtype=mx.int32)

    def expected(start):
        rows = []
        for tree, left in zip(parents.tolist(), padding.tolist()):
            batch = []
            for node in range(start, len(tree)):
                visible = [position >= left for position in range(4)]
                ancestry = [False] * len(tree)
                current = node
                while current >= 0:
                    ancestry[current] = True
                    current = tree[current]
                batch.append(visible + ancestry)
            rows.append(batch)
        return np.array(rows, dtype=np.bool_)[:, None]

    full = _tree_mask(parents, 4, padding)
    incremental = _incremental_tree_mask(parents, 5, 4, padding)
    mx.eval(full, incremental)
    assert np.array_equal(np.array(full), expected(0))
    assert np.array_equal(np.array(incremental), expected(5))


@needs_mlx
def test_target_tree_verifier_matches_independent_ar_paths():
    """Every verified node represents the same target context as its AR path."""
    import mlx.core as mx

    from escha_mlx.mtp import _target_tree_forward

    mx.random.seed(303)
    target = _target()
    target.eval()
    prompt = mx.array([[1, 2, 3]], dtype=mx.uint32)
    verify_ids = mx.array([[4, 5, 6, 7]], dtype=mx.uint32)
    parents = mx.array([[-1, 0, 0, 1]], dtype=mx.int32)
    depths = mx.array([[0, 1, 1, 2]], dtype=mx.int32)
    tree_cache = target.make_cache()
    target.language_model.model(prompt, cache=tree_cache)
    tree_hidden, tree_logits, _ = _target_tree_forward(
        target, verify_ids, parents, depths, tree_cache
    )
    mx.eval(tree_hidden, tree_logits)

    for node in range(verify_ids.shape[1]):
        path = []
        current = node
        while current >= 0:
            path.append(current)
            current = int(parents[0, current].item())
        path.reverse()
        cache = target.make_cache()
        target.language_model.model(prompt, cache=cache)
        hidden = target.language_model.model(
            mx.take(verify_ids[0], mx.array(path, dtype=mx.int32))[None],
            cache=cache,
        )
        logits = target.language_model.lm_head(hidden)
        mx.eval(hidden, logits)

        assert np.allclose(
            np.array(tree_hidden[0, node]),
            np.array(hidden[0, -1]),
            rtol=2e-3,
            atol=5e-3,
        )
        assert np.allclose(
            np.array(tree_logits[0, node]),
            np.array(logits[0, -1]),
            rtol=2e-3,
            atol=5e-3,
        )
        assert int(mx.argmax(tree_logits[0, node]).item()) == int(
            mx.argmax(logits[0, -1]).item()
        )


@needs_mlx
def test_tree_gdn_uses_each_nodes_parent_state():
    import mlx.core as mx

    from escha_mlx import gdn_cache

    mx.random.seed(31)
    B, T, Hk, Hv, Dk, Dv = 2, 8, 2, 4, 32, 8
    q = mx.random.normal((B, T, Hk, Dk)).astype(mx.float16)
    k = mx.random.normal((B, T, Hk, Dk)).astype(mx.float16)
    v = mx.random.normal((B, T, Hv, Dv)).astype(mx.float16)
    a = mx.random.normal((B, T, Hv)).astype(mx.float16)
    b = mx.random.normal((B, T, Hv)).astype(mx.float16)
    A_log = mx.random.normal((Hv,)).astype(mx.float32)
    dt_bias = mx.random.normal((Hv,)).astype(mx.float32)
    state = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.float32)
    parent_rows = [[-1, 0, 0, 0, 1, 1, 2, 4], [-1, 0, 0, 1, 1, 2, 3, 5]]
    parents = mx.array(parent_rows, dtype=mx.int32)
    tree_out, tree_states = gdn_cache.gated_delta_tree_trace_update(
        q, k, v, a, b, A_log, dt_bias, state, parents
    )
    mx.eval(tree_out, tree_states)

    for batch, parent_row in enumerate(parent_rows):
        for node in range(T):
            path, current = [], node
            while current >= 0:
                path.append(current)
                current = parent_row[current]
            path.reverse()
            indices = mx.array(path, dtype=mx.int32)
            out, states = gdn_cache.gated_delta_trace_update(
                mx.take(q[batch : batch + 1], indices, axis=1),
                mx.take(k[batch : batch + 1], indices, axis=1),
                mx.take(v[batch : batch + 1], indices, axis=1),
                mx.take(a[batch : batch + 1], indices, axis=1),
                mx.take(b[batch : batch + 1], indices, axis=1),
                A_log,
                dt_bias,
                state[batch : batch + 1],
            )
            mx.eval(out, states)
            assert np.allclose(
                np.array(tree_out[batch, node]), np.array(out[0, -1]),
                rtol=2e-3, atol=2e-3,
            )
            assert np.allclose(
                np.array(tree_states[batch, node]), np.array(states[0, -1]),
                rtol=2e-5, atol=2e-5,
            )


@needs_metal
def test_tree_conv_window_kernel_matches_topological_reference():
    import mlx.core as mx

    from escha_mlx import gdn_cache

    rng = np.random.default_rng(47)
    B, T, C, n_keep = 2, 8, 384, 3
    qkv = mx.array(rng.standard_normal((B, T, C)).astype(np.float16))
    state = mx.array(rng.standard_normal((B, n_keep, C)).astype(np.float16))
    parents = mx.array(
        [
            [-1, 0, 0, 1, 1, 2, 3, 6],
            [-1, 0, 1, 0, 3, 2, 5, 4],
        ],
        dtype=mx.int32,
    )

    states, windows = [], []
    for t in range(T):
        choices = mx.stack([state, *states], axis=1)
        index = (parents[:, t] + 1).reshape(B, 1, 1, 1)
        parent = mx.take_along_axis(choices, index, axis=1)[:, 0]
        window = mx.concatenate([parent, qkv[:, t : t + 1]], axis=1)
        windows.append(window)
        states.append(mx.contiguous(window[:, 1:]))
    expected_trace = mx.stack(states, axis=1)
    expected_windows = mx.stack(windows, axis=1).reshape(B * T, n_keep + 1, C)

    trace, actual_windows = gdn_cache.tree_conv_windows(qkv, state, parents)
    mx.eval(expected_trace, expected_windows, trace, actual_windows)

    assert np.array_equal(np.array(trace), np.array(expected_trace))
    assert np.array_equal(np.array(actual_windows), np.array(expected_windows))


@needs_metal
@pytest.mark.parametrize("shape", [(1, 5003), (2, 3, 5003)])
def test_mtp_metal_topk_log_probs_matches_mlx(shape):
    import mlx.core as mx

    from escha_mlx.mtp_topk import metal_topk_log_probs

    rng = np.random.default_rng(53)
    logits = mx.array(rng.standard_normal(shape).astype(np.float16))
    actual = [metal_topk_log_probs(logits) for _ in range(8)]
    assert all(result is not None for result in actual)

    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    expected_indices = mx.argpartition(-log_probs, kth=2, axis=-1)[..., :3]
    expected = np.sort(np.array(expected_indices), axis=-1)
    reference_scores = reference_indices = None
    for scores, indices in actual:
        expected_scores = mx.take_along_axis(log_probs, indices, axis=-1)
        mx.eval(scores, indices, expected_scores)
        got_scores = np.array(scores)
        got_indices = np.array(indices)

        assert scores.shape == indices.shape == (*shape[:-1], 3)
        assert scores.dtype == mx.float16
        assert indices.dtype == mx.uint32
        assert np.array_equal(np.sort(got_indices, axis=-1), expected)
        assert np.allclose(
            got_scores, np.array(expected_scores), rtol=0.0, atol=1e-2
        )
        if reference_scores is not None:
            assert np.array_equal(got_scores, reference_scores)
            assert np.array_equal(got_indices, reference_indices)
        reference_scores, reference_indices = got_scores, got_indices


@needs_mlx
def test_mtp_metal_topk_log_probs_falls_back_outside_specialization():
    import mlx.core as mx

    from escha_mlx.mtp_topk import metal_topk_log_probs

    logits = mx.zeros((1, 256), dtype=mx.float16)
    assert metal_topk_log_probs(logits) is None
    assert metal_topk_log_probs(mx.zeros((5003,), dtype=mx.float16)) is None


@needs_mlx
def test_batch_cache_commits_each_request_path_independently():
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache

    from escha_mlx.mtp_batch import _batch_path_marks, _commit_batch_paths

    prefix, width = 3, 8
    cache = BatchKVCache([0, 1])
    base = mx.arange(2 * (prefix + width), dtype=mx.float32).reshape(
        2, 1, prefix + width, 1
    )
    cache.state = (
        base[:, :, :prefix],
        1000 + base[:, :, :prefix],
        mx.array([3, 2]),
        mx.array([0, 1]),
    )
    marks = _batch_path_marks([cache])
    cache.update_and_fetch(
        base[:, :, prefix:], 1000 + base[:, :, prefix:]
    )
    chains = [[0, 2, 5], [0]]
    _commit_batch_paths([cache], marks, chains, [])
    mx.eval(cache.state)

    first, second = cache.extract(0), cache.extract(1)
    expected_first = mx.concatenate(
        [base[0:1, :, :prefix], mx.take(base[0:1, :, prefix:], mx.array(chains[0]), axis=2)],
        axis=2,
    )
    expected_second = mx.concatenate(
        [base[1:2, :, 1:prefix], mx.take(base[1:2, :, prefix:], mx.array(chains[1]), axis=2)],
        axis=2,
    )
    assert np.array_equal(np.array(first.keys), np.array(expected_first))
    assert np.array_equal(np.array(second.keys), np.array(expected_second))
    assert first.offset == 6 and second.offset == 3
    assert cache.offset.tolist() == [6, 3]


@needs_mlx
def test_single_tree_cache_gathers_all_attention_layers_before_one_eval(monkeypatch):
    import mlx.core as mx

    import escha_mlx.mtp as mtp

    class Cache:
        def __init__(self, start):
            self.keys = start + mx.arange(7, dtype=mx.float32).reshape(1, 1, 7, 1)
            self.values = 100 + self.keys
            self.offset = 7

        def is_trimmable(self):
            return True

    caches = [Cache(0), Cache(20)]
    mark = mtp._CacheMark(offsets=[3, 3])
    real_eval = mx.eval
    calls = []

    def counted_eval(*args):
        calls.append(args)
        return real_eval(*args)

    monkeypatch.setattr(mtp.mx, "eval", counted_eval)
    mtp._commit_tree_cache(caches, mark, [0, 2], [])
    real_eval(*[cache.keys for cache in caches], *[cache.values for cache in caches])

    assert len(calls) == 1
    assert [cache.offset for cache in caches] == [5, 5]
    assert caches[0].keys.reshape(-1).tolist()[:5] == [0, 1, 2, 3, 5]
    assert caches[1].keys.reshape(-1).tolist()[:5] == [20, 21, 22, 23, 25]


@needs_mlx
def test_mtp_rejects_incomplete_directory(tmp_path):
    from escha_mlx.mtp import load_mtp

    with pytest.raises(FileNotFoundError, match="MTP requested"):
        load_mtp(tmp_path, _target())
