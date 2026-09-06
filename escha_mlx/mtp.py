"""Qwen3.8 native multi-token-prediction head and speculative generation.

The checkpoint's ``mtp/`` directory is not a standalone language model.  It is
one full-attention Qwen3.5 decoder layer which consumes the target model's
hidden state together with the embedding of the following token.  Its token
embedding and language-model head are shared with the target.

The Apple-Metal path uses a topk=3, depth=3 tree with three selected draft
nodes plus the verified root (M=4 target verification).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Callable, Generator, Iterable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)

_NORM_WEIGHTS = frozenset(
    {
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.k_norm.weight",
        "norm.weight",
    }
)


@dataclasses.dataclass(frozen=True)
class _TreeSpec:
    """Proposal geometry; the shipped policy remains one measured fixed value."""

    topk: int
    depth: int
    draft_nodes: int

    def __post_init__(self):
        candidates = self.topk + max(0, self.depth - 1) * self.topk**2
        if self.topk < 1 or self.depth < 1:
            raise ValueError("MTP tree topk and depth must be positive")
        if not 1 <= self.draft_nodes <= candidates:
            raise ValueError(
                f"MTP tree draft_nodes must be in [1, {candidates}]"
            )


_DEFAULT_TREE_SPEC = _TreeSpec(topk=3, depth=3, draft_nodes=3)


def _unwrapped_head(model) -> nn.Module:
    """Return the target head without the last-position optimization wrapper."""
    head = model.language_model.lm_head
    # Avoid importing loader.LastPositionHead here: loader imports this module
    # lazily when MTP is requested.
    return getattr(head, "inner", head)


def _target_logits(model, hidden: mx.array) -> mx.array:
    lm = model.language_model
    if lm.args.tie_word_embeddings:
        return lm.model.embed_tokens.as_linear(hidden)
    return _unwrapped_head(model)(hidden)


class MTPHead(nn.Module):
    """The native single-layer MTP head shipped inside Qwen3.8.

    Shared vocabulary modules are deliberately installed with
    ``object.__setattr__``.  They remain reachable for forward calls but are not
    registered as MTP parameters, so evaluating or serializing the head
    does not walk the 2.5 GB embedding/head twice.
    """

    def __init__(self, config: dict, target) -> None:
        super().__init__()
        from mlx_lm.models import qwen3_5 as qwen

        text_config = config.get("text_config", config)
        args = qwen.TextModelArgs.from_dict(text_config)
        if text_config.get("mtp_num_hidden_layers", 1) != 1:
            raise ValueError("escha-mlx currently supports exactly one MTP layer")
        if text_config.get("mtp_use_dedicated_embeddings", False):
            raise ValueError("dedicated MTP embeddings are not supported")

        # Layer zero becomes full attention when the interval is one.  Building
        # the layer directly avoids allocating a temporary vocab-sized Qwen
        # model only to replace its embedding and head.
        args = dataclasses.replace(args, num_hidden_layers=1, full_attention_interval=1)
        self.pre_fc_norm_embedding = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        self.layers = [qwen.DecoderLayer(args, 0)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        object.__setattr__(self, "_embed_tokens", target.language_model.model.embed_tokens)
        object.__setattr__(self, "_lm_head", _unwrapped_head(target))
        self.hidden_size = args.hidden_size

    def make_cache(self):
        from mlx_lm.models.cache import KVCache

        return [KVCache()]

    def forward_hidden(
        self,
        input_ids: mx.array,
        target_hidden: mx.array,
        cache=None,
    ) -> mx.array:
        """Advance the MTP layer and return its hidden states, before the head."""
        from mlx_lm.models.base import create_attention_mask

        if input_ids.ndim != 2 or target_hidden.ndim != 3:
            raise ValueError("MTP expects input_ids [B,S] and target_hidden [B,S,H]")
        if input_ids.shape[:2] != target_hidden.shape[:2]:
            raise ValueError(
                f"MTP token/hidden shape mismatch: {input_ids.shape} vs "
                f"{target_hidden.shape}"
            )
        if target_hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"MTP hidden size {target_hidden.shape[-1]} != {self.hidden_size}"
            )

        embedding = self.pre_fc_norm_embedding(self._embed_tokens(input_ids))
        hidden = self.pre_fc_norm_hidden(target_hidden)
        hidden = self.fc(mx.concatenate([embedding, hidden], axis=-1))
        layer_cache = None if cache is None else cache[0]
        mask = create_attention_mask(hidden, layer_cache)
        hidden = self.layers[0](hidden, mask=mask, cache=layer_cache)
        return self.norm(hidden)

    def forward_tree_hidden(
        self,
        input_ids: mx.array,
        target_hidden: mx.array,
        cache,
        parents: mx.array,
        depths: mx.array,
        *,
        start: int,
        prefix: int,
        base_positions: mx.array,
        left_padding: list[int],
    ) -> mx.array:
        """Advance one topological proposal level on a shared MTP KV prefix."""
        embedding = self.pre_fc_norm_embedding(self._embed_tokens(input_ids))
        hidden = self.pre_fc_norm_hidden(target_hidden)
        hidden = self.fc(mx.concatenate([embedding, hidden], axis=-1))
        layer = self.layers[0]
        normed = layer.input_layernorm(hidden)
        mask = _incremental_tree_mask(
            parents, start, prefix, left_padding
        )
        positions = base_positions + depths[:, start:] - 1
        residual = _tree_attention(
            layer.self_attn, normed, mask, cache[0], positions
        )
        post_attention = hidden + residual
        hidden = post_attention + layer.mlp(
            layer.post_attention_layernorm(post_attention)
        )
        return self.norm(hidden)

    def logits(self, hidden: mx.array) -> mx.array:
        return self._lm_head(hidden)

    def __call__(self, input_ids, target_hidden, cache=None):
        return self.logits(self.forward_hidden(input_ids, target_hidden, cache))


def load_mtp(path: str | Path, target) -> MTPHead:
    """Load ``path/mtp`` and share the target model's vocabulary modules."""
    from safetensors import safe_open

    from .loader import resolve_module

    mtp_dir = Path(path) / "mtp"
    config_path = mtp_dir / "config.json"
    weights_path = mtp_dir / "model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"MTP requested but {mtp_dir} is incomplete; expected config.json and "
            "model.safetensors"
        )

    config = json.loads(config_path.read_text())
    mtp_head = MTPHead(config, target)
    parameters = dict(tree_flatten(mtp_head.parameters()))
    expected = set(parameters)
    loaded: set[str] = set()

    with safe_open(str(weights_path), framework="numpy") as f:
        for checkpoint_name in f.keys():
            if not checkpoint_name.startswith("mtp."):
                raise ValueError(f"unexpected non-MTP tensor {checkpoint_name!r}")
            name = checkpoint_name[len("mtp.") :]
            if name not in expected:
                raise ValueError(f"unexpected MTP tensor {checkpoint_name!r}")
            weight = f.get_tensor(checkpoint_name)
            wanted_shape = tuple(parameters[name].shape)
            if tuple(weight.shape) != wanted_shape:
                raise ValueError(
                    f"{checkpoint_name}: shape {weight.shape}, expected {wanted_shape}"
                )
            # Qwen3.5 exports Gemma-style RMSNorm deltas; MLX nn.RMSNorm stores
            # the effective multiplicative weight.
            if name in _NORM_WEIGHTS:
                weight = (weight.astype("float32") + 1.0).astype(weight.dtype)
            owner, attr = resolve_module(mtp_head, name)
            setattr(owner, attr, mx.array(weight))
            loaded.add(name)

    missing = expected - loaded
    if missing:
        raise ValueError(f"incomplete MTP checkpoint, missing {sorted(missing)}")

    mtp_head.eval()
    mx.eval(mtp_head.parameters())
    logger.info("escha_mlx: loaded Qwen3.8 native MTP head (%d tensors)", len(loaded))
    return mtp_head


@dataclasses.dataclass
class _CacheMark:
    """Attention-cache offsets before tree verification."""

    offsets: list[int | None]


def _mark_cache(cache: Iterable) -> _CacheMark:
    return _CacheMark([getattr(item, "offset", None) for item in cache])


def _sample_rows(logits: mx.array, sampler: Callable | None):
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    if sampler is None:
        tokens = mx.argmax(logprobs, axis=-1)
    else:
        tokens = sampler(logprobs)
    return tokens.reshape(-1), logprobs


def _target_forward(model, token_ids: mx.array, cache):
    hidden = model.language_model.model(token_ids, cache=cache)
    return hidden, _target_logits(model, hidden)


def _target_hidden(model, token_ids: mx.array, cache):
    """Target body only; avoids a vocab projection during prefill/replay."""
    return model.language_model.model(token_ids, cache=cache)


def _eval_initialized_cache(cache) -> None:
    """Evaluate cache arrays without passing an unallocated KV state to MLX."""
    states = []
    for item in cache:
        if getattr(item, "keys", None) is None:
            continue
        state = item.state
        if all(value is not None for value in state):
            states.append(state)
    if states:
        mx.eval(states)


def _topk_log_probs_ops(logits: mx.array, topk: int):
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    indices = mx.argpartition(-log_probs, kth=topk - 1, axis=-1)[..., :topk]
    scores = mx.take_along_axis(log_probs, indices, axis=-1)
    return scores, indices.astype(mx.uint32)


def _topk_log_probs(logits: mx.array, topk: int):
    from .mtp_topk import metal_topk_log_probs

    accelerated = metal_topk_log_probs(logits, topk=topk)
    return (
        accelerated
        if accelerated is not None
        else _topk_log_probs_ops(logits, topk)
    )


@dataclasses.dataclass
class _TreeProposal:
    tokens: mx.array       # [1, selected draft nodes]
    parents: mx.array      # [1, root + selected nodes], root parent == -1
    depths: mx.array       # [1, root + selected nodes]


def _propose_tree(
    mtp_head: MTPHead,
    seed_hidden: mx.array,
    seed_logits: mx.array,
    committed_cache,
    spec: _TreeSpec = _DEFAULT_TREE_SPEC,
) -> _TreeProposal:
    """Build the fixed topk-3/depth-3 lattice and select three draft nodes.

    Proposal levels are appended temporarily to the committed MTP cache with a
    tree mask, so all branches share one physical KV prefix. All 3 + 9 + 9 path
    candidates are ranked by cumulative log probability; taking the best nodes
    automatically retains ancestors because every child score is no greater
    than its parent's score.
    """
    topk, depth, num_draft_nodes = spec.topk, spec.depth, spec.draft_nodes

    batch = seed_logits.shape[0]
    scores, tokens = _topk_log_probs(seed_logits, topk)
    all_scores = [scores]
    all_tokens = [tokens]
    all_parents = [mx.full((batch, topk), -1, dtype=mx.int32)]
    all_depths = [mx.ones((batch, topk), dtype=mx.int32)]
    item = committed_cache[0]
    if hasattr(item, "_idx"):
        prefix = item._idx
        base_positions = item.offset[:, None]
        left_padding = item.left_padding
    else:
        prefix = item.offset
        base_positions = mx.full((batch, 1), item.offset, dtype=mx.int32)
        left_padding = [0] * batch

    appended = 0
    try:
        if depth > 1:
            frontier_scores = scores
            frontier_tokens = tokens
            frontier_nodes = mx.broadcast_to(
                mx.arange(topk, dtype=mx.int32), (batch, topk)
            )
            frontier_cache_nodes = frontier_nodes
            processed_parents = all_parents[0]
            processed_depths = all_depths[0]
            seed_rows = mx.broadcast_to(
                seed_hidden[:, -1:, :], (batch, topk, seed_hidden.shape[-1])
            )
            frontier_hidden = mtp_head.forward_tree_hidden(
                frontier_tokens,
                seed_rows,
                committed_cache,
                processed_parents,
                processed_depths,
                start=0,
                prefix=prefix,
                base_positions=base_positions,
                left_padding=left_padding,
            )
            appended += topk
            candidate_base = topk

            for level in range(1, depth):
                next_logits = mtp_head.logits(frontier_hidden)
                child_log_probs, child_tokens = _topk_log_probs(next_logits, topk)
                path_scores = frontier_scores[:, :, None] + child_log_probs
                flat_scores = path_scores.reshape(batch, topk * topk)
                flat_tokens = child_tokens.reshape(batch, topk * topk)
                parents = mx.repeat(frontier_nodes, topk, axis=1)
                depths = mx.full(
                    (batch, topk * topk), level + 1, dtype=mx.int32
                )
                all_scores.append(flat_scores)
                all_tokens.append(flat_tokens)
                all_parents.append(parents)
                all_depths.append(depths)

                if level + 1 < depth:
                    chosen = mx.argpartition(
                        -flat_scores, kth=topk - 1, axis=-1
                    )[:, :topk]
                    parent_rows = chosen // topk
                    frontier_scores = mx.take_along_axis(
                        flat_scores, chosen, axis=1
                    )
                    frontier_tokens = mx.take_along_axis(
                        flat_tokens, chosen, axis=1
                    )
                    hidden_index = parent_rows[:, :, None]
                    hidden_index = mx.broadcast_to(
                        hidden_index,
                        (batch, topk, frontier_hidden.shape[-1]),
                    )
                    parent_hidden = mx.take_along_axis(
                        frontier_hidden, hidden_index, axis=1
                    )
                    cache_parents = mx.take_along_axis(
                        frontier_cache_nodes, parent_rows, axis=1
                    )
                    start = appended
                    processed_parents = mx.concatenate(
                        [processed_parents, cache_parents], axis=1
                    )
                    processed_depths = mx.concatenate(
                        [
                            processed_depths,
                            mx.full(
                                (batch, topk), level + 1, dtype=mx.int32
                            ),
                        ],
                        axis=1,
                    )
                    frontier_hidden = mtp_head.forward_tree_hidden(
                        frontier_tokens,
                        parent_hidden,
                        committed_cache,
                        processed_parents,
                        processed_depths,
                        start=start,
                        prefix=prefix,
                        base_positions=base_positions,
                        left_padding=left_padding,
                    )
                    frontier_cache_nodes = mx.broadcast_to(
                        start + mx.arange(topk, dtype=mx.int32),
                        (batch, topk),
                    )
                    frontier_nodes = candidate_base + chosen.astype(mx.int32)
                    appended += topk
                candidate_base += topk * topk

        candidate_scores = mx.concatenate(all_scores, axis=1)
        candidate_tokens = mx.concatenate(all_tokens, axis=1)
        candidate_parents = mx.concatenate(all_parents, axis=1)
        candidate_depths = mx.concatenate(all_depths, axis=1)
        count = num_draft_nodes
        selected = mx.argpartition(
            -candidate_scores, kth=count - 1, axis=-1
        )[:, :count]
        # Candidate order is depth-major. Restoring that order makes every parent
        # precede its children, which both tree kernels rely on.
        selected = mx.sort(selected, axis=-1)
        selected_tokens = mx.take_along_axis(candidate_tokens, selected, axis=1)
        selected_parents = mx.take_along_axis(candidate_parents, selected, axis=1)
        selected_depths = mx.take_along_axis(candidate_depths, selected, axis=1)

        # Map parents from candidate-lattice coordinates into the compact tree
        # without synchronizing selected indices back to Python. Depth-major
        # ordering plus non-increasing cumulative log probability guarantees
        # that every selected child's parent is also selected.
        parent_matches = selected_parents[:, :, None] == selected[:, None, :]
        compact_parents = mx.argmax(
            parent_matches.astype(mx.int32), axis=-1
        ).astype(mx.int32) + 1
        compact_parents = mx.where(selected_parents < 0, 0, compact_parents)
        tree_parents = mx.concatenate(
            [mx.full((batch, 1), -1, dtype=mx.int32), compact_parents], axis=1
        )
        tree_depths = mx.concatenate(
            [mx.zeros((batch, 1), dtype=mx.int32), selected_depths], axis=1
        )
        # Cache trimming below changes Python offsets, not the arrays captured
        # by this graph. Schedule the proposal first so later MTP-cache writes
        # remain ordered behind it without making the CPU wait here.
        mx.async_eval(selected_tokens, tree_parents, tree_depths)
    finally:
        if appended:
            for cache_item in committed_cache:
                cache_item.trim(appended)

    return _TreeProposal(
        tokens=selected_tokens.astype(mx.uint32),
        parents=tree_parents,
        depths=tree_depths,
    )


def _accepted_tree_paths(verify_ids, target_tokens, parents):
    """Materialize and follow the independently accepted path for each row."""
    node_rows = verify_ids.tolist()
    target_rows = target_tokens.tolist()
    parent_rows = parents if isinstance(parents, list) else parents.tolist()
    chains = []
    for nodes, targets, tree in zip(node_rows, target_rows, parent_rows):
        chain = [0]
        while True:
            parent = chain[-1]
            child = next(
                (
                    node
                    for node in range(1, len(tree))
                    if tree[node] == parent and nodes[node] == targets[parent]
                ),
                None,
            )
            if child is None:
                break
            chain.append(child)
        chains.append(chain)
    return node_rows, target_rows, parent_rows, chains


def _rope_at_positions(rope, x: mx.array, positions: mx.array) -> mx.array:
    """Apply MLX's standard RoPE convention at arbitrary tree positions."""
    dims = rope.dims
    base = rope.base
    scale = rope.scale
    dtype = x.dtype
    half = dims // 2
    freq = base ** (mx.arange(half, dtype=mx.float32) * (2.0 / dims))
    theta = positions[:, None, :, None].astype(mx.float32) * scale / freq
    cos, sin = mx.cos(theta), mx.sin(theta)
    rotated = x[..., :dims].astype(mx.float32)
    if rope.traditional:
        even, odd = rotated[..., ::2], rotated[..., 1::2]
        pair = mx.stack([even * cos - odd * sin, even * sin + odd * cos], axis=-1)
        rotated = pair.reshape(*rotated.shape)
    else:
        first, second = rotated[..., :half], rotated[..., half:]
        rotated = mx.concatenate(
            [first * cos - second * sin, first * sin + second * cos], axis=-1
        )
    return mx.concatenate([rotated.astype(dtype), x[..., dims:]], axis=-1)


def _tree_attention(attention, x, mask, cache, positions):
    """Qwen3.5 full attention with per-node RoPE positions and tree mask."""
    from mlx_lm.models.base import scaled_dot_product_attention

    B, L, _ = x.shape
    q_out = attention.q_proj(x)
    queries, gate = mx.split(
        q_out.reshape(B, L, attention.num_attention_heads, -1), 2, axis=-1
    )
    gate = gate.reshape(B, L, -1)
    if hasattr(attention, "inner"):
        from .dense import project_pair

        keys, values = project_pair(attention.k_proj, attention.v_proj, x)
    else:
        keys, values = attention.k_proj(x), attention.v_proj(x)
    queries = attention.q_norm(queries).transpose(0, 2, 1, 3)
    keys = attention.k_norm(
        keys.reshape(B, L, attention.num_key_value_heads, -1)
    ).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, attention.num_key_value_heads, -1).transpose(
        0, 2, 1, 3
    )
    queries = _rope_at_positions(attention.rope, queries, positions)
    keys = _rope_at_positions(attention.rope, keys, positions)
    keys, values = cache.update_and_fetch(keys, values)
    output = scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=attention.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attention.o_proj(output * mx.sigmoid(gate))


def _tree_ancestry(parents: mx.array) -> mx.array:
    """Return ``[B, node, ancestor]`` visibility without a CPU round-trip."""
    batch, width = parents.shape
    ancestors = mx.arange(width, dtype=mx.int32)[None, None, :]
    current = mx.broadcast_to(
        mx.arange(width, dtype=mx.int32)[None, :], (batch, width)
    )
    visible = mx.zeros((batch, width, width), dtype=mx.bool_)
    # Parent indices are topological, so width hops is a finite upper bound for
    # every valid tree. The loop builds a lazy MLX graph; it does not read data.
    for _ in range(width):
        valid = current >= 0
        visible = visible | (valid[:, :, None] & (current[:, :, None] == ancestors))
        safe = mx.maximum(current, 0)
        parent = mx.take_along_axis(parents, safe, axis=1)
        current = mx.where(valid, parent, -1)
    return visible


def _tree_mask_rows(
    parents: mx.array,
    start: int,
    prefix: int,
    left_padding,
) -> mx.array:
    """Join prefix padding and tree ancestry for query rows ``start:``."""
    batch, width = parents.shape
    query_rows = width - start
    padding = mx.array(left_padding, dtype=mx.int32).reshape(batch, 1, 1)
    prefix_visible = mx.arange(prefix, dtype=mx.int32)[None, None, :] >= padding
    prefix_visible = mx.broadcast_to(
        prefix_visible, (batch, query_rows, prefix)
    )
    ancestry = _tree_ancestry(parents)[:, start:, :]
    return mx.concatenate([prefix_visible, ancestry], axis=-1)[:, None]


def _incremental_tree_mask(
    parents: mx.array,
    start: int,
    prefix: int,
    left_padding,
) -> mx.array:
    """Mask one newly appended proposal level against a shared KV prefix."""
    return _tree_mask_rows(parents, start, prefix, left_padding)


def _tree_mask(parents: mx.array, prefix: int, left_padding) -> mx.array:
    return _tree_mask_rows(parents, 0, prefix, left_padding)


def _target_tree_forward(model, token_ids, parents, depths, cache):
    """Verify a batch of equal-width trees in a single target-body pass."""
    from . import gdn_cache

    text_model = model.language_model.model
    attention_cache = next(item for item in cache if item.is_trimmable())
    if hasattr(attention_cache, "_idx"):
        prefix = attention_cache._idx
        base_positions = attention_cache.offset[:, None]
        left_padding = attention_cache.left_padding
    else:
        prefix = int(attention_cache.offset)
        base_positions = mx.full((token_ids.shape[0], 1), prefix, dtype=mx.int32)
        left_padding = [0] * token_ids.shape[0]
    positions = depths + base_positions
    attention_mask = _tree_mask(parents, prefix, left_padding)
    hidden = text_model.embed_tokens(token_ids)
    trace = []
    for layer, layer_cache in zip(text_model.layers, cache):
        normed = layer.input_layernorm(hidden)
        if layer.is_linear:
            residual, layer_trace = gdn_cache.trace_gdn_forward(
                layer.linear_attn,
                normed,
                cache=layer_cache,
                tree_parents=parents,
            )
            trace.append(layer_trace)
        else:
            residual = _tree_attention(
                layer.self_attn, normed, attention_mask, layer_cache, positions
            )
        post_attention = hidden + residual
        hidden = post_attention + layer.mlp(
            layer.post_attention_layernorm(post_attention)
        )
    hidden = text_model.norm(hidden)
    return hidden, _target_logits(model, hidden), trace


def _commit_tree_cache(cache, mark: _CacheMark, chain: list[int], trace) -> None:
    """Compact selected attention nodes and commit matching recurrent states."""
    from . import gdn_cache

    selected = []
    chain_array = mx.array(chain)
    for index, item in enumerate(cache):
        if not item.is_trimmable():
            continue
        prefix = mark.offsets[index]
        candidate_width = item.offset - prefix
        candidate_keys = item.keys[..., prefix : prefix + candidate_width, :]
        candidate_values = item.values[..., prefix : prefix + candidate_width, :]
        selected_keys = mx.take(candidate_keys, chain_array, axis=2)
        selected_values = mx.take(candidate_values, chain_array, axis=2)
        selected.append((item, prefix, selected_keys, selected_values))
    # Evaluate every gather before any overlapping in-place cache write. One
    # synchronization covers all full-attention layers instead of one per layer.
    if selected:
        mx.eval([(keys, values) for _, _, keys, values in selected])
    for item, prefix, selected_keys, selected_values in selected:
        item.keys[..., prefix : prefix + len(chain), :] = selected_keys
        item.values[..., prefix : prefix + len(chain), :] = selected_values
        item.offset = prefix + len(chain)
    gdn_cache.commit_tree_states(trace, [chain[-1]])


def mtp_generate_step(
    prompt: mx.array,
    model,
    *,
    mtp_head: MTPHead | None = None,
    max_tokens: int = 256,
    sampler: Callable | None = None,
    prefill_step_size: int = 256,
) -> Generator[tuple[int, mx.array, bool], None, None]:
    """Generate with the measured topk-3/depth-3/M=4 MTP proposal tree."""
    if max_tokens <= 0:
        return
    if prompt.ndim != 1 or prompt.size == 0:
        raise ValueError("prompt must be a non-empty one-dimensional token array")
    if prefill_step_size < 1:
        raise ValueError("prefill_step_size must be at least one")

    mtp_head = mtp_head or getattr(model, "mtp", None)
    if mtp_head is None:
        raise ValueError("model has no native MTP head; load it with load_mtp=True")

    target_cache = model.make_cache()
    mtp_cache = mtp_head.make_cache()
    prompt = prompt.astype(mx.uint32)
    prompt_len = int(prompt.size)

    # Target prefill covers the full prompt.  MTP prefill is shifted by one:
    # H(x_t) is paired with embedding(x_{t+1}).  The final H is retained and
    # paired with the first target-generated token below.
    final_target_hidden = None
    for start in range(0, prompt_len, prefill_step_size):
        end = min(start + prefill_step_size, prompt_len)
        target_hidden = _target_hidden(model, prompt[start:end][None], target_cache)
        usable = min(end, prompt_len - 1) - start
        if usable > 0:
            mtp_head.forward_hidden(
                prompt[start + 1 : start + 1 + usable][None],
                target_hidden[:, :usable],
                mtp_cache,
            )
        final_target_hidden = target_hidden[:, -1:]
        mx.eval(final_target_hidden, [c.state for c in target_cache])
        _eval_initialized_cache(mtp_cache)

    first_logits = _target_logits(model, final_target_hidden)[:, -1, :]
    first, first_logprobs = _sample_rows(first_logits, sampler)
    mx.eval(first, first_logprobs)
    current = first.astype(mx.uint32)

    if max_tokens == 1:
        yield int(current.item()), first_logprobs[0], False
        return

    # Processing the verified token once gives the first proposal and keeps the
    # MTP cache one token ahead of the target cache, as in SGLang's MTP path.
    mtp_hidden = mtp_head.forward_hidden(current[None], final_target_hidden, mtp_cache)
    mtp_logits = mtp_head.logits(mtp_hidden)[:, -1, :]
    mx.eval(mtp_hidden, mtp_logits, [c.state for c in mtp_cache])

    produced = 1
    yield int(current.item()), first_logprobs[0], False

    while produced < max_tokens:
        remaining = max_tokens - produced
        if remaining > 1:
            tree = _propose_tree(
                mtp_head,
                mtp_hidden,
                mtp_logits,
                mtp_cache,
            )
            verify_ids = mx.concatenate([current[None], tree.tokens], axis=1)
            target_mark = _mark_cache(target_cache)
            verify_hidden, verify_logits, gdn_trace = _target_tree_forward(
                model, verify_ids, tree.parents, tree.depths, target_cache
            )
            target_tokens, target_logprobs = _sample_rows(
                verify_logits[0], sampler
            )
            mx.eval(
                verify_hidden,
                target_tokens,
                target_logprobs,
                [item.state for item in target_cache],
            )

            node_rows, target_rows, _, chains = _accepted_tree_paths(
                verify_ids, target_tokens[None], tree.parents
            )
            node_tokens = node_rows[0]
            target_ids = target_rows[0]
            chain = chains[0]

            n_recurrent = sum(
                not item.is_trimmable() for item in target_cache
            )
            if len(gdn_trace) != n_recurrent:
                raise RuntimeError("incomplete GDN trace during tree verification")
            _commit_tree_cache(target_cache, target_mark, chain, gdn_trace)

            accepted_ids = [node_tokens[node] for node in chain[1:]]
            last_node = chain[-1]
            new_current_id = target_ids[last_node]
            new_current = mx.array([new_current_id], dtype=mx.uint32)

            replay_ids = mx.array(
                accepted_ids + [new_current_id], dtype=mx.uint32
            )[None]
            replay_hidden = mx.take(
                verify_hidden, mx.array(chain, dtype=mx.int32), axis=1
            )
            mtp_hidden = mtp_head.forward_hidden(
                replay_ids, replay_hidden, mtp_cache
            )[:, -1:]
            mtp_logits = mtp_head.logits(mtp_hidden)[:, -1, :]
            mx.eval(mtp_hidden, mtp_logits, [item.state for item in mtp_cache])

            for child_index, token_id in enumerate(accepted_ids):
                if produced >= max_tokens:
                    return
                produced += 1
                parent_node = chain[child_index]
                yield token_id, target_logprobs[parent_node], True
            if produced >= max_tokens:
                return
            produced += 1
            yield new_current_id, target_logprobs[last_node], False
            current = new_current
            continue

        # Only one output remains. Process the pending current token once to
        # obtain the final target token; building a speculative tree cannot
        # reduce the number of target calls at this boundary.
        verify_hidden, verify_logits = _target_forward(
            model, current[None], target_cache
        )
        target_tokens, target_logprobs = _sample_rows(verify_logits[0], sampler)
        mx.eval(
            verify_hidden,
            target_tokens,
            target_logprobs,
            [item.state for item in target_cache],
        )
        yield int(target_tokens[0].item()), target_logprobs[0], False
        return


def stream_generate(
    model,
    tokenizer,
    prompt,
    *,
    max_tokens: int = 256,
    mtp_head: MTPHead | None = None,
    **kwargs,
):
    """Streaming text wrapper matching ``mlx_lm.generate.stream_generate``."""
    from mlx_lm.generate import GenerationResponse, generation_stream, wired_limit
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special)
        prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer
    step = mtp_generate_step(
        prompt, model, mtp_head=mtp_head, max_tokens=max_tokens, **kwargs
    )
    with wired_limit(model, [generation_stream]):
        started = time.perf_counter()
        for index, (token, logprobs, from_draft) in enumerate(step):
            if index == 0:
                prompt_time = time.perf_counter() - started
                prompt_tps = prompt.size / prompt_time
                decode_started = time.perf_counter()
            stop = token in tokenizer.eos_token_ids
            if not stop:
                detokenizer.add_token(token)
            final = stop or index + 1 == max_tokens
            if not final:
                yield GenerationResponse(
                    text=detokenizer.last_segment,
                    token=token,
                    logprobs=logprobs,
                    from_draft=from_draft,
                    prompt_tokens=prompt.size,
                    prompt_tps=prompt_tps,
                    generation_tokens=index + 1,
                    generation_tps=(index + 1) / (time.perf_counter() - decode_started),
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason=None,
                )
                continue

            detokenizer.finalize()
            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=index + 1,
                generation_tps=(index + 1) / (time.perf_counter() - decode_started),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason="stop" if stop else "length",
            )
            return
