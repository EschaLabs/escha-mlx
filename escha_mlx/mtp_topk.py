"""Small-row top-k log probabilities for native-MTP tree proposals.

The general MLX path materializes normalized logits and then runs a vocabulary-
wide argpartition. Tree drafting only needs a few values and indices, so the
Metal path reduces 2048-logit blocks and then merges their summaries. Its
logsumexp association can differ by one f16 ulp from MLX; this is proposal-only
scoring and every selected token is still verified by the target model.
"""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_CHUNK = 2048
_THREADS = 256
_TOPK = 3
_HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"


_PARTIAL_SOURCE = r"""
    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint block = thread_position_in_grid.x >> 8;
    uint row = thread_position_in_grid.y;
    uint base = block * CHUNK;

    float best[TOPK];
    uint best_idx[TOPK];
    for (uint j = 0; j < TOPK; ++j) {
      best[j] = -INFINITY;
      best_idx[j] = 0;
    }
    for (uint item = 0; item < ITEMS; ++item) {
      uint index = base + tid + item * 256;
      if (index >= N) continue;
      float value = (float)logits[(ulong)row * N + index];
      for (uint j = 0; j < TOPK; ++j) {
        if (value > best[j]) {
          for (uint shift = TOPK - 1; shift > j; --shift) {
            best[shift] = best[shift - 1];
            best_idx[shift] = best_idx[shift - 1];
          }
          best[j] = value;
          best_idx[j] = index;
          break;
        }
      }
    }

    threadgroup float candidates[256 * TOPK];
    threadgroup uint candidate_indices[256 * TOPK];
    for (uint j = 0; j < TOPK; ++j) {
      candidates[tid * TOPK + j] = best[j];
      candidate_indices[tid * TOPK + j] = best_idx[j];
    }

    // Keep the broadcast maximum and subgroup sums in separate storage.
    // Reusing subgroup[0] lets SIMD group zero overwrite the maximum while
    // another group is still reading it in the exp loop.
    threadgroup float subgroup_max[8];
    threadgroup float subgroup_sum[8];
    float reduced_max = simd_max(best[0]);
    if (lane == 0) subgroup_max[sg] = reduced_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float group_max = subgroup_max[0];
      for (uint j = 1; j < 8; ++j)
        group_max = max(group_max, subgroup_max[j]);
      part_max[(ulong)row * BLOCKS + block] = group_max;
      subgroup_max[0] = group_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (uint item = 0; item < ITEMS; ++item) {
      uint index = base + tid + item * 256;
      if (index < N)
        sum += exp((float)logits[(ulong)row * N + index] - subgroup_max[0]);
    }
    sum = simd_sum(sum);
    if (lane == 0) subgroup_sum[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
      float group_sum = 0.0f;
      for (uint j = 0; j < 8; ++j) group_sum += subgroup_sum[j];
      ulong out = ((ulong)row * BLOCKS + block) * TOPK;
      float group_best[TOPK];
      uint group_idx[TOPK];
      for (uint j = 0; j < TOPK; ++j) {
        group_best[j] = -INFINITY;
        group_idx[j] = 0;
      }
      for (uint candidate = 0; candidate < 256 * TOPK; ++candidate) {
        float value = candidates[candidate];
        uint index = candidate_indices[candidate];
        for (uint j = 0; j < TOPK; ++j) {
          if (value > group_best[j]) {
            for (uint shift = TOPK - 1; shift > j; --shift) {
              group_best[shift] = group_best[shift - 1];
              group_idx[shift] = group_idx[shift - 1];
            }
            group_best[j] = value;
            group_idx[j] = index;
            break;
          }
        }
      }
      for (uint j = 0; j < TOPK; ++j) {
        part_values[out + j] = group_best[j];
        part_indices[out + j] = group_idx[j];
      }
      part_sum[(ulong)row * BLOCKS + block] = group_sum;
    }
"""


_FINAL_SOURCE = r"""
    uint tid = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row = thread_position_in_grid.y;

    float block_max = tid < BLOCKS
        ? part_max[(ulong)row * BLOCKS + tid] : -INFINITY;
    float reduced_max = simd_max(block_max);
    threadgroup float subgroup_max[8];
    threadgroup float subgroup_sum[8];
    if (lane == 0) subgroup_max[sg] = reduced_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float global_max = subgroup_max[0];
      for (uint j = 1; j < 8; ++j)
        global_max = max(global_max, subgroup_max[j]);
      subgroup_max[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = subgroup_max[0];

    float adjusted = tid < BLOCKS
        ? part_sum[(ulong)row * BLOCKS + tid] * exp(block_max - global_max)
        : 0.0f;
    adjusted = simd_sum(adjusted);
    if (lane == 0) subgroup_sum[sg] = adjusted;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
      float total = 0.0f;
      for (uint j = 0; j < 8; ++j) total += subgroup_sum[j];
      float log_norm = global_max + log(total);
      float best[TOPK];
      uint best_idx[TOPK];
      for (uint j = 0; j < TOPK; ++j) {
        best[j] = -INFINITY;
        best_idx[j] = 0;
      }
      ulong base = (ulong)row * BLOCKS * TOPK;
      for (uint candidate = 0; candidate < BLOCKS * TOPK; ++candidate) {
        float value = part_values[base + candidate];
        uint index = part_indices[base + candidate];
        for (uint j = 0; j < TOPK; ++j) {
          if (value > best[j]) {
            for (uint shift = TOPK - 1; shift > j; --shift) {
              best[shift] = best[shift - 1];
              best_idx[shift] = best_idx[shift - 1];
            }
            best[j] = value;
            best_idx[j] = index;
            break;
          }
        }
      }
      for (uint j = 0; j < TOPK; ++j) {
        scores[(ulong)row * TOPK + j] = (half)(best[j] - log_norm);
        indices[(ulong)row * TOPK + j] = best_idx[j];
      }
    }
"""


@lru_cache(maxsize=1)
def _partial_kernel():
    return mx.fast.metal_kernel(
        name="escha_mtp_topk_partial_k3",
        input_names=["logits"],
        output_names=["part_values", "part_indices", "part_max", "part_sum"],
        header=_HEADER,
        source=_PARTIAL_SOURCE,
    )


@lru_cache(maxsize=1)
def _final_kernel():
    return mx.fast.metal_kernel(
        name="escha_mtp_topk_final_k3",
        input_names=["part_values", "part_indices", "part_max", "part_sum"],
        output_names=["scores", "indices"],
        header=_HEADER,
        source=_FINAL_SOURCE,
    )


def metal_topk_log_probs(logits: mx.array, *, topk: int = _TOPK):
    """Return top-3 normalized scores/indices, or `None` for MLX fallback."""
    if (
        topk != _TOPK
        or logits.ndim < 2
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or logits.dtype != mx.float16
        or logits.shape[-1] < _CHUNK
    ):
        return None

    shape = logits.shape
    vocab = shape[-1]
    rows = 1
    for dimension in shape[:-1]:
        rows *= dimension
    blocks = (vocab + _CHUNK - 1) // _CHUNK
    if blocks > _THREADS:
        return None
    flat = logits.reshape(rows, vocab)
    partial = _partial_kernel()(
        inputs=[flat],
        template=[
            ("InT", logits.dtype), ("N", vocab), ("TOPK", _TOPK),
            ("CHUNK", _CHUNK), ("ITEMS", _CHUNK // _THREADS),
            ("BLOCKS", blocks),
        ],
        grid=(_THREADS * blocks, rows, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[
            (rows, blocks, _TOPK), (rows, blocks, _TOPK),
            (rows, blocks), (rows, blocks),
        ],
        output_dtypes=[mx.float32, mx.uint32, mx.float32, mx.float32],
    )
    scores, indices = _final_kernel()(
        inputs=list(partial),
        template=[("TOPK", _TOPK), ("BLOCKS", blocks)],
        grid=(_THREADS, rows, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(rows, _TOPK), (rows, _TOPK)],
        output_dtypes=[mx.float16, mx.uint32],
    )
    output_shape = (*shape[:-1], _TOPK)
    return scores.reshape(output_shape), indices.reshape(output_shape)
