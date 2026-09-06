"""Reduced-precision storage for the GatedDeltaNet recurrent state.

Why this is a bandwidth lever, not just a capacity one.  The §1 byte ledger
counted weights only; the GDN state is read AND written every decode step:

    30 GDN layers x 32 v-heads x 128 x 128 x 4 B = 62.9 MB/seq in f32
    (+ a ~1.5 MB conv state), touched twice per step

At B=16 that is 2.06 GB of a 9.1 GB step -- larger than the entire dense weight
budget.  Halving it is ~-11% bytes at B=16 and ~-21% at B=32, and it roughly
doubles the sequences that fit a given memory envelope.

WHERE THE PRECISION ACTUALLY GOES.  mlx-lm's `gated_delta_step` Metal kernel
loads the state into `float` registers, runs the whole T-token recurrence in
f32, and casts only on the final store:

    float state[n_per_t];
    state[i] = static_cast<float>(i_state[s_idx]);   // load
    ... T timesteps entirely in f32 ...
    o_state[s_idx] = static_cast<StT>(state[i]);     // store

So the storage dtype costs exactly ONE rounding per kernel call, not per
timestep.  That makes prefill nearly free (one rounding per chunk of T=256) and
decode the worst case (T=1 -> one rounding per token) on a quantity that is a
geometrically-decaying accumulator: an error injected at step t survives as
prod(g) and is continuously re-injected, so the steady-state relative error is
roughly eps/(1-g) rather than eps.  With f16 eps ~ 4.9e-4 and a typical decay
g, that is a real risk and NOT one to reason about analytically -- it is
measured in tests/test_gdn_state_dtype.py and bench/sweep_gdn_state.py.

The delta rule is partly self-correcting (delta = (v - kv_mem) * beta drives the
readout back toward v), which is the mechanism that could make this survive; the
measurement is what decides, not the argument.

MEASURED (bench/sweep_gdn_state.py, teacher-forced on real text with the f32
model's own greedy continuation, 384 decode steps after a 512-token prefill):

    fp16   top-1 agreement 99.74%   KL 2.78e-4   rel 1.09e-2   drift FLAT
    bf16   top-1 agreement 99.48%   KL 4.78e-4   rel 1.30e-2   (worse: 8-bit mantissa)

The drift is flat across the run (KL first quarter 2.5e-4 -> last quarter
1.9e-4), i.e. bounded, not accumulating -- the delta rule's self-correction
holds.  Measure on REAL text: random token ids put the model off-distribution
where logits are flat and near-ties abound, which understates agreement badly
(94.0% vs 99.74% for the identical configuration).

Is that drift acceptable?  Calibrated against the variation the runtime already
has -- same prompt, same fixed continuation, 192 decode steps:

    batch shape B=1 -> B=16 (f32 state)   rel 1.07e-2   KL 2.59e-4   top-1 100%
    fp16 state (B=1)                      rel 1.04e-2   KL 2.52e-4   top-1 100%

**fp16 state perturbs logits slightly LESS than changing the batch size from 1
to 16 already does.**  Batch-shape variation is unavoidable in a batched server
(the row-blocking factor R changes with row count), so this adds no new class of
variation.  It remains fully deterministic: identical shape + identical input ->
identical bits.

DEFAULT: fp16.  It buys ~+10% at B>=32 and 31.5 MB/seq; when it landed
(2026-07-30) that raised the concurrency ceiling from B=64 (17.89 GB) to B=128
(18.98 GB) and with it aggregate throughput 140 -> 167.5 tok/s (current
post-fusion numbers live in docs/PERFORMANCE.md).  ESCHA_MLX_GDN_STATE=fp32 restores the
previous numerics exactly.  Note this DOES change bs1 output relative to the f32
build -- by the amount quantified above.

FIRST-STATE PEAK.  Upstream creates a full f32 zero state when state=None, then
GDNStateCache casts the kernel result to the configured storage dtype. At high
batch sizes the lazy graph can therefore retain both full-sized states. The
zero-state Metal kernel below initializes its f32 accumulator registers to zero
without an input state and writes fp16/bf16 directly. Subsequent calls use the
upstream kernel unchanged.
"""
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache

from . import envs

logger = logging.getLogger(__name__)

# Index 0 of the GDN ArraysCache is the conv state, index 1 the recurrent state.
# Only index 1 is large (2.1 MB/layer vs ~50 KB), and only index 1 is the
# f32-accumulated quantity, so only index 1 is converted.
_STATE_SLOT = 1

_DTYPES = {
    "fp32": mx.float32,
    "float32": mx.float32,
    "fp16": mx.float16,
    "float16": mx.float16,
    "bf16": mx.bfloat16,
    "bfloat16": mx.bfloat16,
}

# Canonical name per dtype. The cache stores the NAME, never the Dtype object:
# mlx.core.Dtype cannot be pickled, and mlx-lm's BatchGenerator.split() runs
# copy.deepcopy() over the whole prompt cache every time a continuous-batching
# batch splits. Holding a Dtype here killed the server on its first split.
_NAME_OF = {mx.float32: "fp32", mx.float16: "fp16", mx.bfloat16: "bf16"}

_ORIGINAL_GATED_DELTA_UPDATE = None
_INITIAL_STATE_DTYPE_NAME: str | None = None


def _make_zero_state_kernel(has_mask: bool = False):
    """First-call GDN kernel with register-zero state and no state input."""
    if not mx.metal.is_available():
        return None

    mask_source = "mask[b_idx * T + t]" if has_mask else "true"
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          state[i] = 0.0f;
        }}

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;
        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            y[dv_idx] = static_cast<InT>(0);
          }}
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}

        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "T"]
    if has_mask:
        inputs.append("mask")
    suffix = "_mask" if has_mask else ""
    return mx.fast.metal_kernel(
        name=f"escha_gated_delta_zero{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


_ZERO_STATE_KERNEL = _make_zero_state_kernel()
_ZERO_STATE_MASKED_KERNEL = _make_zero_state_kernel(has_mask=True)


def _make_trace_kernel(has_mask: bool = False):
    """GDN kernel which exposes the recurrent state after every token.

    Normal generation still uses mlx-lm's final-state-only kernel.  The larger
    output exists only inside an MTP target verification transaction.
    """
    if not mx.metal.is_available():
        return None

    mask_source = "mask[b_idx * T + t]" if has_mask else "true"
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;
        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            y[dv_idx] = static_cast<InT>(0);
          }}

          auto o_state = state_trace
              + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            o_state[s_idx] = static_cast<StT>(state[i]);
          }}

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")
    suffix = "_mask" if has_mask else ""
    return mx.fast.metal_kernel(
        name=f"escha_gated_delta_trace{suffix}",
        input_names=inputs,
        output_names=["y", "state_trace"],
        source=source,
    )


_TRACE_KERNEL = _make_trace_kernel()
_TRACE_MASKED_KERNEL = _make_trace_kernel(has_mask=True)


def _make_tree_trace_kernel():
    """GDN recurrence where every candidate starts from its tree parent."""
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto base_state = state_in + (n * Dv + dv_idx) * Dk;
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          int parent = parents[b_idx * T + t];
          device const StT* parent_state = base_state;
          if (parent >= 0) {
            parent_state = state_trace
                + (((b_idx * T + parent) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          }

          float state[n_per_t];
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(parent_state[s_idx]) * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
          float out = 0.0f;
          auto o_state = state_trace
              + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] += k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
            o_state[s_idx] = static_cast<StT>(state[i]);
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0)
            y[dv_idx] = static_cast<InT>(out);

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
    """
    return mx.fast.metal_kernel(
        name="escha_gated_delta_tree_trace",
        input_names=["q", "k", "v", "g", "beta", "state_in", "parents", "T"],
        output_names=["y", "state_trace"],
        source=source,
    )


_TREE_TRACE_KERNEL = _make_tree_trace_kernel()


def _make_tree_conv_window_kernel():
    """Build every parent-relative depthwise-convolution window in one pass."""
    if not mx.metal.is_available():
        return None

    source = """
        uint linear = thread_position_in_grid.x;
        if (linear >= (uint)(B * C)) return;
        uint b = linear / C;
        uint c = linear - b * C;

        // One thread owns one channel and walks topologically ordered nodes.
        // Reading an earlier node's trace is therefore ordered without a
        // cross-threadgroup barrier.
        for (uint t = 0; t < T; ++t) {
          int parent = parents[b * T + t];
          ulong window_base = ((ulong)b * T + t) * KS * C + c;
          ulong trace_base = ((ulong)b * T + t) * (KS - 1) * C + c;

          for (uint k = 0; k < KS - 1; ++k) {
            InT value;
            if (parent < 0) {
              value = conv_state[((ulong)b * (KS - 1) + k) * C + c];
            } else {
              value = conv_trace[
                  ((ulong)b * T + (uint)parent) * (KS - 1) * C
                  + (ulong)k * C + c];
            }
            windows[window_base + (ulong)k * C] = value;
            if (k > 0)
              conv_trace[trace_base + (ulong)(k - 1) * C] = value;
          }

          InT current = qkv[((ulong)b * T + t) * C + c];
          windows[window_base + (ulong)(KS - 1) * C] = current;
          conv_trace[trace_base + (ulong)(KS - 2) * C] = current;
        }
    """
    return mx.fast.metal_kernel(
        name="escha_gdn_tree_conv_windows",
        input_names=["qkv", "conv_state", "parents"],
        output_names=["conv_trace", "windows"],
        source=source,
    )


_TREE_CONV_WINDOW_KERNEL = _make_tree_conv_window_kernel()


def tree_conv_windows(qkv, conv_state, parents):
    """Return parent-relative conv states and `[B*T, KS, C]` windows."""
    B, T, C = qkv.shape
    if parents.shape != (B, T):
        raise ValueError(
            f"tree conv parents must have shape {(B, T)}, got {parents.shape}"
        )
    if conv_state.ndim != 3 or conv_state.shape[0] != B or conv_state.shape[2] != C:
        raise ValueError(
            "tree conv state must be [B, kernel_size-1, channels] matching qkv"
        )
    n_keep = conv_state.shape[1]
    if n_keep < 1:
        raise ValueError("tree conv requires a kernel size of at least two")
    kernel_size = n_keep + 1
    if (
        _TREE_CONV_WINDOW_KERNEL is not None
        and mx.default_device() == mx.gpu
        and qkv.dtype == conv_state.dtype
    ):
        threads = B * C
        return _TREE_CONV_WINDOW_KERNEL(
            inputs=[qkv, conv_state, parents],
            template=[
                ("InT", qkv.dtype), ("B", B), ("T", T),
                ("C", C), ("KS", kernel_size),
            ],
            grid=((threads + 255) // 256 * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[
                (B, T, n_keep, C),
                (B * T, kernel_size, C),
            ],
            output_dtypes=[qkv.dtype, qkv.dtype],
        )

    conv_states, windows = [], []
    for t in range(T):
        choices = mx.stack([conv_state, *conv_states], axis=1)
        index = (parents[:, t] + 1).reshape(B, 1, 1, 1)
        parent_state = mx.take_along_axis(choices, index, axis=1)[:, 0]
        window = mx.concatenate([parent_state, qkv[:, t : t + 1]], axis=1)
        windows.append(window)
        conv_states.append(mx.contiguous(window[:, 1:]))
    return (
        mx.stack(conv_states, axis=1),
        mx.stack(windows, axis=1).reshape(B * T, kernel_size, C),
    )


def gated_delta_zero_kernel(q, k, v, g, beta, state_type, mask=None):
    """Run the GDN recurrence from zero without allocating an input state."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    kernel = _ZERO_STATE_KERNEL
    inputs = [q, k, v, g, beta, T]
    if mask is not None:
        kernel = _ZERO_STATE_MASKED_KERNEL
        inputs.append(mask)
    if kernel is None:
        raise RuntimeError("zero-state GDN kernel requires Metal")
    return kernel(
        inputs=inputs,
        template=[
            ("InT", q.dtype),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk)],
        output_dtypes=[q.dtype, state_type],
    )


def gated_delta_trace_update(q, k, v, a, b, A_log, dt_bias, state, mask=None):
    """Return GDN output plus state after each position as ``[B,T,H,Dv,Dk]``."""
    from mlx_lm.models import gated_delta

    beta = mx.sigmoid(b)
    g = gated_delta.compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, _, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    kernel = _TRACE_MASKED_KERNEL if mask is not None else _TRACE_KERNEL
    if kernel is not None and mx.default_device() == mx.gpu:
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            inputs.append(mask)
        return kernel(
            inputs=inputs,
            template=[
                ("InT", q.dtype),
                ("StT", state.dtype),
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hk", Hk),
                ("Hv", Hv),
            ],
            grid=(32, Dv, B * Hv),
            threadgroup=(32, 4, 1),
            output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
            output_dtypes=[q.dtype, state.dtype],
        )

    # Portable reference for CPU tests. Keep the recurrence state at the
    # precision mlx-lm's ops path chooses, and cast only captured snapshots.
    if (repeat := Hv // Hk) > 1:
        q = mx.repeat(q, repeat, -2)
        k = mx.repeat(k, repeat, -2)
    outputs, states = [], []
    for t in range(T):
        old_state = state
        decay = g[:, t, :, None, None]
        state = state * decay
        kv_mem = (state * k[:, t, :, None, :]).sum(axis=-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        state = state + k[:, t, :, None, :] * delta[..., None]
        output = (state * q[:, t, :, None, :]).sum(axis=-1).astype(q.dtype)
        if mask is not None:
            active = mx.expand_dims(mask[:, t], axis=(1, 2, 3))
            state = mx.where(active, state, old_state)
            output = mx.where(mask[:, t, None, None], output, 0)
        outputs.append(output)
        states.append(state.astype(old_state.dtype))
    return mx.stack(outputs, axis=1), mx.stack(states, axis=1)


def gated_delta_tree_trace_update(
    q, k, v, a, b, A_log, dt_bias, state, parents
):
    """Return outputs/states for topologically ordered tree nodes.

    ``parents[b, t]`` is -1 for a child of the committed prefix, otherwise the
    index of an earlier node in the same T-wide verification tree.
    """
    from mlx_lm.models import gated_delta

    beta = mx.sigmoid(b)
    g = gated_delta.compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, _, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    parents = parents.astype(mx.int32)
    if _TREE_TRACE_KERNEL is not None and mx.default_device() == mx.gpu:
        return _TREE_TRACE_KERNEL(
            inputs=[q, k, v, g, beta, state, parents, T],
            template=[
                ("InT", q.dtype),
                ("StT", state.dtype),
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hk", Hk),
                ("Hv", Hv),
            ],
            grid=(32, Dv, B * Hv),
            threadgroup=(32, 4, 1),
            output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
            output_dtypes=[q.dtype, state.dtype],
        )

    if (repeat := Hv // Hk) > 1:
        q = mx.repeat(q, repeat, -2)
        k = mx.repeat(k, repeat, -2)
    outputs, states = [], []
    for t in range(T):
        choices = mx.stack([state, *states], axis=1)
        index = (parents[:, t] + 1).reshape(B, 1, 1, 1, 1)
        parent_state = mx.take_along_axis(choices, index, axis=1)[:, 0]
        next_state = parent_state * g[:, t, :, None, None]
        kv_mem = (next_state * k[:, t, :, None, :]).sum(axis=-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        next_state = next_state + k[:, t, :, None, :] * delta[..., None]
        output = (next_state * q[:, t, :, None, :]).sum(axis=-1).astype(q.dtype)
        states.append(next_state.astype(state.dtype))
        outputs.append(output)
    return mx.stack(outputs, axis=1), mx.stack(states, axis=1)


def trace_gdn_forward(self, inputs, mask=None, cache=None, *, tree_parents=None):
    """Run one GatedDeltaNet layer and return its speculative state trace.

    This is called explicitly by the MTP verifier. Keeping the trace transaction
    in the caller avoids process-global state and leaves mlx-lm's class untouched.
    """
    if cache is None or cache.lengths is not None or self.sharding_group is not None:
        raise ValueError("MTP GDN tracing currently requires an unpadded local cache")

    B, S, _ = inputs.shape
    qkv, z = _project_qkv_z(self, inputs)
    z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)

    conv_state = cache[0]
    if conv_state is None:
        conv_state = mx.zeros(
            (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
        )
    n_keep = self.conv_kernel_size - 1
    if tree_parents is None:
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_trace = mx.stack(
            [mx.contiguous(conv_input[:, t : t + n_keep, :])
             for t in range(1, S + 1)],
            axis=1,
        )
        conv_out = nn.silu(self.conv1d(conv_input))
    else:
        conv_trace, window_batch = tree_conv_windows(qkv, conv_state, tree_parents)
        cache[0] = conv_trace[:, -1]
        # Treat every node as a separate length-one convolution while keeping
        # all candidates in one projection/conv graph.
        conv_out = nn.silu(self.conv1d(window_batch)).reshape(B, S, self.conv_dim)

    q, k, v = [
        tensor.reshape(B, S, heads, dim)
        for tensor, heads, dim in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    if tree_parents is None:
        output, state_trace = gated_delta_trace_update(
            q, k, v, a, b, self.A_log, self.dt_bias, cache[1], mask
        )
    else:
        output, state_trace = gated_delta_tree_trace_update(
            q, k, v, a, b, self.A_log, self.dt_bias, cache[1],
            tree_parents,
        )
    cache[1] = state_trace[:, -1]
    cache.advance(S)

    output = self.norm(output, z)
    output = self.out_proj(output.reshape(B, S, -1))
    return output, (cache, conv_trace, state_trace)


def _project_qkv_z(module, inputs):
    """Share one small-row coded projection dispatch in traced GDN forwards."""
    from .dense import EschaLinear, project_pair

    if isinstance(module.in_proj_qkv, EschaLinear) and isinstance(
        module.in_proj_z, EschaLinear
    ):
        return project_pair(module.in_proj_qkv, module.in_proj_z, inputs)
    return module.in_proj_qkv(inputs), module.in_proj_z(inputs)


def commit_tree_states(trace, positions) -> None:
    """Commit a potentially different accepted tree node for every batch row."""
    positions = mx.array(positions).astype(mx.int32)
    B = positions.size
    for cache, conv_states, recurrent_states in trace:
        conv_index = positions.reshape(B, 1, 1, 1)
        state_index = positions.reshape(B, 1, 1, 1, 1)
        cache[0] = mx.take_along_axis(conv_states, conv_index, axis=1)[:, 0]
        cache[1] = mx.take_along_axis(recurrent_states, state_index, axis=1)[:, 0]


def _gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None,
                        mask=None, use_kernel=True):
    """Use the allocation-free kernel only for a missing initial state."""
    original = _ORIGINAL_GATED_DELTA_UPDATE
    if original is None:
        raise RuntimeError("GDN zero-state patch installed without upstream function")
    if (state is not None or not use_kernel or mx.default_device() != mx.gpu
            or not mx.metal.is_available()):
        return original(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)

    from mlx_lm.models import gated_delta

    beta = mx.sigmoid(b)
    g = gated_delta.compute_g(A_log, a, dt_bias)
    state_type = _DTYPES[_INITIAL_STATE_DTYPE_NAME]
    return gated_delta_zero_kernel(q, k, v, g, beta, state_type, mask)


def _install_zero_state_patch(dtype: mx.Dtype) -> None:
    """Patch the upstream symbol used by Qwen3.5's GDN modules."""
    from mlx_lm.models import qwen3_5

    global _ORIGINAL_GATED_DELTA_UPDATE, _INITIAL_STATE_DTYPE_NAME
    if _ORIGINAL_GATED_DELTA_UPDATE is None:
        _ORIGINAL_GATED_DELTA_UPDATE = qwen3_5.gated_delta_update
    _INITIAL_STATE_DTYPE_NAME = _NAME_OF[dtype]
    qwen3_5.gated_delta_update = _gated_delta_update


def _restore_upstream_patch() -> None:
    if _ORIGINAL_GATED_DELTA_UPDATE is None:
        return
    from mlx_lm.models import qwen3_5

    qwen3_5.gated_delta_update = _ORIGINAL_GATED_DELTA_UPDATE


def state_dtype() -> mx.Dtype:
    """Storage dtype for the GDN recurrent state (ESCHA_MLX_GDN_STATE)."""
    v = envs.ESCHA_MLX_GDN_STATE.get()
    return _DTYPES[v]


class GDNStateCache(ArraysCache):
    """ArraysCache that stores the recurrent state in a chosen dtype.

    The first recurrence is handled by the register-zero kernel above. Casting
    on __setitem__ remains a safety net for other writes; subsequent upstream
    kernel calls template on `state.dtype` and already emit the same dtype.
    """

    def __init__(self, size: int = 2, left_padding=None,
                 dtype: mx.Dtype | None = None) -> None:
        super().__init__(size, left_padding)
        dt = dtype if dtype is not None else state_dtype()
        if dt not in _NAME_OF:
            raise ValueError(f"unsupported GDN state dtype {dt}")
        self._gdn_dtype_name = _NAME_OF[dt]

    @property
    def gdn_dtype(self) -> mx.Dtype:
        """Resolved on access; only the name is stored (see _NAME_OF)."""
        return _DTYPES[self._gdn_dtype_name]

    def __setitem__(self, idx, value):
        if idx == _STATE_SLOT and value is not None and value.dtype != self.gdn_dtype:
            value = value.astype(self.gdn_dtype)
        self.cache[idx] = value

    def extract(self, idx):
        # ArraysCache.extract hardcodes its own class, which would silently drop
        # the dtype for any server path that splits a batch.
        cache = type(self)(len(self.cache), dtype=self.gdn_dtype)
        cache.cache = [c[idx: idx + 1] if c is not None else None for c in self.cache]
        return cache


def install(model, dtype: mx.Dtype | None = None) -> mx.Dtype:
    """Point the model's make_cache at GDNStateCache. Returns the dtype in use.

    Requesting f32 is a genuine no-op: the stock ArraysCache already stores
    f32, so that configuration leaves mlx-lm's code path entirely untouched
    rather than routing through an identity cast.
    """
    dt = dtype if dtype is not None else state_dtype()
    if dt == mx.float32:
        _restore_upstream_patch()
        logger.info("escha_mlx: GDN state kept at f32 (ESCHA_MLX_GDN_STATE=fp32) "
                    "— restores pre-2026-07-30 numerics exactly")
        return dt

    _install_zero_state_patch(dt)

    lm = model.language_model
    layers = lm.model.layers
    from mlx_lm.models.cache import KVCache

    def make_cache():
        return [GDNStateCache(size=2, dtype=dt) if l.is_linear else KVCache()
                for l in layers]

    lm.make_cache = make_cache
    model.make_cache = make_cache
    n_linear = sum(1 for l in layers if l.is_linear)
    # Read the geometry off the model rather than assuming it: the state is
    # [num_v_heads, head_k_dim, head_v_dim] per layer, and those differ per
    # architecture (32 v-heads on the 35B MoE, 48 on the 27B dense). A
    # hardcoded 32 understates the dense saving by 1.5x in the one line an
    # operator reads before sizing concurrency.
    gdn = next((l.linear_attn for l in layers if l.is_linear), None)
    per_layer = (gdn.num_v_heads * gdn.head_k_dim * gdn.head_v_dim
                 if gdn is not None else 0)
    logger.info("escha_mlx: GDN recurrent state -> %s across %d linear layers; "
                "allocation-free first state (%.1f MB/seq saved)", dt, n_linear,
                n_linear * per_layer * (4 - dt.size) / 1e6)
    return dt
