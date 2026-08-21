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
import os

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

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
    v = os.environ.get("ESCHA_MLX_GDN_STATE", "fp16").lower()
    if v not in _DTYPES:
        raise ValueError(
            f"ESCHA_MLX_GDN_STATE must be one of {sorted(set(_DTYPES))}, got {v!r}")
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
