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

DEFAULT: fp16.  It buys ~+10% at B>=32 and 31.5 MB/seq, which raises the
concurrency ceiling from B=64 (17.89 GB) to B=128 (18.98 GB) and with it
aggregate throughput 140 -> 167.5 tok/s.  ESCHA_MLX_GDN_STATE=fp32 restores the
previous numerics exactly.  Note this DOES change bs1 output relative to the f32
build -- by the amount quantified above.
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


def state_dtype() -> mx.Dtype:
    """Storage dtype for the GDN recurrent state (ESCHA_MLX_GDN_STATE)."""
    v = os.environ.get("ESCHA_MLX_GDN_STATE", "fp16").lower()
    if v not in _DTYPES:
        raise ValueError(
            f"ESCHA_MLX_GDN_STATE must be one of {sorted(set(_DTYPES))}, got {v!r}")
    return _DTYPES[v]


class GDNStateCache(ArraysCache):
    """ArraysCache that stores the recurrent state in a chosen dtype.

    Casting on __setitem__ (rather than patching GatedDeltaNet.__call__) keeps
    this to one override and leaves mlx-lm untouched: the model writes
    `cache[1] = state` and the kernel reads whatever dtype it finds, because
    `gated_delta_kernel` templates on `state.dtype` already.
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
        logger.info("escha_mlx: GDN state kept at f32 (ESCHA_MLX_GDN_STATE=fp32) "
                    "— restores pre-2026-07-30 numerics exactly")
        return dt

    lm = model.language_model
    layers = lm.model.layers
    from mlx_lm.models.cache import KVCache

    def make_cache():
        return [GDNStateCache(size=2, dtype=dt) if l.is_linear else KVCache()
                for l in layers]

    lm.make_cache = make_cache
    model.make_cache = make_cache
    n_linear = sum(1 for l in layers if l.is_linear)
    logger.info("escha_mlx: GDN recurrent state -> %s across %d linear layers "
                "(%.1f MB/seq saved)", dt, n_linear,
                n_linear * 32 * 128 * 128 * (4 - dt.size) / 1e6)
    return dt
