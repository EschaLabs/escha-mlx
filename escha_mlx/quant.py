"""Exact repack of escha int8 dense tensors into MLX-native affine Q8.

The checkpoint stores dense linears as `weight_int8` [N, K] + per-row
`weight_scale` f16 [N] with the contract
    wf = f16(f32(w8) * f32(scale));  y = f16(f32-accum x @ wf^T).
MLX affine quantization dequantizes w_hat = scale_g * q + bias_g with
UNSIGNED q packed little-end-first into uint32. The mapping

    q = w8 + 128,   scale_g = scale,   bias_g = -128 * scale

is exact when computed in f32 (both products are <=19-bit-mantissa exact),
so we store scales/biases in **f32**: MLX computes the dequant in the scales'
dtype (measured: f16 scales give half-precision dequant on the CPU backend —
up to 37 ulp off — while f32 scales are bit-exact on every backend).
``validate_pack`` asserts the round-trip at load time; if MLX ever changes
its packing order or dequant semantics we fail loudly instead of serving
garbage.

One (documented) deviation from the w8a16 contract: quantized_matmul
multiplies the exact f32 dequant values instead of rounding each weight to
f16 first. Both paths accumulate in f32 and round the output to f16 once; the per-weight
difference is <=0.5 ulp f16 (validated against the w8a16 goldens in
tests/test_mlx_cpu.py).

Fallback: ESCHA_MLX_DENSE=fp16 dequantizes to fp16 nn.Linear (2x bytes,
bit-identical weights to the contract's wf).

Group size (ESCHA_MLX_Q8_GROUP, default 128): because the escha scale is
per-OUTPUT-CHANNEL, ``pack_q8`` writes the SAME scale/bias into every group of
a row -- so the group size has no effect whatsoever on the represented values,
only on how many times that constant is stored.  scales+biases are f32 (see
above), i.e. 8 bytes per group: at group 64 that is 12.5% metadata on top of
the 1 byte/weight payload, at group 128 it is 6.25%.  Across the 2.16 GB of Q8
streams this model reads per token that is a ~120 MB/token saving for provably
zero numerical change -- the largest free win in the byte ledger.  128 is
MLX's maximum affine group size, hence the default.
"""
from __future__ import annotations

import logging
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

_VALIDATED: set[int] = set()

# MLX affine group sizes, largest first. 128 is the upper bound MLX accepts.
_GROUPS = (128, 64, 32)

DEFAULT_GROUP = 128


def fit_group(k: int, requested: int = DEFAULT_GROUP) -> int:
    """Largest supported group size <= `requested` that divides `k`.

    Every tensor in the shipped export has K in {512, 2048, 4096}, so the
    requested size always wins; this exists so an export with an odd K
    degrades to a smaller group instead of failing the assert in pack_q8.
    Correctness is unaffected either way (the scale is per-row constant).
    """
    for g in _GROUPS:
        if g <= requested and k % g == 0:
            return g
    raise ValueError(f"no MLX affine group size in {_GROUPS} divides K={k}")


def pack_q8(w8: np.ndarray, scale: np.ndarray, group_size: int = DEFAULT_GROUP):
    """int8 [N, K] + scale [N] -> (weight u32 [N, K/4], scales/biases f32 [N, K/gs]).

    q = w8 + 128 is exactly w8 XOR 0x80 on the two's-complement byte, and
    MLX's little-end-first uint32 packing is exactly little-endian memory
    byte order — so the pack is one XOR pass + a dtype view (no 4x int32
    temporaries; matters when repacking the 509 MB embed/head on a 24 GB Mac).
    """
    n, k = w8.shape
    assert k % group_size == 0 and k % 4 == 0, (n, k, group_size)
    q = np.ascontiguousarray(w8).view(np.uint8) ^ np.uint8(0x80)
    packed = q.reshape(n, k).view(np.uint32)  # [N, K/4], little-endian
    s32 = scale.astype(np.float32)
    scales = np.repeat(s32[:, None], k // group_size, axis=1)
    biases = np.repeat((np.float32(-128.0) * s32)[:, None], k // group_size, axis=1)
    return packed, scales, biases


def validate_pack(group_size: int = DEFAULT_GROUP) -> None:
    """One-time on-device check: dequant(pack_q8(w8)) -> f16 == f16(w8*scale).

    Validated per group size, not once globally: a single export can mix group
    sizes (see fit_group), and each one is a distinct MLX dequant path.
    """
    if group_size in _VALIDATED:
        return
    rng = np.random.default_rng(0)
    w8 = rng.integers(-128, 128, size=(4, 512), dtype=np.int8)
    scale = (rng.random(4, dtype=np.float32) * 0.05 + 1e-3).astype(np.float16)
    packed, scales, biases = pack_q8(w8, scale, group_size)
    deq = mx.dequantize(mx.array(packed), mx.array(scales), mx.array(biases),
                        group_size=group_size, bits=8)
    want = (w8.astype(np.float32) * scale.astype(np.float32)[:, None]).astype(np.float16)
    got = np.array(deq.astype(mx.float16))
    if not np.array_equal(got.view(np.uint16), want.view(np.uint16)):
        raise RuntimeError(
            "escha_mlx: MLX affine-Q8 repack round-trip is NOT bit-exact on this "
            "mlx version — refusing to serve. (Packing order or dequant semantics "
            "changed upstream; see escha_mlx/quant.py.)")
    _VALIDATED.add(group_size)
    logger.info("escha_mlx: Q8 repack validated bit-exact (group_size=%d)", group_size)


def dense_mode() -> str:
    return os.environ.get("ESCHA_MLX_DENSE", "q8")


class EschaQ8Linear(nn.Module):
    """y = f16( x @ dequant(w)^T ), f32 accumulate — the w8a16 contract."""

    def __init__(self, packed: np.ndarray, scales: np.ndarray, biases: np.ndarray,
                 group_size: int) -> None:
        super().__init__()
        self.weight = mx.array(packed)
        self.scales = mx.array(scales)
        self.biases = mx.array(biases)
        self._group_size = group_size

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                transpose=True, group_size=self._group_size, bits=8)
        return y.astype(x.dtype)


class EschaQ8Embedding(nn.Module):
    def __init__(self, packed: np.ndarray, scales: np.ndarray, biases: np.ndarray,
                 group_size: int, dtype=mx.float16) -> None:
        super().__init__()
        self.weight = mx.array(packed)
        self.scales = mx.array(scales)
        self.biases = mx.array(biases)
        self._group_size = group_size
        self._dtype = dtype

    def __call__(self, ids: mx.array) -> mx.array:
        rows = mx.dequantize(self.weight[ids], self.scales[ids], self.biases[ids],
                             group_size=self._group_size, bits=8)
        return rows.astype(self._dtype)


def make_linear(w8: np.ndarray, scale: np.ndarray,
                group_size: int = DEFAULT_GROUP) -> nn.Module:
    n, k = w8.shape
    if dense_mode() == "fp16":
        lin = nn.Linear(k, n, bias=False)
        lin.weight = mx.array((w8.astype(np.float32) * scale.astype(np.float32)[:, None])
                              .astype(np.float16))
        return lin
    g = fit_group(k, group_size)
    validate_pack(g)
    return EschaQ8Linear(*pack_q8(w8, scale, g), g)


def make_embedding(w8: np.ndarray, scale: np.ndarray,
                   group_size: int = DEFAULT_GROUP) -> nn.Module:
    n, k = w8.shape
    if dense_mode() == "fp16":
        emb = nn.Embedding(n, k)
        emb.weight = mx.array((w8.astype(np.float32) * scale.astype(np.float32)[:, None])
                              .astype(np.float16))
        return emb
    g = fit_group(k, group_size)
    validate_pack(g)
    return EschaQ8Embedding(*pack_q8(w8, scale, g), g)
