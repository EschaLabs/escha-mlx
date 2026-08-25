"""Portable NumPy reference for the escha codec (bit-exact ground truth).

This module is the semantic contract for every Metal kernel in this package:
each kernel is gated by comparison against these functions, which are in turn
gated by the reference golden vectors committed under ``tests/data/`` (for
goldens covering a new architecture, open a model-request issue and we will
supply them).

Terminology (public export naming):
  code  — packed weight stream, int16 [E, in/16, out/16, 16*K] per projection
          (dense exports carry one projection: [in/16, out/16, 16*K])
  rin   — per-channel fp16 input scale vector  [E, in]
  rout  — per-channel fp16 output scale vector [E, out]
  s_in  — per-channel fp32 end-to-end input scale  [in]   (dense exports only)
  s_out — per-channel fp32 end-to-end output scale [out]  (dense exports only)
  cbA   — the production codebook (integer hash decode, see ``cba_lut``)

The per-linear forward contract (all rounding points deliberate):
  xh  = f16( H128(x_f32 * rin_f32) * RS )          # per-128-block transform
  mid = xh_f32 @ W_f32                             # W = decoded fp16 tiles
  y   = f16( H128(mid) * RS * rout_f32 )
where H128 is the unnormalized 128-point Walsh-Hadamard transform (Sylvester /
natural order) applied on contiguous 128-channel blocks and RS = 1/sqrt(128).
"""
from __future__ import annotations

import numpy as np

M32 = 0xFFFFFFFF
RS = 0.088388347648  # 1/sqrt(128), the exact f32 constant the format pins


# ---------------------------------------------------------------------------
# codebook
# ---------------------------------------------------------------------------

def cba_lut() -> np.ndarray:
    """65,536-entry fp16 LUT: 16-bit state -> decoded half (cbA codebook).

    decode(x) = fp16_lo(r) + fp16_hi(r)  with fp16 RNE addition, where
    r = ((x * 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60  (32-bit arithmetic).
    """
    x = np.arange(65536, dtype=np.uint64)
    x = (x * np.uint64(0xCBAC1FED)) & np.uint64(M32)
    x = (x & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)
    lo = (x & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16)
    hi = ((x >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16)
    return (lo + hi).astype(np.float16)  # numpy f16+f16 rounds RNE, like the GPU


_LUT = cba_lut()


# ---------------------------------------------------------------------------
# tile decode
# ---------------------------------------------------------------------------

def words_u32(tile_u16: np.ndarray) -> np.ndarray:
    """View one tile's 16*K uint16 as (8*K) little-endian uint32 words."""
    u16 = np.ascontiguousarray(tile_u16.astype(np.uint16))
    return u16.view(np.uint32) if u16.size % 2 == 0 else None


def decode8_k2(words: np.ndarray, lane: int) -> np.ndarray:
    """The 8 states lane owns (K=2). words = the tile's 16 uint32."""
    t_off = lane * 8
    i1 = t_off >> 4
    i0 = (i1 + 15) & 15
    merged = (int(words[i0]) << 32) | int(words[i1])
    shift = ((~t_off) & 8) << 1  # 16 for even lanes, 0 for odd
    w = (merged >> shift) & M32
    out = np.empty(8, dtype=np.uint16)
    for j in range(8):
        out[j] = (w >> (2 * (7 - j))) & 0xFFFF
    return out


def decode8_k3(words: np.ndarray, lane: int) -> np.ndarray:
    """The 8 states lane owns (K=3). words = the tile's 24 uint32."""
    bits = 3
    t_off = lane * 8
    b1 = (t_off + 257) * bits
    b0 = b1 - 16
    b2 = b1 + bits * 7
    i0 = b0 >> 5
    i2 = (b2 - 1) >> 5
    s2 = ((i2 + 1) << 5) - b2
    merged = (int(words[i0 % 24]) << 32) | int(words[i2 % 24])
    w7 = (merged >> s2) & M32
    w3 = (merged >> (s2 + bits * 4)) & M32
    ws = [w3 >> (bits * 3), w3 >> (bits * 2), w3 >> bits, w3,
          w7 >> (bits * 3), w7 >> (bits * 2), w7 >> bits, w7]
    return np.array([w & 0xFFFF for w in ws], dtype=np.uint16)


def lane_positions(lane: int) -> list[tuple[int, int]]:
    """(row, col) inside the 16x16 tile for each of the lane's 8 values.

    Derived from the production reconstruct shuffle; also the accumulation
    order used by the fused GEMV kernels (value j of lane L multiplies input
    channel `row` and accumulates into output channel `col`).
    """
    l0 = lane & ~4
    c_off = (lane >> 2) & 1
    pos = []
    for j in range(8):
        fi = j >> 1
        row = (lane & 3) * 2 + (j & 1) + (fi & 1) * 8
        col = 2 * ((l0 >> 3) + (4 if j >= 4 else 0)) + c_off
        pos.append((row, col))
    return pos


def decode_tile(tile_u16: np.ndarray, K: int) -> np.ndarray:
    """Decode one packed tile (16*K uint16) -> (16, 16) fp16 weight tile."""
    words = words_u32(tile_u16)
    dec8 = decode8_k2 if K == 2 else decode8_k3
    tile = np.zeros((16, 16), dtype=np.float16)
    for lane in range(32):
        vals = _LUT[dec8(words, lane)]
        for j, (r, c) in enumerate(lane_positions(lane)):
            tile[r, c] = vals[j]
    return tile


def reconstruct(code: np.ndarray, in_features: int, out_features: int, K: int) -> np.ndarray:
    """packed (in/16, out/16, 16K) int16/uint16 -> (in, out) fp16 bare weight."""
    tk, tn = in_features // 16, out_features // 16
    code = code.reshape(tk, tn, 16 * K)
    out = np.zeros((in_features, out_features), dtype=np.float16)
    for kt in range(tk):
        for nt in range(tn):
            out[kt * 16:(kt + 1) * 16, nt * 16:(nt + 1) * 16] = decode_tile(code[kt, nt], K)
    return out


def _k2_lane_consts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i0s, i1s, shifts = [], [], []
    for lane in range(32):
        t_off = lane * 8
        i1 = t_off >> 4
        i0s.append((i1 + 15) & 15)
        i1s.append(i1)
        shifts.append(((~t_off) & 8) << 1)
    return (np.array(i0s), np.array(i1s), np.array(shifts, dtype=np.uint64))


def _k3_lane_consts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i0s, i2s, s2s = [], [], []
    for lane in range(32):
        t_off = lane * 8
        b1 = (t_off + 257) * 3
        b0, b2 = b1 - 16, b1 + 21
        i0, i2 = b0 >> 5, (b2 - 1) >> 5
        i0s.append(i0 % 24)
        i2s.append(i2 % 24)
        s2s.append(((i2 + 1) << 5) - b2)
    return (np.array(i0s), np.array(i2s), np.array(s2s, dtype=np.uint64))


def _positions() -> np.ndarray:
    """flat position (row*16+col) for value index lane*8+j, shape (256,)."""
    pos = np.empty(256, dtype=np.int64)
    for lane in range(32):
        for j, (r, c) in enumerate(lane_positions(lane)):
            pos[lane * 8 + j] = r * 16 + c
    return pos


_POS = _positions()


def reconstruct_fast(code: np.ndarray, in_features: int, out_features: int, K: int) -> np.ndarray:
    """Vectorized reconstruct — bit-identical to reconstruct() (asserted in tests)."""
    tk, tn = in_features // 16, out_features // 16
    words = np.ascontiguousarray(code.reshape(tk * tn, 16 * K).astype(np.uint16)).view(np.uint32)
    nt_tiles = words.shape[0]
    if K == 2:
        i0s, i1s, shifts = _k2_lane_consts()
        merged = (words[:, i0s].astype(np.uint64) << 32) | words[:, i1s].astype(np.uint64)
        w = (merged >> shifts) & np.uint64(M32)  # (T, 32)
        js = np.array([2 * (7 - j) for j in range(8)], dtype=np.uint64)
        states = (w[:, :, None] >> js) & np.uint64(0xFFFF)
    else:
        i0s, i2s, s2s = _k3_lane_consts()
        merged = (words[:, i0s].astype(np.uint64) << 32) | words[:, i2s].astype(np.uint64)
        w7 = (merged >> s2s) & np.uint64(M32)
        w3 = (merged >> (s2s + np.uint64(12))) & np.uint64(M32)
        sh = np.array([9, 6, 3, 0], dtype=np.uint64)
        lo = (w3[:, :, None] >> sh) & np.uint64(0xFFFF)  # j = 0..3
        hi = (w7[:, :, None] >> sh) & np.uint64(0xFFFF)  # j = 4..7
        states = np.concatenate([lo, hi], axis=2)
    vals = _LUT[states.astype(np.uint16).reshape(nt_tiles, 256)]  # (T, 256)
    tiles = np.zeros((nt_tiles, 256), dtype=np.float16)
    tiles[:, _POS] = vals
    return (tiles.reshape(tk, tn, 16, 16).transpose(0, 2, 1, 3)
            .reshape(in_features, out_features).copy())


# ---------------------------------------------------------------------------
# transforms + linear forward
# ---------------------------------------------------------------------------

def h128(x: np.ndarray) -> np.ndarray:
    """Unnormalized 128-point WHT (Sylvester order) on 128-blocks of last dim."""
    shape = x.shape
    n = shape[-1]
    assert n % 128 == 0, n
    y = x.astype(np.float32).reshape(-1, n // 128, 128).copy()
    h = 1
    while h < 128:
        y = y.reshape(-1, n // 128, 128 // (2 * h), 2, h)
        a = y[..., 0, :].copy()
        b = y[..., 1, :].copy()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        h *= 2
    return y.reshape(shape)


def input_transform(x: np.ndarray, rin: np.ndarray) -> np.ndarray:
    """xh = f16( H128(x * rin) * RS ).  x [..., IC] any float, rin [IC] or [..., IC]."""
    return (h128(x.astype(np.float32) * rin.astype(np.float32)) * RS).astype(np.float16)


def output_transform(mid: np.ndarray, rout: np.ndarray) -> np.ndarray:
    """y = f16( H128(mid) * RS * rout ).  mid [..., OC] f32."""
    return (h128(mid.astype(np.float32)) * RS * rout.astype(np.float32)).astype(np.float16)


def expert_linear(x: np.ndarray, code: np.ndarray, rin: np.ndarray, rout: np.ndarray,
                  K: int, w_bare: np.ndarray | None = None) -> np.ndarray:
    """Full single-expert linear: x [M, IC] -> [M, OC] fp16."""
    ic, oc = rin.shape[-1], rout.shape[-1]
    if w_bare is None:
        w_bare = reconstruct(code, ic, oc, K)
    xh = input_transform(x, rin)
    mid = xh.astype(np.float32) @ w_bare.astype(np.float32)
    return output_transform(mid, rout)


def fold_scales(rin: np.ndarray, rout: np.ndarray,
                s_in: np.ndarray | None, s_out: np.ndarray | None):
    """Fold the end-to-end scales into the transform vectors -> (f32, f32).

    Dense exports ship two extra per-channel vectors alongside rin/rout: s_in
    [IC] and s_out [OC], the scales learned by the end-to-end fine-tune.  The
    deployed contract is

        y = f16(x * s_in) @ W_deploy * s_out + bias

    and because s_in multiplies the activation at exactly the point rin does
    (per input channel, before the transform) and s_out at exactly the point
    rout does (per output channel, after it), the pair collapses into the
    existing two-vector form with no new kernel and no new tensor:

        rin_eff = f32(rin) * f32(s_in)      rout_eff = f32(rout) * f32(s_out)

    Deliberate deviation, one rounding point FEWER than applying the scales
    separately: doing that rounds ``x * s_in`` to f16 before applying rin, and
    rounds again before applying s_out, because the transform primitive takes
    the scale vectors as separate arguments.  Folding keeps both products in f32
    and rounds once, at the same place every other escha-mlx linear rounds.
    Checked against the reference deploy reconstruction on shipped tensors
    (tests/test_dense_linear.py): the two agree to fp16 rounding, which is the
    same bar the Q8 repack deviation in escha_mlx.quant is held to.

    Passing None for either scale returns that vector unchanged (as f32) — the
    MoE exports, whose s_in/s_out are all-ones, take that path.
    """
    ri = rin.astype(np.float32)
    ro = rout.astype(np.float32)
    if s_in is not None:
        ri = ri * s_in.astype(np.float32)
    if s_out is not None:
        ro = ro * s_out.astype(np.float32)
    return ri, ro


def dense_linear(x: np.ndarray, code: np.ndarray, rin: np.ndarray, rout: np.ndarray,
                 K: int, bias: np.ndarray | None = None) -> np.ndarray:
    """Full dense escha linear: x [M, IC] -> [M, OC] fp16.

    Same codec and rounding contract as ``expert_linear`` — one weight stream
    instead of an E-stacked one — plus the additive fp16 output correction the
    end-to-end fine-tune leaves behind.  ``rin``/``rout`` are expected to be
    the folded vectors from ``fold_scales``.  Call ``expert_linear`` directly if
    you need its pre-decoded-weight escape hatch.
    """
    y = expert_linear(x, code, rin, rout, K)
    if bias is None:
        return y
    return (y.astype(np.float32) + bias.astype(np.float32)).astype(np.float16)


def swiglu(gate_up_f16: np.ndarray) -> np.ndarray:
    """silu(g)*u on the fp16-rounded merged output (gate = first half)."""
    i = gate_up_f16.shape[-1] // 2
    g = gate_up_f16[..., :i].astype(np.float32)
    s = (g / (1.0 + np.exp(-g))).astype(np.float16)
    return (s * gate_up_f16[..., i:]).astype(np.float16)


# ---------------------------------------------------------------------------
# dense int8 + full MoE block reference
# ---------------------------------------------------------------------------

def w8a16(x: np.ndarray, w8: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """y = f16( x_f32 @ f16(w8*scale)_f32^T ).  w8 [O, K] int8, scale [O]."""
    wf = (w8.astype(np.float32) * scale.astype(np.float32)[:, None]).astype(np.float16)
    return (x.astype(np.float32) @ wf.astype(np.float32).T).astype(np.float16)


def moe_block(x: np.ndarray, weights: dict, top_k: int = 8,
              ids: np.ndarray | None = None, scores: np.ndarray | None = None) -> np.ndarray:
    """Full MoE block reference (router + routed experts + shared expert).

    weights keys (numpy): gate_w [E,H] f16; gu_code/gu_rin/gu_rout,
    dn_code/dn_rin/dn_rout (E-stacked); sh_{gate,up,down}_{w8,scale};
    shg_w [1,H] f16.  Router: fp32 logits (rounded f16 like a Linear),
    top-k, softmax over the top-k values.  ids/scores can be injected to
    bypass routing (golden-test mode).
    """
    n, hdim = x.shape
    gate_w = weights["gate_w"]
    logits = (x.astype(np.float32) @ gate_w.astype(np.float32).T).astype(np.float16).astype(np.float32)
    if ids is None:
        ids = np.argsort(-logits, axis=-1, kind="stable")[:, :top_k]
        top = np.take_along_axis(logits, ids, axis=-1)
        e = np.exp(top - top.max(axis=-1, keepdims=True))
        scores = e / e.sum(axis=-1, keepdims=True)
    out = np.zeros((n, hdim), dtype=np.float64)
    for t in range(n):
        for s in range(ids.shape[1]):
            e_id = int(ids[t, s])
            h = expert_linear(x[t:t + 1], weights["gu_code"][e_id],
                              weights["gu_rin"][e_id], weights["gu_rout"][e_id], K=2)
            h = swiglu(h)
            d = expert_linear(h, weights["dn_code"][e_id],
                              weights["dn_rin"][e_id], weights["dn_rout"][e_id], K=3)
            w16 = np.float16(scores[t, s])
            out[t] += (d[0].astype(np.float32) * np.float32(w16)).astype(np.float64)
    g = w8a16(x, weights["sh_gate_w8"], weights["sh_gate_scale"]).astype(np.float32)
    u = w8a16(x, weights["sh_up_w8"], weights["sh_up_scale"]).astype(np.float32)
    hh = ((g / (1.0 + np.exp(-g))) * u).astype(np.float16)
    sh = w8a16(hh, weights["sh_down_w8"], weights["sh_down_scale"]).astype(np.float32)
    shg = (x.astype(np.float32) @ weights["shg_w"].astype(np.float32).T).astype(np.float16).astype(np.float32)
    sgate = 1.0 / (1.0 + np.exp(-shg))
    return (out + sh * sgate).astype(np.float16)
