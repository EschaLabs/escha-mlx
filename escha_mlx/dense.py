"""Architecture-agnostic dense-linear toolkit for escha runtimes.

Shared by every dense architecture plugin (escha_mlx/models/): the single-stream
coded-weight container and the ``nn.Module`` that executes one trellis-coded
linear. Where a mixture-of-experts model codes only its experts and leaves the
rest of the block in fp16/int8, a dense export codes *every* projection — so
this module is the whole hot path of such a model, and the ~30 lines below are
what the architecture plugin assembles a decoder layer out of.

Relationship to escha_mlx.moe: the codec is the same engine. A dense linear is
the degenerate E=1 case of an expert stream, and the Metal kernels take it as a
compile-time variant (``msl.dense_gemv`` / ``msl.dense_scaled_had``) rather than
a separate implementation — the decode, the accumulation order and the store are
the same instructions, so the dense kernels inherit every gate the expert
kernels already pass. What the variant removes is the per-row indirection: no
``row_expert`` buffer, no dependent load to find the stream, no gather of the
transform vectors.

Per-linear forward (see escha_mlx.ref for the rounding contract):
    xh  = f16( H128(x * rin) * RS )
    mid = xh @ decode(code)                       (f32, fused Metal kernel)
    y   = f16( H128(mid) * RS * rout ) + bias

``rin``/``rout`` are the *folded* transform vectors: dense exports ship the
end-to-end scales s_in/s_out separately, and ``ref.fold_scales`` multiplies them
in at load time because they act at exactly the points rin/rout do. See that
function for the (documented, one-rounding-point-fewer) deviation this implies.

Row counts. A decode step passes one row per sequence, a prefill chunk passes
hundreds. The per-row GEMV reads the whole coded stream per row — no batch
amortization at all — so above a few rows the forward switches to the
row-blocked kernel, which shares one decode across R rows and is bit-identical
to the per-row path (``msl.dense_block_r`` picks R; ESCHA_MLX_DENSE_BLOCK_R
pins it).

Paths:
  * fused (default on Metal)  — escha_mlx.msl kernels.
  * ops   (ESCHA_MLX_LINEAR=ops or no Metal) — numpy tile decode + mx matmul.
    Slow, and it materialises the fp16 weight (IC*OC*2 bytes per linear, cached
    on first use); it exists so the full model runs — and is testable — on any
    backend, not as a deployment path.
"""
from __future__ import annotations

import logging
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import msl, ref
from .moe import had_blocks   # the shared 128-block WHT, not expert-specific

logger = logging.getLogger(__name__)

RS = ref.RS


def linear_mode() -> str:
    """Dense escha linear execution path (``fused`` on Metal, else ``ops``).

    Distinct from ESCHA_MLX_DENSE, which selects how the *int8* dense tensors
    (embed/head) are repacked — see escha_mlx.quant.
    """
    mode = os.environ.get("ESCHA_MLX_LINEAR",
                          "fused" if mx.metal.is_available() else "ops")
    if mode not in ("fused", "ops"):
        raise ValueError(f"ESCHA_MLX_LINEAR must be 'fused' or 'ops', got {mode!r}")
    return mode


def pack_code(code_i16: np.ndarray) -> mx.array:
    """int16 [TK, TN, 16K] -> uint32 [TK, TN, 8K] device array.

    Exists so a streaming loader can convert each code stream the moment it
    arrives and drop the numpy copy.  A dense export groups its tensors by leaf
    name, not by module — every ``escha_s_in`` in the model precedes the first
    ``escha_code`` — so a loader that waited for each linear's full leaf set
    before converting would hold the entire coded body (the bulk of the
    checkpoint) in numpy *and* then again in device memory.
    """
    return mx.array(msl.code_to_u32(code_i16))


class EschaWeight:
    """One dense linear's coded stream plus its folded transform vectors.

    Kept off the module parameter tree (the owner holds it under a
    leading-underscore attribute), so the generic loader must ``mx.eval`` the
    arrays this exposes via ``arrays()`` — same contract as
    ``escha_mlx.moe.EschaExperts``.
    """

    def __init__(self, code: np.ndarray | mx.array, rin: np.ndarray, rout: np.ndarray,
                 s_in: np.ndarray | None = None, s_out: np.ndarray | None = None,
                 config: np.ndarray | None = None) -> None:
        if isinstance(code, mx.array):
            tk, tn, wpt = code.shape          # already packed: 8K uint32 words
            if wpt % 8:
                raise ValueError(f"packed escha code last dim must be 8*K, got {wpt}")
            self.K = wpt // 8
            code_mx = code
        else:
            tk, tn, wpt2 = code.shape         # raw export: 16K int16 words
            if wpt2 % 16:
                raise ValueError(f"escha code last dim must be 16*K, got {wpt2}")
            self.K = wpt2 // 16
            code_mx = pack_code(code)
        self.IC, self.OC = tk * 16, tn * 16
        if rin.shape[-1] != self.IC or rout.shape[-1] != self.OC:
            raise ValueError(
                f"transform vectors {rin.shape[-1]}/{rout.shape[-1]} do not match "
                f"the code stream's {self.IC}x{self.OC}")
        # The kernels assume this and do not check it: the transforms run one
        # 128-block per threadgroup and the GEMV grid covers 128 output channels
        # per threadgroup, so a dimension that is a multiple of 16 (tile-aligned)
        # but not of 128 would silently drop the remainder instead of failing.
        if self.IC % 128 or self.OC % 128:
            raise ValueError(
                f"escha linear dimensions must be multiples of 128, got "
                f"{self.IC}x{self.OC}")
        # escha_config is [L, K, V, codebook_id, IC, OC]. The shapes above are
        # authoritative (they are what the kernels index with), so the config is
        # a cross-check rather than a source: a K or a shape that disagrees means
        # the stream and its metadata came from different exports, which would
        # otherwise decode into plausible-looking noise.
        if config is not None:
            cfg = [int(v) for v in np.asarray(config).ravel()]
            if len(cfg) >= 6:
                _, k_cfg, _, cb, ic_cfg, oc_cfg = cfg[:6]
                if (k_cfg, ic_cfg, oc_cfg) != (self.K, self.IC, self.OC):
                    raise ValueError(
                        f"escha_config {(k_cfg, ic_cfg, oc_cfg)} disagrees with the "
                        f"code stream {(self.K, self.IC, self.OC)}")
                if cb != 1:
                    raise ValueError(
                        f"escha_config selects codebook id {cb}; this runtime "
                        f"implements the production codebook (id 1) only")
        self.code = code_mx                                  # [TK, TN, 8K] u32
        ri, ro = ref.fold_scales(rin, rout, s_in, s_out)
        self.rin = mx.array(ri)                              # [IC] f32
        self.rout = mx.array(ro)                             # [OC] f32
        self._w_np: np.ndarray | None = None
        self._code_np: np.ndarray | None = None

    def weight_numpy(self) -> np.ndarray:
        """Decoded fp16 bare weight [IC, OC] — ops path only (see module doc)."""
        if self._w_np is None:
            if self._code_np is None:
                self._code_np = np.array(self.code)
            self._w_np = ref.reconstruct_fast(
                self._code_np.view(np.uint16).view(np.int16),
                self.IC, self.OC, self.K).astype(np.float32)
        return self._w_np

    def arrays(self) -> list[mx.array]:
        return [self.code, self.rin, self.rout]


class EschaLinear(nn.Module):
    """A trellis-coded dense linear. Drop-in for ``nn.Linear`` (bias folded in)."""

    def __init__(self, w: EschaWeight, bias: np.ndarray | None = None) -> None:
        super().__init__()
        self._w = w
        self.bias = None if bias is None else mx.array(bias.astype(np.float16))
        self._mode = linear_mode()
        # Read once at construction: flipping it per-forward would defeat the
        # compile cache and make A/Bs depend on call order.
        self._fused_had = msl.use_fused_had() and self._mode == "fused"
        self._block_r = msl.dense_block_r if self._mode == "fused" else None

    @property
    def K(self) -> int:
        return self._w.K

    def _input_rows(self, rows: mx.array) -> mx.array:
        """f16( H128(rows * rin) * RS )."""
        if self._fused_had and rows.dtype == mx.float16:
            return msl.dense_scaled_had(rows, self._w.rin, RS)
        xr = rows.astype(mx.float32) * self._w.rin
        return (had_blocks(xr) * RS).astype(mx.float16)

    def _output_rows(self, mid: mx.array) -> mx.array:
        """f16( H128(mid) * RS * rout )."""
        if self._fused_had and mid.dtype == mx.float32:
            return msl.dense_scaled_had_out(mid, self._w.rout, RS)
        return (had_blocks(mid) * RS * self._w.rout).astype(mx.float16)

    def _gemv(self, xh: mx.array) -> mx.array:
        w = self._w
        if self._mode == "fused":
            # Above a few rows the row-blocked kernel shares one decode of the
            # coded stream across R rows. Without it a prefill chunk would
            # decode every projection once PER TOKEN -- the per-row kernel has
            # no batch amortization by construction. Bit-identical either way.
            r = 1 if self._block_r is None else self._block_r(xh.shape[0])
            if r > 1:
                return msl.dense_gemm_rows(xh, w.code, w.K, w.IC, w.OC, r)
            return msl.dense_gemv(xh, w.code, w.K, w.IC, w.OC)
        # ops path: numpy decode + matmul (test/CPU only — slow, see module doc)
        return mx.array(np.array(xh).astype(np.float32) @ w.weight_numpy())

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        y = self._output_rows(self._gemv(self._input_rows(rows)))
        if self.bias is not None:
            # The forward contract rounds the transform output to f16 before the
            # correction is added; keep that rounding point and add in f32.
            y = (y.astype(mx.float32) + self.bias.astype(mx.float32)).astype(mx.float16)
        return y.reshape(*shape[:-1], self._w.OC).astype(x.dtype)


def build(group: dict[str, np.ndarray]) -> EschaLinear:
    """Assemble one linear from its streamed tensors, keyed by HF leaf name.

    Required: ``escha_code``, ``escha_rin``, ``escha_rout``. Optional:
    ``escha_s_in``/``escha_s_out`` (absent on exports without an end-to-end
    stage — they are then implicitly ones), ``escha_config`` (cross-checked),
    ``bias``.
    """
    w = EschaWeight(group["escha_code"], group["escha_rin"], group["escha_rout"],
                    group.get("escha_s_in"), group.get("escha_s_out"),
                    group.get("escha_config"))
    return EschaLinear(w, group.get("bias"))


#: Leaf names that together define one coded linear.
LEAVES = frozenset({"escha_code", "escha_rin", "escha_rout",
                    "escha_s_in", "escha_s_out", "escha_config", "bias"})

#: The subset that must be present before a linear can be built.
REQUIRED = frozenset({"escha_code", "escha_rin", "escha_rout"})
