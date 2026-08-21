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
    y   = f16( H128(mid) * RS * rout ) [+ bias]
The bracketed correction is OFF by default because the reference runtime does
not apply it either -- see ``apply_bias``, which is worth reading before
comparing this runtime's output to anything.

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
    Slow, and it materialises the decoded weight as **f32** (IC*OC*4 bytes per
    linear, cached for the module's lifetime) because that is what the
    reference contract multiplies in. It exists so a model runs — and is
    testable — on any backend, not as a deployment path, and it does NOT scale
    to a large checkpoint: touching every linear of a 24.3 G-parameter coded
    body once materialises ~97 GB. Use it on small models, on a truncated
    load, or one layer at a time.
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

#: Rates the codec implements. Both msl and ref dispatch on `K == 2 ? k2 : k3`,
#: so anything else must be refused at load rather than mis-decoded.
SUPPORTED_K = frozenset({2, 3})

#: escha_config layout: [L, K, V, codebook_id, in_features, out_features].
_CONFIG_LEN = 6
_PRODUCTION_CODEBOOK = 1


def _check_scale(name: str, vec, expect: int) -> None:
    """A scale vector must be 1-D of exactly `expect` entries, or absent."""
    if vec is None:
        return
    shape = tuple(np.shape(vec))
    if shape != (expect,):
        raise ValueError(f"{name} must have shape ({expect},), got {shape}")


_WARNED_OPS_ON_METAL = False


def _warn_ops_on_metal() -> None:
    """Say so, once, when the NumPy oracle runs on a machine that has Metal.

    Every fallback in this module is silent and condition-driven — the fused
    transforms need f16/f32 inputs, the fused GEMV needs `fused` mode — so an
    ESCHA_MLX_LINEAR left exported in a shell degrades the whole model to the
    oracle path with correct output, no error, and orders of magnitude less
    throughput. That reads as "the kernels are slow", not as a misconfiguration.
    """
    global _WARNED_OPS_ON_METAL
    if _WARNED_OPS_ON_METAL:
        return
    _WARNED_OPS_ON_METAL = True
    logger.warning(
        "escha_mlx: dense linears are using the NumPy oracle path "
        "(ESCHA_MLX_LINEAR=ops) on a machine that HAS Metal. This is orders of "
        "magnitude slower and materialises each weight in f32 — unset "
        "ESCHA_MLX_LINEAR to use the kernels.")


def apply_bias() -> bool:
    """Whether to apply the per-linear correction the export ships (default NO).

    A dense export stores a `bias` beside every coded linear -- the additive
    correction the end-to-end stage learned. On the shipped Qwen3.8-27B all 400
    are non-zero, and applying one moves that linear's output by 6.7-8.3%
    (measured on the committed goldens): four orders of magnitude above the
    fp16-rounding differences every other gate in this package is held to. This
    is a fork in the model, not a rounding choice.

    The path the published numbers come from does NOT apply them. That
    quantization method registers exactly the six escha_* tensors and no bias,
    every coded linear is constructed `bias=False`, and unmatched `.bias` names
    are dropped by a GPTQ-era
    `if name.endswith(".bias") and name not in params_dict: continue`
    -- silently, with no warning. So every published number for this checkpoint
    was produced WITHOUT the correction.

    Default off, therefore: this runtime's job is to reproduce the published
    model, and a 6-8% per-linear divergence compounding over 64 layers would
    make any comparison against the model card meaningless. ESCHA_MLX_BIAS=1
    applies them, which is what the export's own written contract
    (`y = (x * s_in) @ W * s_out + bias`) asks for -- run it as a paired A/B on
    a real task before treating either as correct.
    """
    return os.environ.get("ESCHA_MLX_BIAS", "0") == "1"


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
        # K is inferred from the stream, so an unimplemented rate must be
        # refused HERE: both the kernels and the reference select the unpacker
        # with `K == 2 ? k2 : k3`, so any other K would be decoded by the K=3
        # unpacker and yield plausible Gaussian weights with no error at all.
        if self.K not in SUPPORTED_K:
            raise ValueError(
                f"escha code implies K={self.K}; this runtime implements "
                f"K in {sorted(SUPPORTED_K)}")
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
            # Refuse an unexpected header rather than skipping the check: a
            # shorter one would silently bypass the codebook gate below, which
            # is the only thing standing between a future non-production
            # codebook and a model that decodes to confident noise.
            if len(cfg) != _CONFIG_LEN:
                raise ValueError(
                    f"escha_config has {len(cfg)} fields, expected {_CONFIG_LEN} "
                    f"([L, K, V, codebook_id, in_features, out_features]); this "
                    f"export was written by an incompatible version")
            _, k_cfg, _, cb, ic_cfg, oc_cfg = cfg
            if (k_cfg, ic_cfg, oc_cfg) != (self.K, self.IC, self.OC):
                raise ValueError(
                    f"escha_config {(k_cfg, ic_cfg, oc_cfg)} disagrees with the "
                    f"code stream {(self.K, self.IC, self.OC)}")
            if cb != _PRODUCTION_CODEBOOK:
                raise ValueError(
                    f"escha_config selects codebook id {cb}; this runtime "
                    f"implements the production codebook "
                    f"(id {_PRODUCTION_CODEBOOK}) only")
        # The scales are folded by a bare multiply, which BROADCASTS: an s_in
        # of shape [n, IC] would yield a 2-D rin that the fused kernel indexes
        # as `blk*128 + tid`, silently applying row 0 to every channel, and a
        # scalar would silently apply one scale everywhere. Check the shape,
        # not just the trailing length.
        _check_scale("escha_s_in", s_in, self.IC)
        _check_scale("escha_s_out", s_out, self.OC)
        self.code = code_mx                                  # [TK, TN, 8K] u32
        ri, ro = ref.fold_scales(rin, rout, s_in, s_out)
        self.rin = mx.array(ri)                              # [IC] f32
        self.rout = mx.array(ro)                             # [OC] f32
        self._w_np: np.ndarray | None = None
        self._code_np: np.ndarray | None = None

    def weight_numpy(self) -> np.ndarray:
        """Decoded bare weight [IC, OC] as f32 — ops path only.

        Cached for the module's lifetime, at 4 bytes per coded weight. See the
        module docstring: this does not scale to a large checkpoint.
        """
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
        if self._mode == "ops" and mx.metal.is_available():
            _warn_ops_on_metal()
        # Read once at construction: flipping it per-forward would defeat the
        # compile cache and make A/Bs depend on call order.
        self._fused_had = msl.use_fused_had() and self._mode == "fused"
        self._block_r_pin = msl.dense_block_r_pin() if self._mode == "fused" else None

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
            r = msl.dense_block_r(xh.shape[0], self._block_r_pin)
            if r >= 16 and msl.use_dense_mat():
                # Deterministic but NOT bit-identical (reassociated f32 sum);
                # default off, ESCHA_MLX_DENSE_MAT=1. See msl.use_dense_mat.
                return msl.dense_gemm_mat(xh, w.code, w.K, w.IC, w.OC)
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
    bias = group.get("bias") if apply_bias() else None
    return EschaLinear(w, bias)


def coded_bytes(obj) -> int:
    """Bytes of every coded stream under `obj` — the ones NOT on the parameter tree.

    ``EschaWeight`` deliberately hangs off a leading-underscore attribute so the
    coded stream never enters ``Module.parameters()``. That is right for
    training and update semantics and wrong for accounting: anything that sizes
    a model by walking ``parameters()`` sees a coded linear as its bias alone —
    literally zero bytes when there is no bias — so a dense model weighs in at
    a rounding error instead of its real size. Byte ledgers must add this.

    Accepts a Module, or any list/tuple/dict of them (mlx's ``Module`` is itself
    a dict whose values are its children, so the walk is uniform).
    """
    total = 0
    stack = [obj]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, EschaLinear):
            total += sum(a.nbytes for a in node._w.arrays())
            continue
        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    return total


#: Leaf names that together define one coded linear.
LEAVES = frozenset({"escha_code", "escha_rin", "escha_rout",
                    "escha_s_in", "escha_s_out", "escha_config", "bias"})

#: The subset that must be present before a linear can be built.
REQUIRED = frozenset({"escha_code", "escha_rin", "escha_rout"})
