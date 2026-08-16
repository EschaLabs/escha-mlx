"""Stock-MLX-quantized checkpoints (non-eschamoe) through the escha runtime.

WHY THIS PATH EXISTS.  escha-mlx's model skeleton *is* mlx-lm's — only the MoE
block and the dense linears are swapped for the trellis codec; attention, the
GDN kernels, the KV/state caches and the generation loop are upstream,
untouched.  So the runtime layer wrapped around the codec is not codec-specific
at all:

  * the GDN recurrent-state cache (fp16 storage + the allocation-free
    first-state Metal kernel, escha_mlx.gdn_cache) patches
    `mlx_lm.models.qwen3_5.gated_delta_update` and the model's `make_cache`.
    Neither knows anything about how the weights are stored.
  * the last-position LM head (escha_mlx.loader.LastPositionHead) is a wrapper
    around whatever `lm_head` module the skeleton already has — with a 248 320
    vocabulary it saves the same prefill GEMM either way.
  * the wired-limit policy and the server's continuous batching / ignore_eos
    are storage-agnostic by construction.

A checkpoint MLX quantized itself therefore runs here with no codec involved:
mlx-lm builds and quantizes the skeleton exactly as `mlx_lm.load` would, and
escha installs its own quirks on top.

WHAT THIS PATH IS NOT.  Nothing here is bit-exact-gated, because there is no
escha codec in it — the weights are dequantized by MLX's own affine kernels and
those numerics are upstream's to define.  `tests/` gates the codec, not this.
The 2-bit footprint is not available either: a stock 4-bit export of a
35B-class MoE is ~19.5 GB resident, versus 12.3 GB for the W2 build.  What this
path buys is escha's serving and memory layer on an ordinary MLX checkpoint,
not escha's compression.

SUPPORTED ARCHITECTURES.  `NATIVE_ARCHITECTURES` lists the `model_type`s whose
escha extras are known to apply — the Qwen3.5/3.6 hybrid GDN + attention family,
which is the family the GDN state cache exists for.  Anything else is refused by
name rather than silently served with a runtime whose assumptions it may not
meet; `ESCHA_MLX_NATIVE_ANY=1` lifts the refusal and installs only the quirks
that structurally apply (measure before trusting it).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from . import gdn_cache
from .loader import LastPositionHead, apply_wired_limit, use_last_logit

logger = logging.getLogger(__name__)

# model_type -> the escha runtime is known to fit this architecture.
NATIVE_ARCHITECTURES = frozenset({"qwen3_5_moe", "qwen3_5"})


def allow_any() -> bool:
    """ESCHA_MLX_NATIVE_ANY=1 — serve any mlx-lm architecture (unvalidated)."""
    return os.environ.get("ESCHA_MLX_NATIVE_ANY", "0") != "0"


def native_model_type(path: str | Path) -> str | None:
    """`model_type` of a local non-eschamoe MLX checkpoint, else None.

    Deliberately local-directory only: a Hugging Face repo id that has not been
    downloaded yet has no config to read, and escha has no business intercepting
    a load it cannot first inspect — mlx-lm handles those unchanged.
    """
    cfg_file = Path(path) / "config.json"
    if not cfg_file.is_file():
        return None
    try:
        config = json.loads(cfg_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    qc = config.get("quantization_config") or {}
    if isinstance(qc, dict) and qc.get("quant_method") == "eschamoe":
        return None      # the codec path owns this checkpoint
    mt = config.get("model_type")
    return mt if isinstance(mt, str) else None


def can_load(path: str | Path) -> bool:
    """True when escha_mlx should serve this checkpoint through the native path."""
    mt = native_model_type(path)
    return mt is not None and (mt in NATIVE_ARCHITECTURES or allow_any())


def _text_model(model):
    """The TextModel that owns lm_head / layers (VLM wrappers nest it)."""
    return getattr(model, "language_model", model)


def _has_gdn(model) -> bool:
    lm = _text_model(model)
    layers = getattr(getattr(lm, "model", None), "layers", None)
    return bool(layers) and any(getattr(l, "is_linear", False) for l in layers)


def install_quirks(model) -> list[str]:
    """Apply the storage-agnostic escha runtime extras. Returns what was applied."""
    applied: list[str] = []

    if use_last_logit():
        lm = _text_model(model)
        head = getattr(lm, "lm_head", None)
        if head is None:
            # Tied embeddings: the head is `embed_tokens.as_linear` inside the
            # skeleton's own __call__, with no module to wrap. Not an error —
            # just no saving available.
            logger.info("escha_mlx: tied word embeddings — no lm_head module to "
                        "restrict to the last position")
        else:
            lm.lm_head = LastPositionHead(head)
            applied.append("last-position-head")
            logger.info("escha_mlx: LM head restricted to the last position "
                        "(ESCHA_MLX_LAST_LOGIT=0 for per-position logits)")

    if _has_gdn(model):
        gdn_cache.install(model)
        applied.append("gdn-state-cache")
    else:
        logger.info("escha_mlx: no GDN layers in this model — recurrent-state "
                    "cache not installed")
    return applied


def load_model(path: str | Path):
    """Build a stock-MLX checkpoint through mlx-lm, then install escha's quirks."""
    from mlx_lm.utils import load_model as mlx_load_model

    path = Path(path)
    model_type = native_model_type(path)
    if model_type is None:
        raise ValueError(
            f"{path} is not a readable non-eschamoe MLX checkpoint "
            "(no config.json, or it declares quant_method=eschamoe)")
    if model_type not in NATIVE_ARCHITECTURES and not allow_any():
        raise ValueError(
            f"escha_mlx: model_type {model_type!r} is not validated on the native "
            f"(stock-MLX) path — validated: {sorted(NATIVE_ARCHITECTURES)}. Set "
            "ESCHA_MLX_NATIVE_ANY=1 to try it anyway (unmeasured), or use "
            "mlx_lm directly.")

    # Before any allocation, so the weights themselves land wired.
    apply_wired_limit()

    t0 = time.time()
    model, _config = mlx_load_model(path)
    applied = install_quirks(model)
    model.eval()
    logger.info("escha_mlx: native %s model ready in %.1fs (quirks: %s)",
                model_type, time.time() - t0, ", ".join(applied) or "none")
    return model
