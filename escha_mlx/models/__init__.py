"""Architecture registry — one plugin module per supported `model_type`.

The codec (ref/msl/quant) is the engine and never forks per architecture.
Each architecture contributes exactly one module here, resolved from the
checkpoint's `config.json` `model_type` (the same convention mlx-lm uses),
and exports:

    MODEL_TYPE: str
        The config.json model_type this plugin serves.

    class CheckpointLoader:
        __init__(config: dict, group_size: int)
            Build the mlx-lm skeleton (available as `.model`) before any
            tensor is read.
        consume(name: str, w: np.ndarray) -> None
            One streamed tensor, raw HF name. Group coded trios / Q8 pairs,
            install as soon as a group completes, stash the fp16 remainder.
        finalize() -> list[mx.array]
            Install whatever needed the full stream (MoE blocks), run the
            skeleton's sanitize over the fp16 remainder, apply post-load
            quirks (cache patches, head wrappers). Returns the off-tree
            arrays the generic loader must mx.eval.

The generic side (streaming, Q8 repack, wired limit, tokenizer/eos handling)
lives in escha_mlx.loader and is shared by every plugin. To add an
architecture, copy qwen3_5_moe.py as the template, add goldens under
tests/data/<model_type>/, extend the synthetic-checkpoint test, and register
the module below.
"""
from __future__ import annotations

from importlib import import_module

REGISTRY: dict[str, str] = {
    "qwen3_5_moe": ".qwen3_5_moe",
}


def resolve(config: dict):
    """Return the architecture plugin module for this checkpoint config."""
    model_type = config.get("model_type")
    if model_type not in REGISTRY:
        raise ValueError(
            f"escha_mlx: unsupported model_type {model_type!r} — supported: "
            f"{sorted(REGISTRY)}. New architectures: see escha_mlx/models/__init__.py "
            f"and CONTRIBUTING.md.")
    return import_module(REGISTRY[model_type], __name__)
