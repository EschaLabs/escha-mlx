"""escha-mlx — the escha 2-bit trellis runtime for Apple Silicon (MLX).

Usage:
    from escha_mlx import load
    model, tokenizer = load("/path/to/Qwen3.6-35B-A3B-Escha-W2")

CLI:
    python -m escha_mlx.generate --model <dir> --prompt "..."
    python -m escha_mlx.server --model <dir> --port 8080
"""
from __future__ import annotations

__version__ = "0.1.0"  # keep in sync with pyproject.toml [project] version

from .loader import is_escha_checkpoint, load, load_model  # noqa: F401
