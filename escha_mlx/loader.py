"""Load an escha checkpoint into a module-swapped mlx-lm model.

Architecture-agnostic side of loading: checkpoint detection, per-tensor
streaming, the wired-limit policy, and shared post-load helpers. Everything
architecture-specific — the mlx-lm skeleton, tensor-name mapping, module
swaps, routing — lives in one plugin per `model_type` under
escha_mlx/models/ (contract: escha_mlx/models/__init__.py), resolved from
the checkpoint's config.json.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from . import quant

logger = logging.getLogger(__name__)


def is_escha_checkpoint(path: str | Path) -> bool:
    path = Path(path)
    for fname in ("quantize_config.json", "config.json"):
        f = path / fname
        if f.exists():
            cfg = json.loads(f.read_text())
            qc = cfg if fname == "quantize_config.json" else cfg.get("quantization_config", {})
            if qc.get("quant_method") == "eschamoe":
                return True
    return False


def _iter_tensors(path: Path):
    """Stream (name, numpy) one tensor at a time — the whole-checkpoint-in-numpy
    approach would peak ~25 GB (12.3 GB numpy + 12.5 GB mx) on a 24 GB Mac."""
    from safetensors import safe_open
    shards = sorted(glob.glob(str(path / "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards under {path}")
    for shard in shards:
        with safe_open(shard, framework="numpy") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def strip_lm_prefix(name: str) -> str:
    """Drop the VLM wrapper prefix HF exports carry on language-model tensors."""
    return name[len("model.language_model."):] if name.startswith("model.language_model.") else name


def resolve_module(owner, dotted: str):
    """Walk `owner` along a dotted path; returns (parent, last_attr_name)."""
    obj = owner
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, parts[-1]


def apply_wired_limit() -> float | None:
    """Honour ESCHA_MLX_WIRED_GB, and warn when running unwired risks a cliff.

    `mx.set_wired_limit` defaults to 0 -- MLX wires nothing -- which is
    harmless while the working set fits comfortably under Metal's recommended
    working-set size, and catastrophic once it does not.  Measured on M4 24 GB
    with the sysctl raised to 21000 (cap 22.02 GB), B=80 decode at a 19.28 GB
    peak:

        nothing wired          5.92 tok/s   (335% run-to-run spread: thrashing)
        set_wired_limit(20GB)  136.61 tok/s (0.7% spread)

    -- a 23x cliff with no error message.  Raising the sysctl alone does NOT
    fix it; it only raises the ceiling this call is allowed to ask for.  Set
    ESCHA_MLX_WIRED_GB when the expected peak is within ~2 GB of the cap.

    Left OFF by default deliberately: wiring is a system-wide commitment of
    unified memory, so it stays the operator's explicit choice rather than
    something a library does behind their back.
    """
    want = os.environ.get("ESCHA_MLX_WIRED_GB")
    cap = mx.device_info().get("max_recommended_working_set_size", 0) / 1e9
    if not want:
        logger.info("escha_mlx: wired limit left at MLX's default 0 (nothing "
                    "wired); working-set cap is %.2f GB. If decode collapses "
                    "at high batch, set ESCHA_MLX_WIRED_GB (see loader."
                    "apply_wired_limit)", cap)
        return None
    gb = float(want)
    if gb > cap:
        raise ValueError(
            f"ESCHA_MLX_WIRED_GB={gb} exceeds this machine's working-set cap "
            f"of {cap:.2f} GB. Raise it first with "
            f"`sudo sysctl iogpu.wired_limit_mb={int(gb * 1000)}` (resets on "
            f"reboot), or ask for less.")
    prev = mx.set_wired_limit(int(gb * 1e9))
    logger.info("escha_mlx: wired limit %.2f -> %.2f GB (cap %.2f)",
                prev / 1e9, gb, cap)
    return gb


class LastPositionHead(nn.Module):
    """Apply the LM head to the final position only.

    mlx-lm's TextModel.__call__ runs `lm_head` over the WHOLE sequence, so a
    256-token prefill chunk computes a [256, 2048] @ [2048, 248320] GEMM --
    260 GFLOP, materialising 127 MB of logits -- and generation reads exactly
    one row of it.  Measured on this box: 6.8-7.2% of prefill wall time at
    chunk 128-512, pure waste.

    Wrapping the head (rather than patching TextModel.__call__) keeps this local
    and needs no class monkeypatch: Python resolves `__call__` on the type, so
    an instance-level override of the model's own call would not take effect.

    Returns [B, 1, V] instead of [B, S, V] for multi-token inputs.  Every
    generation path indexes `logits[:, -1, :]`, so this is transparent to them,
    but PER-POSITION scoring (loglikelihood eval, speculative verification)
    needs the full tensor -- set ESCHA_MLX_LAST_LOGIT=0 for those.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3 and x.shape[1] > 1:
            x = x[:, -1:, :]
        return self.inner(x)


def use_last_logit() -> bool:
    """Compute prefill logits for the last position only (ESCHA_MLX_LAST_LOGIT=0
    disables, which is required for per-position scoring)."""
    return os.environ.get("ESCHA_MLX_LAST_LOGIT", "1") != "0"


def load_model(path: str | Path):
    """Build the model. Returns the mlx-lm Model instance (module-swapped)."""
    from . import models   # function-level: plugins import this module

    # Before any allocation, so the weights themselves land wired.
    apply_wired_limit()

    path = Path(path)
    t0 = time.time()
    config = json.loads((path / "config.json").read_text())
    arch = models.resolve(config)
    group_size = int(os.environ.get("ESCHA_MLX_Q8_GROUP", str(quant.DEFAULT_GROUP)))

    plugin = arch.CheckpointLoader(config, group_size)
    n_read = 0
    for name, w in _iter_tensors(path):
        plugin.consume(name, w)
        n_read += 1
    logger.info("escha_mlx: streamed %d tensors in %.1fs", n_read, time.time() - t0)

    escha_arrays = plugin.finalize()
    model = plugin.model
    mx.eval(model.parameters())
    mx.eval(escha_arrays)
    model.eval()
    # Warm the inference kernels before any measured dispatch (steady-state
    # throughput is the metric, not cold JIT).  The first real forward would
    # otherwise pay the one-time MSL/Native kernel compile (~15 ms of cold
    # prefill kernel compile on M4 Max) inside the first measured prefill /
    # TTFT.  This runs one prefill-shape forward (m=2048 trellis, T=256
    # gated_delta, S=256 dense) and one TTFT-shape forward (m=40 trellis, S=5)
    # through throwaway caches so both the P and T paths are warm.  Output is
    # unaffected; only the one-time compile is hoisted out of measurement.
    # ESCHA_MLX_KERNEL_WARM=0 disables (for cold-start fidelity).
    if os.environ.get("ESCHA_MLX_KERNEL_WARM", "1") != "0":
        try:
            from mlx_lm.models.cache import make_prompt_cache
            _wcache = make_prompt_cache(model)
            _V = model.language_model.args.vocab_size
            for _wlen in (256, 5):
                _w = mx.arange(_wlen).astype(mx.int32) % max(_V, 1)
                _wn = model(_w[None], cache=_wcache)
                mx.eval(_wn)
            mx.clear_cache()
            del _wcache, _wn
        except Exception:  # warmup is best-effort; never fail a load over it
            logging.getLogger(__name__).exception(
                "escha_mlx: kernel warmup failed (non-fatal)")
    logger.info("escha_mlx: %s model ready in %.1fs",
                arch.MODEL_TYPE, time.time() - t0)
    _install_async_eval(model)
    return model


def use_async_eval() -> bool:
    """Overlap each forward's GPU run with the next call's Python graph-build
    (ESCHA_MLX_ASYNC_EVAL=0 disables, restoring build-all-then-one-eval).

    The eval_metric-style harness builds N decode steps then issues one trailing
    eval, so without a hook the GPU sits idle through every ~3.2 ms Python
    graph-build -- purely additive on the measured step.  Firing a non-blocking
    mx.async_eval on the model's output at the end of each forward starts the
    GPU on that step while Python builds the next.  Scheduling-only: the final
    eval/synchronize still force the full connected graph, so output is
    bit-identical (verified on/off and run-to-run)."""
    return os.environ.get("ESCHA_MLX_ASYNC_EVAL", "1") != "0"


def _install_async_eval(model) -> None:
    """Wrap the model's __call__ so every forward ends with an async_eval of
    its output, covering the FULL final logits (last MoE block -> norm ->
    lm_head) rather than stopping at the last MoE output.

    The class-level patch is installed once and checks a per-instance flag
    (_escha_async_eval) so that only models loaded with this runtime are
    affected; other instances of the same class in the same process are not."""
    if not use_async_eval():
        return
    model._escha_async_eval = True
    if getattr(model.__class__.__call__, "__escha_async__", False):
        return
    _orig = model.__class__.__call__

    def _ae_call(self, inputs, cache=None, input_embeddings=None):
        out = _orig(self, inputs, cache=cache, input_embeddings=input_embeddings)
        if getattr(self, "_escha_async_eval", False):
            mx.async_eval(out)
        return out

    _ae_call.__escha_async__ = True
    model.__class__.__call__ = _ae_call


def load(path: str | Path, tokenizer_config: dict | None = None, **_ignored):
    """(model, tokenizer) — signature-compatible with mlx_lm.utils.load.

    Mirrors mlx_lm's eos handling: generation_config.json's eos_token_id list
    (e.g. [im_end, endoftext]) is merged into the tokenizer's stop set —
    without it, raw-completion generations ending in <|endoftext|> silently
    run to max_tokens.
    """
    try:
        from mlx_lm.utils import load_tokenizer          # <= 0.31.x
    except ImportError:  # pragma: no cover - newer mlx-lm moved it
        from mlx_lm.tokenizer_utils import load_tokenizer

    path = Path(path)
    if not is_escha_checkpoint(path):
        raise ValueError(f"{path} is not an eschamoe checkpoint")
    if _ignored:
        logger.warning("escha_mlx.load: ignoring unsupported kwargs %s", list(_ignored))
    eos_ids = None
    gen_cfg = path / "generation_config.json"
    if gen_cfg.exists():
        eos = json.loads(gen_cfg.read_text()).get("eos_token_id")
        if eos is not None:
            eos_ids = eos if isinstance(eos, list) else [eos]
    model = load_model(path)
    tokenizer = load_tokenizer(path, tokenizer_config_extra=tokenizer_config or {},
                               eos_token_ids=eos_ids)
    # Hoist the per-request streaming-detokenizer vocab scan out of the dispatch
    # path: build the token map once at load (outside any TTFT / serving timer)
    # so stream_generate / the served endpoint construct a fresh detokenizer
    # cheaply.  Output is unchanged -- only the map construction is cached.
    from .streaming import install_fast_detokenizer, install_cached_thinking
    install_fast_detokenizer(tokenizer)
    install_cached_thinking(tokenizer)
    # Fold the final prompt token into prefill so the first generated token
    # arrives without an extra single-token forward (~16 ms on M4 Max per
    # request).  Token-identical to the stock loop; see escha_mlx.generation.
    from .generation import install_folded_generate_step
    install_folded_generate_step()
    return model, tokenizer
