"""Load an escha (eschamoe) checkpoint into an mlx-lm qwen3_5_moe model.

Consumes the public HF export directly (no conversion artifact):
  * routed experts: `...mlp.experts.{gate_up_proj,down_proj}.escha_{code,rin,rout}`
    (E-stacked; `escha_s_in`/`escha_s_out` are all-ones and `escha_config` is
    redundant — both dropped, like every other escha runtime).
  * dense linears / embed / lm_head: `weight_int8` + `weight_scale` pairs ->
    exact MLX affine-Q8 repack (escha_mlx.quant).
  * everything else fp16 -> mlx-lm's own sanitize (language_model renames,
    conv1d layout, the (1+w) norm shift) + update.

The model skeleton, GDN kernels, attention and KV/state caches are mlx-lm's;
only the MoE block and the quantized dense modules are replaced.
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
import numpy as np
from mlx.utils import tree_unflatten

from . import gdn_cache, moe, quant

logger = logging.getLogger(__name__)

_DROP_LEAVES = {"escha_s_in", "escha_s_out", "escha_config"}


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


def _strip(name: str) -> str:
    return name[len("model.language_model."):] if name.startswith("model.language_model.") else name


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
    from mlx_lm.models import qwen3_5_moe

    # Before any allocation, so the weights themselves land wired.
    apply_wired_limit()

    path = Path(path)
    t0 = time.time()
    config = json.loads((path / "config.json").read_text())
    args = qwen3_5_moe.ModelArgs.from_dict(config)
    model = qwen3_5_moe.Model(args)
    text_args = model.language_model.args
    n_layers = text_args.num_hidden_layers
    top_k = text_args.num_experts_per_tok
    group_size = int(os.environ.get("ESCHA_MLX_Q8_GROUP", str(quant.DEFAULT_GROUP)))

    layers = model.language_model.model.layers

    def _resolve(owner, dotted: str):
        obj = owner
        parts = dotted.split(".")
        for p in parts[:-1]:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        return obj, parts[-1]

    # Streaming single pass: every tensor is converted to its final (mx) form
    # as soon as its dependency group completes, then the numpy copy is freed.
    experts_np: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    experts_mx: dict[tuple[int, str], moe.EschaExperts] = {}
    int8_np: dict[str, dict[str, np.ndarray]] = {}
    shared_np: dict[str, dict[str, np.ndarray]] = {}   # small, held to block build
    mlp_fp16: dict[tuple[int, str], np.ndarray] = {}
    base: dict[str, np.ndarray] = {}
    n_q8 = 0
    dropped = 0
    n_read = 0

    def _install_q8(base_name: str, pair: dict[str, np.ndarray]) -> None:
        nonlocal n_q8
        w8, scale = pair["weight_int8"], pair["weight_scale"]
        if base_name == "lm_head":
            model.language_model.lm_head = quant.make_linear(w8, scale, group_size)
        elif base_name == "embed_tokens":
            model.language_model.model.embed_tokens = quant.make_embedding(w8, scale, group_size)
        elif base_name.startswith("layers."):
            rest = base_name[len("layers."):]
            idx, dotted = rest.split(".", 1)
            parent, attr = _resolve(layers[int(idx)], dotted)
            setattr(parent, attr, quant.make_linear(w8, scale, group_size))
        else:
            raise ValueError(f"unexpected int8 tensor: {base_name}")
        n_q8 += 1

    for name, w in _iter_tensors(path):
        n_read += 1
        if name.startswith("mtp.") or ".visual." in name or name.startswith("visual."):
            dropped += 1
            continue
        s = _strip(name)
        parts = s.split(".")
        if ".mlp.experts." in s:
            layer = int(parts[1])
            proj, leaf = parts[4], parts[5]
            if leaf in _DROP_LEAVES:
                dropped += 1
                continue
            group = experts_np.setdefault((layer, proj), {})
            group[leaf] = w
            if len(group) == 3:
                experts_mx[(layer, proj)] = moe.EschaExperts(
                    group["escha_code"], group["escha_rin"], group["escha_rout"])
                del experts_np[(layer, proj)]
            continue
        if s.endswith(".weight_int8") or s.endswith(".weight_scale"):
            base_name, leaf = s.rsplit(".", 1)
            if ".shared_expert." in s:
                shared_np.setdefault(base_name, {})[leaf] = w
                continue
            pair = int8_np.setdefault(base_name, {})
            pair[leaf] = w
            if len(pair) == 2:
                _install_q8(base_name, pair)
                del int8_np[base_name]
            continue
        if s.endswith(".mlp.gate.weight") or s.endswith(".mlp.shared_expert_gate.weight"):
            mlp_fp16[(int(parts[1]), parts[3])] = w
            continue
        base[name] = w
    assert not experts_np and not int8_np, (list(experts_np), list(int8_np))
    logger.info("escha_mlx: streamed %d tensors in %.1fs", n_read, time.time() - t0)

    # ---- MoE blocks ------------------------------------------------------
    escha_arrays: list[mx.array] = []
    for i in range(n_layers):
        gu = experts_mx.pop((i, "gate_up_proj"))
        dn = experts_mx.pop((i, "down_proj"))
        assert gu.K == 2 and dn.K == 3, (gu.K, dn.K)
        pref = f"layers.{i}.mlp.shared_expert"
        shared = {}
        for p in ("gate", "up", "down"):
            pair = shared_np.pop(f"{pref}.{p}_proj")
            shared[f"{p}_w8"] = pair["weight_int8"]
            shared[f"{p}_scale"] = pair["weight_scale"]
        block = moe.EschaSparseMoeBlock(
            hidden_size=text_args.hidden_size,
            num_experts=text_args.num_experts,
            top_k=top_k,
            gu=gu, dn=dn,
            gate_w=mlp_fp16.pop((i, "gate")),
            shg_w=mlp_fp16.pop((i, "shared_expert_gate")),
            shared=shared,
            group_size=group_size,
        )
        layers[i].mlp = block
        escha_arrays += gu.arrays() + dn.arrays()
    assert not experts_mx, f"unconsumed expert tensors: {list(experts_mx)[:4]}"

    # ---- fp16 remainder through mlx-lm's own sanitize -------------------
    assert any(k.endswith("conv1d.weight") and v.shape[-1] != 1 for k, v in base.items()), \
        "conv1d already sanitized? norm (1+w) shift heuristic would not fire"
    sanitized = model.sanitize({k: mx.array(v) for k, v in base.items()})
    model.update(tree_unflatten(list(sanitized.items())))

    if use_last_logit():
        model.language_model.lm_head = LastPositionHead(
            model.language_model.lm_head)
        logger.info("escha_mlx: LM head restricted to the last position "
                    "(ESCHA_MLX_LAST_LOGIT=0 for per-position logits)")

    mx.eval(model.parameters())
    mx.eval(escha_arrays)
    model.eval()
    logger.info("escha_mlx: model ready in %.1fs (%d MoE layers, %d Q8 dense, %d dropped)",
                time.time() - t0, n_layers, n_q8, dropped)
    return model


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
    gdn_cache.install(model)
    tokenizer = load_tokenizer(path, tokenizer_config_extra=tokenizer_config or {},
                               eos_token_ids=eos_ids)
    return model, tokenizer
