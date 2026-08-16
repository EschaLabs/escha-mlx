"""Native (stock-MLX-quantized) load path — synthetic end-to-end checkpoint.

The CI stand-in for a real 4-bit export: writes a tiny checkpoint through
mlx-lm's OWN quantizer (`mlx_lm.utils.quantize_model`, the code path that
produced the published ones, per-layer 8-bit gate overrides included), then
runs the REAL `escha_mlx.load_model` — format dispatch, mlx-lm build, quirk
installation — and a forward pass.

What this gates is the RUNTIME WIRING, not numerics: there is no escha codec on
this path, so there is nothing to hold bit-exact against a reference (see
escha_mlx/native.py). The numerics belong to MLX's affine kernels.
"""
from __future__ import annotations

import json

import pytest

from .conftest import needs_mlx

H, INTER, VOCAB, LAYERS, E = 256, 128, 512, 2, 4
GROUP, BITS = 64, 4

NATIVE_CONFIG = {
    "model_type": "qwen3_5_moe",
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": H,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "vocab_size": VOCAB,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,   # layer 0 linear (GDN), layer 1 full attn
        "num_experts": E,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "moe_intermediate_size": INTER,
        "shared_expert_intermediate_size": INTER,
        "tie_word_embeddings": False,
    },
}


def _write_tiny_native_checkpoint(path):
    """Tiny export written by mlx-lm's own quantizer — nothing escha-specific."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_5_moe as skel
    from mlx_lm.utils import quantize_model

    model = skel.Model(skel.ModelArgs.from_dict(NATIVE_CONFIG))
    model, config = quantize_model(model, dict(NATIVE_CONFIG),
                                   group_size=GROUP, bits=BITS)
    path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path / "model.safetensors"),
                        dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(config))
    return config


@needs_mlx
def test_detection_does_not_claim_eschamoe_or_foreign_checkpoints(tmp_path):
    from escha_mlx import handles, is_escha_checkpoint, native

    escha = tmp_path / "escha"
    escha.mkdir()
    (escha / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_5_moe",
         "quantization_config": {"quant_method": "eschamoe", "bits": 2.0}}))
    # The codec path owns it: the native path must not claim it...
    assert native.native_model_type(escha) is None
    assert not native.can_load(escha)
    # ...and `handles` still routes it here, via is_escha_checkpoint.
    assert is_escha_checkpoint(escha) and handles(escha)

    other = tmp_path / "llama"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert native.native_model_type(other) == "llama"
    assert not native.can_load(other)      # not a validated architecture
    assert not handles(other)              # -> left to mlx-lm verbatim
    with pytest.raises(ValueError, match="not validated on the native"):
        native.load_model(other)

    missing = tmp_path / "nope"            # e.g. an undownloaded HF repo id
    assert native.native_model_type(missing) is None and not handles(missing)


@needs_mlx
def test_native_any_env_lifts_the_architecture_gate(tmp_path, monkeypatch):
    from escha_mlx import handles, native

    other = tmp_path / "llama"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "llama"}))
    monkeypatch.setenv("ESCHA_MLX_NATIVE_ANY", "1")
    assert native.can_load(other) and handles(other)


@needs_mlx
def test_synthetic_native_checkpoint_loads_and_runs(tmp_path, monkeypatch):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from escha_mlx import handles, is_escha_checkpoint, load_model
    from escha_mlx.gdn_cache import GDNStateCache
    from escha_mlx.loader import LastPositionHead

    ckpt = tmp_path / "tiny-native-4bit"
    config = _write_tiny_native_checkpoint(ckpt)

    # The export really is per-layer quantized the way the published ones are.
    assert config["quantization"]["bits"] == BITS
    assert config["quantization"][f"language_model.model.layers.0.mlp.gate"] == {
        "group_size": 64, "bits": 8}
    assert not is_escha_checkpoint(ckpt) and handles(ckpt)

    monkeypatch.setenv("ESCHA_MLX_GDN_STATE", "fp16")
    model = load_model(ckpt)

    # -- quirks installed -------------------------------------------------
    assert isinstance(model.language_model.lm_head, LastPositionHead)
    cache = model.make_cache()
    assert isinstance(cache[0], GDNStateCache)   # layer 0 is linear (GDN)
    assert cache[0].gdn_dtype == mx.float16
    assert isinstance(cache[1], KVCache)         # layer 1 is full attention

    # -- forward ----------------------------------------------------------
    ids = mx.array([[3, 7, 11, 13, 17]])
    logits = model(ids, cache=cache)
    mx.eval(logits)
    assert logits.shape == (1, 1, VOCAB), "last-position head should trim prefill"
    assert mx.isfinite(logits).all().item()

    step = model(mx.array([[23]]), cache=cache)
    mx.eval(step)
    assert step.shape == (1, 1, VOCAB)
    assert cache[0][1] is not None and cache[0][1].dtype == mx.float16
    assert mx.isfinite(step).all().item()


@needs_mlx
def test_last_logit_opt_out_restores_per_position_logits(tmp_path, monkeypatch):
    import mlx.core as mx

    from escha_mlx import load_model
    from escha_mlx.loader import LastPositionHead

    ckpt = tmp_path / "tiny-native-4bit"
    _write_tiny_native_checkpoint(ckpt)
    monkeypatch.setenv("ESCHA_MLX_LAST_LOGIT", "0")
    model = load_model(ckpt)
    assert not isinstance(model.language_model.lm_head, LastPositionHead)
    logits = model(mx.array([[3, 7, 11, 13, 17]]), cache=model.make_cache())
    mx.eval(logits)
    assert logits.shape == (1, 5, VOCAB)
