"""Architecture registry + synthetic end-to-end checkpoint test.

The synthetic test is the CI stand-in for the 12.3 GB model: it writes a tiny
but format-faithful eschamoe export (coded expert trios, Q8 pairs, raw-HF fp16
remainder including the unsanitized conv1d layout), then runs the REAL
`load_model` pipeline — registry dispatch, streaming consume, module swaps,
mlx-lm sanitize, post-load quirks — and a forward pass. Every new architecture
plugin must add its own variant of this test.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from .conftest import needs_mlx

E, H, INTER, VOCAB, LAYERS = 4, 256, 128, 512, 2

TINY_CONFIG = {
    "model_type": "qwen3_5_moe",
    "hidden_size": H,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": VOCAB,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 2,     # layer 0 linear (GDN), layer 1 full attn
    "num_experts": E,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "moe_intermediate_size": INTER,
    "shared_expert_intermediate_size": INTER,
    "tie_word_embeddings": False,
    "quantization_config": {"quant_method": "eschamoe", "bits": 2.0},
}

_NORM_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                  "model.norm.weight", ".q_norm.weight", ".k_norm.weight")


def test_resolve_known_and_unknown():
    from escha_mlx import models
    arch = models.resolve({"model_type": "qwen3_5_moe"})
    assert arch.MODEL_TYPE == "qwen3_5_moe"
    assert hasattr(arch, "CheckpointLoader")
    with pytest.raises(ValueError, match="qwen3_5_moe"):
        models.resolve({"model_type": "made_up_arch"})


def _q8_pair(rng, out_dim, in_dim):
    return (rng.integers(-128, 128, size=(out_dim, in_dim), dtype=np.int8),
            (rng.random(out_dim) * 0.01 + 1e-3).astype(np.float16))


def _write_tiny_checkpoint(path, rng):
    """Format-faithful tiny export: raw HF names, unsanitized conv/norm layout."""
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_5_moe as skel
    from safetensors.numpy import save_file

    ref_model = skel.Model(skel.ModelArgs.from_dict(TINY_CONFIG))
    t: dict[str, np.ndarray] = {}

    for i in range(LAYERS):
        pref = f"model.language_model.layers.{i}.mlp."
        # coded expert trios: gate_up K=2, down K=3
        t[pref + "experts.gate_up_proj.escha_code"] = rng.integers(
            -32768, 32768, size=(E, H // 16, (2 * INTER) // 16, 32), dtype=np.int16)
        t[pref + "experts.gate_up_proj.escha_rin"] = (
            rng.standard_normal((E, H)) * 0.05).astype(np.float16)
        t[pref + "experts.gate_up_proj.escha_rout"] = (
            rng.standard_normal((E, 2 * INTER)) * 0.05).astype(np.float16)
        t[pref + "experts.down_proj.escha_code"] = rng.integers(
            -32768, 32768, size=(E, INTER // 16, H // 16, 48), dtype=np.int16)
        t[pref + "experts.down_proj.escha_rin"] = (
            rng.standard_normal((E, INTER)) * 0.05).astype(np.float16)
        t[pref + "experts.down_proj.escha_rout"] = (
            rng.standard_normal((E, H)) * 0.05).astype(np.float16)
        # redundant leaves the loader must DROP
        t[pref + "experts.gate_up_proj.escha_s_in"] = np.ones((E, H), dtype=np.float16)
        t[pref + "experts.gate_up_proj.escha_config"] = np.zeros((4,), dtype=np.int32)
        # shared expert Q8 + fp16 gates
        for p, (o, k) in (("gate", (INTER, H)), ("up", (INTER, H)), ("down", (H, INTER))):
            w8, sc = _q8_pair(rng, o, k)
            t[pref + f"shared_expert.{p}_proj.weight_int8"] = w8
            t[pref + f"shared_expert.{p}_proj.weight_scale"] = sc
        t[pref + "gate.weight"] = (rng.standard_normal((E, H)) * 0.1).astype(np.float16)
        t[pref + "shared_expert_gate.weight"] = (
            rng.standard_normal((1, H)) * 0.1).astype(np.float16)

    # embed / lm_head as Q8 pairs
    w8, sc = _q8_pair(rng, VOCAB, H)
    t["model.language_model.embed_tokens.weight_int8"] = w8
    t["model.language_model.embed_tokens.weight_scale"] = sc
    w8, sc = _q8_pair(rng, VOCAB, H)
    t["lm_head.weight_int8"] = w8
    t["lm_head.weight_scale"] = sc

    # one dense layer linear as Q8 (exercises the layers.* install path)
    q8_dense = "layers.1.self_attn.q_proj.weight"
    params = dict(tree_flatten(ref_model.parameters()))
    q_shape = params["language_model.model." + q8_dense].shape
    w8, sc = _q8_pair(rng, q_shape[0], q_shape[1])
    t["model.language_model." + q8_dense + "_int8"] = w8
    t["model.language_model." + q8_dense.replace(".weight", ".weight_scale")] = sc

    # tensors the loader must drop outright
    t["mtp.head.weight"] = np.zeros((4, 4), dtype=np.float16)
    t["model.visual.patch_embed.weight"] = np.zeros((4, 4), dtype=np.float16)

    # fp16 remainder in RAW HF form (inverse of mlx-lm's sanitize)
    for name, arr in params.items():
        if ".mlp." in name:
            continue                       # replaced by the escha MoE block
        if "embed_tokens" in name or name.startswith("language_model.lm_head"):
            continue                       # Q8 above
        if name == "language_model.model." + q8_dense:
            continue                       # Q8 above
        v = np.array(arr).astype(np.float16)
        if name.endswith("conv1d.weight"):
            v = np.moveaxis(v, 1, 2)       # skeleton [C,K,1] -> HF [C,1,K]
        if any(name.endswith(sfx) for sfx in _NORM_SUFFIXES) and v.ndim == 1:
            v = v - 1.0                    # HF stores w-1; sanitize adds it back
        if name.startswith("language_model.model."):
            hf = name.replace("language_model.model.", "model.language_model.", 1)
        else:
            hf = name.removeprefix("language_model.")
        t[hf] = v

    save_file(t, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(TINY_CONFIG))
    (path / "quantize_config.json").write_text(
        json.dumps({"quant_method": "eschamoe", "bits": 2.0}))


@needs_mlx
def test_synthetic_checkpoint_end_to_end(tmp_path):
    import mlx.core as mx
    from escha_mlx.loader import LastPositionHead, load_model
    from escha_mlx.models.qwen3_5_moe import EschaSparseMoeBlock
    from escha_mlx.quant import EschaQ8Embedding, EschaQ8Linear

    _write_tiny_checkpoint(tmp_path, np.random.default_rng(0))
    model = load_model(tmp_path)

    layers = model.language_model.model.layers
    assert all(isinstance(l.mlp, EschaSparseMoeBlock) for l in layers)
    assert isinstance(model.language_model.model.embed_tokens, EschaQ8Embedding)
    assert isinstance(layers[1].self_attn.q_proj, EschaQ8Linear)
    head = model.language_model.lm_head
    assert isinstance(head, LastPositionHead) and isinstance(head.inner, EschaQ8Linear)

    def forward():
        cache = model.make_cache()
        out = model(mx.array([[1, 2, 3, 4]]), cache=cache)
        mx.eval(out, [c.state for c in cache])
        return np.array(out.astype(mx.float32)), cache

    out1, cache1 = forward()
    assert out1.shape == (1, 1, VOCAB)          # LastPositionHead: final position only
    assert np.isfinite(out1).all()
    # The allocation-free first GDN kernel must emit fp16 state directly. If it
    # regresses to upstream, a large first forward briefly keeps both the f32
    # output state and the cache's lazy fp16 cast alive.
    assert cache1[0][1].dtype == mx.float16
    out2, _ = forward()
    assert np.array_equal(out1, out2), "synthetic forward must be bit-deterministic"
