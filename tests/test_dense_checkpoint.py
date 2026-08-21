"""Structural contract of a real dense export, checked without loading weights.

The synthetic test in tests/test_models.py proves the loader works on a
checkpoint this repo writes itself, which cannot catch the export drifting away
from what the loader expects. This one reads only the safetensors HEADERS of an
actual shipped model — shapes and dtypes, no tensor data — and asserts the
things the loader would otherwise discover the expensive way, or not at all:

  * every coded linear carries its full leaf set (a missing rout decodes to
    noise, a missing s_out is a silent ~2% error);
  * the rate declared in config.json's layer_meta matches the rate implied by
    each code tensor's own shape;
  * the set of coded modules is exactly the set of linears the skeleton has —
    no coded tensor without a home, no linear left in fp16 by accident;
  * every coded linear is Hadamard-compatible (both dimensions a multiple of
    128) and tile-aligned, which the kernels assume and do not check.

Skipped unless a dense checkpoint is present (ESCHA_DENSE_MODEL).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from .conftest import DENSE_CKPT, needs_dense_ckpt, needs_slow

CODED_LEAVES = ("escha_code", "escha_rin", "escha_rout",
                "escha_s_in", "escha_s_out", "escha_config")


def _headers(path: Path) -> dict[str, dict]:
    """{tensor name: {dtype, shape}} from every shard's header only."""
    out: dict[str, dict] = {}
    shards = sorted(path.glob("*.safetensors"))
    assert shards, f"no shards under {path}"
    for shard in shards:
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = v
    return out


@pytest.fixture(scope="module")
def dense_ckpt():
    path = Path(DENSE_CKPT)
    return path, _headers(path), json.loads((path / "config.json").read_text())


@needs_dense_ckpt
def test_checkpoint_is_recognised(dense_ckpt):
    from escha_mlx import loader, models

    path, _, config = dense_ckpt
    assert loader.is_escha_checkpoint(path)
    assert models.resolve(config).MODEL_TYPE == config["model_type"]


@needs_dense_ckpt
def test_every_coded_linear_has_its_full_leaf_set(dense_ckpt):
    _, tensors, _ = dense_ckpt
    bases = {n.rsplit(".", 1)[0] for n in tensors if n.endswith(".escha_code")}
    assert bases, "no coded linears found"
    missing = {b: [lf for lf in CODED_LEAVES if f"{b}.{lf}" not in tensors]
               for b in bases}
    assert not any(missing.values()), \
        {b: m for b, m in missing.items() if m}


@needs_dense_ckpt
def test_declared_rate_matches_the_code_stream(dense_ckpt):
    """config.json's layer_meta K vs the K implied by each code tensor's shape.

    These are written by different parts of the exporter; a disagreement means
    a linear would be decoded at the wrong rate — which produces a full-range,
    finite, entirely wrong weight.
    """
    _, tensors, config = dense_ckpt
    meta = config.get("quantization_config", {}).get("layer_meta", {})
    assert meta, "export carries no layer_meta"
    checked = 0
    for base, m in meta.items():
        info = tensors.get(f"{base}.escha_code")
        assert info is not None, f"layer_meta names {base} but it has no code"
        tk, tn, wpt = info["shape"]
        assert wpt % 16 == 0, (base, wpt)
        assert wpt // 16 == m["K"], (base, wpt // 16, m["K"])
        assert (tk * 16, tn * 16) == (m["in_features"], m["out_features"]), base
        checked += 1
    coded = {n.rsplit(".", 1)[0] for n in tensors if n.endswith(".escha_code")}
    assert checked == len(coded), (checked, len(coded))


@needs_dense_ckpt
def test_coded_shapes_satisfy_kernel_preconditions(dense_ckpt):
    """Both dimensions must be a multiple of 128 (one Hadamard block, and the
    GEMV grid covers 128 output channels per threadgroup)."""
    _, tensors, _ = dense_ckpt
    for name, info in tensors.items():
        if not name.endswith(".escha_code"):
            continue
        tk, tn, _ = info["shape"]
        ic, oc = tk * 16, tn * 16
        assert ic % 128 == 0 and oc % 128 == 0, (name, ic, oc)
        assert tensors[name.replace(".escha_code", ".escha_rin")]["shape"] == [ic]
        assert tensors[name.replace(".escha_code", ".escha_rout")]["shape"] == [oc]


@needs_dense_ckpt
def test_coded_modules_are_exactly_the_skeleton_linears(dense_ckpt):
    """No coded tensor without a module to install it on, and no linear left
    unquantized that the export meant to code."""
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_5 as skel
    from escha_mlx.loader import strip_lm_prefix

    _, tensors, config = dense_ckpt
    # Build the skeleton at 2 layers to enumerate module names cheaply, then
    # compare per-layer suffixes rather than absolute names.
    small = json.loads(json.dumps(config))
    tc = small["text_config"]
    tc["num_hidden_layers"] = len(tc["layer_types"])
    model = None
    try:
        import mlx.core as mx  # noqa: F401
        tc["vocab_size"], tc["hidden_size"], tc["intermediate_size"] = 256, 128, 128
        tc["num_attention_heads"], tc["num_key_value_heads"] = 2, 1
        tc["head_dim"] = 64
        tc["linear_num_key_heads"], tc["linear_num_value_heads"] = 2, 4
        tc["linear_key_head_dim"] = tc["linear_value_head_dim"] = 32
        small.pop("vision_config", None)
        model = skel.Model(skel.ModelArgs.from_dict(small))
    except Exception as exc:  # pragma: no cover - skeleton must build
        pytest.fail(f"could not build the skeleton for {config['model_type']}: {exc}")

    linears = {
        n.removeprefix("language_model.model.").removesuffix(".weight")
        for n, a in tree_flatten(model.parameters())
        if n.endswith(".weight") and a.ndim == 2 and ".layers." in n
        and "conv1d" not in n and "norm" not in n
    }
    coded = {strip_lm_prefix(n).rsplit(".", 1)[0]
             for n in tensors if n.endswith(".escha_code")}
    # The small GDN routers (in_proj_a / in_proj_b) ship as fp16 weights.
    fp16_linears = {strip_lm_prefix(n).removesuffix(".weight")
                    for n, i in tensors.items()
                    if n.endswith(".weight") and len(i["shape"]) == 2
                    and ".layers." in n and "conv1d" not in n}
    assert coded | fp16_linears == linears, {
        "coded_without_module": sorted(coded - linears)[:5],
        "module_without_weights": sorted(linears - (coded | fp16_linears))[:5],
    }


@needs_dense_ckpt
def test_embed_and_head_are_int8_pairs(dense_ckpt):
    _, tensors, _ = dense_ckpt
    for base in ("model.language_model.embed_tokens", "lm_head"):
        w8 = tensors[f"{base}.weight_int8"]
        sc = tensors[f"{base}.weight_scale"]
        assert w8["dtype"] == "I8"
        assert sc["shape"] == [w8["shape"][0]], (base, sc["shape"])


@needs_dense_ckpt
@needs_slow
def test_truncated_real_load(monkeypatch):
    """Run the REAL loader over the REAL tensors of the first few layers.

    The synthetic checkpoint proves the pipeline against an export this repo
    writes; this proves it against the one actually shipped — real tensor names,
    real dtypes, the nested text_config/vision_config, and the real mixed rate.
    Truncating the layer count keeps it to a few GB; embed and lm_head are the
    full-size real tensors.
    """
    import copy
    import glob
    import mlx.core as mx
    import numpy as np
    from safetensors import safe_open
    from escha_mlx import models
    from escha_mlx.dense import EschaLinear
    from escha_mlx.loader import LastPositionHead
    from escha_mlx.quant import EschaQ8Embedding, EschaQ8Linear

    monkeypatch.setenv("ESCHA_MLX_LINEAR", "ops")
    monkeypatch.setenv("ESCHA_MLX_BIAS", "1")   # exercise the correction path
    n_layers = 4                     # 3 linear_attention + 1 full_attention
    config = json.loads((Path(DENSE_CKPT) / "config.json").read_text())
    small = copy.deepcopy(config)
    small["text_config"]["num_hidden_layers"] = n_layers
    small["text_config"]["layer_types"] = small["text_config"]["layer_types"][:n_layers]
    assert "full_attention" in small["text_config"]["layer_types"], \
        "truncation must keep at least one attention layer"

    plugin = models.resolve(small).CheckpointLoader(small, 128)
    for shard in sorted(glob.glob(str(Path(DENSE_CKPT) / "*.safetensors"))):
        with safe_open(shard, framework="numpy") as f:
            for name in f.keys():
                if name.startswith("model.language_model.layers."):
                    if int(name.split(".")[3]) >= n_layers:
                        continue
                plugin.consume(name, f.get_tensor(name))
    arrays = plugin.finalize()
    model = plugin.model
    mx.eval(model.parameters())
    mx.eval(arrays)

    layers = model.language_model.model.layers
    for layer in layers:
        if layer.is_linear:
            mods = {n: getattr(layer.linear_attn, n)
                    for n in ("in_proj_qkv", "in_proj_z", "out_proj")}
            # the small GDN routers are not coded
            assert not isinstance(layer.linear_attn.in_proj_a, EschaLinear)
            assert not isinstance(layer.linear_attn.in_proj_b, EschaLinear)
        else:
            mods = {f"{k}_proj": getattr(layer.self_attn, f"{k}_proj") for k in "qkvo"}
        mods.update({n: getattr(layer.mlp, n)
                     for n in ("gate_proj", "up_proj", "down_proj")})
        assert all(isinstance(m, EschaLinear) for m in mods.values()), \
            {k: type(m).__name__ for k, m in mods.items()}
        assert all(m.bias is not None for m in mods.values())   # ESCHA_MLX_BIAS=1
        # the shipped mixed rate
        assert mods["up_proj"].K == 3 and mods["down_proj"].K == 3
        assert mods["gate_proj"].K == 2

    assert isinstance(model.language_model.model.embed_tokens, EschaQ8Embedding)
    head = model.language_model.lm_head
    assert isinstance(head, LastPositionHead) and isinstance(head.inner, EschaQ8Linear)

    # mlx-lm's (1+w) norm shift must have fired: HF stores these centred on 0.
    # Reduce in f32 — a 5120-wide f16 sum loses the tail entirely.
    norm = model.language_model.model.norm.weight
    assert abs(float(mx.mean(norm.astype(mx.float32))) - 1.944138) < 1e-4
    inp = layers[0].input_layernorm.weight
    assert abs(float(mx.mean(inp.astype(mx.float32))) - (1.0 - 0.0334)) < 1e-2
