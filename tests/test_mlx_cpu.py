"""MLX-backend tests that do NOT need Metal (run on Linux CPU or macOS)."""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import needs_mlx

pytestmark = needs_mlx


@pytest.mark.parametrize("group", [32, 64, 128])
def test_q8_repack_bit_exact(group):
    from escha_mlx import quant
    quant._VALIDATED.discard(group)
    quant.validate_pack(group)


def test_q8_group_size_is_numerically_free():
    """group 128 must be BIT-IDENTICAL to group 64, not merely close.

    escha scales are per-output-channel, so pack_q8 replicates one constant
    across every group of a row -- the group size changes only how many copies
    of that constant are stored (8 bytes each).  This is the gate behind
    defaulting to 128 for a ~120 MB/token byte saving: if MLX ever made the
    dequant group-dependent, this fails instead of silently shifting numerics.
    """
    import mlx.core as mx
    from escha_mlx import quant
    rng = np.random.default_rng(3)
    w8 = rng.integers(-128, 128, size=(6, 1024), dtype=np.int8)
    scale = (rng.random(6).astype(np.float32) * 0.05 + 1e-3).astype(np.float16)
    x = mx.array(rng.standard_normal((3, 1024)).astype(np.float16))
    outs = {}
    for g in (32, 64, 128):
        assert quant.fit_group(1024, g) == g
        outs[g] = np.array(quant.make_linear(w8, scale, g)(x))
    for g in (32, 128):
        assert np.array_equal(outs[g].view(np.uint16), outs[64].view(np.uint16)), (
            f"group {g} diverged from group 64: "
            f"max|d|={np.abs(outs[g].astype(np.float32) - outs[64].astype(np.float32)).max()}")


def test_fit_group_degrades_on_odd_shapes():
    from escha_mlx import quant
    assert quant.fit_group(2048) == 128          # every shipped tensor
    assert quant.fit_group(512) == 128
    assert quant.fit_group(64 * 3) == 64         # 192: not a multiple of 128
    assert quant.fit_group(32 * 5) == 32         # 160
    assert quant.fit_group(2048, 64) == 64       # explicit request is a ceiling
    with pytest.raises(ValueError):
        quant.fit_group(100)


def test_q8_linear_vs_w8a16_golden(w8a16_golden):
    import mlx.core as mx
    from escha_mlx import quant
    w8, scale, x, want = w8a16_golden
    lin = quant.make_linear(w8, scale.astype(np.float16), 64)
    got = np.array(lin(mx.array(x)))
    d = np.abs(got.astype(np.float32) - want.astype(np.float32)).max()
    rel = d / np.abs(want.astype(np.float32)).max()
    # quantized_matmul multiplies exact f32 dequant values; the w8a16 contract
    # rounds each weight to f16 first — sub-ulp per weight, tiny at the output.
    assert rel < 2e-3, (d, rel)


def test_had_blocks_matches_ref():
    import mlx.core as mx
    from escha_mlx import moe, ref
    rng = np.random.default_rng(11)
    x = rng.standard_normal((4, 256)).astype(np.float32)
    got = np.array(moe.had_blocks(mx.array(x)))
    want = ref.h128(x)
    assert np.abs(got - want).max() < 1e-3 * np.abs(want).max()


def _synth_experts(rng, E, ic, oc, K):
    from escha_mlx import moe
    code = rng.integers(-32768, 32768, size=(E, ic // 16, oc // 16, 16 * K), dtype=np.int16)
    rin = (rng.standard_normal((E, ic)) * 0.05).astype(np.float16)
    rout = (rng.standard_normal((E, oc)) * 0.05).astype(np.float16)
    return moe.EschaExperts(code, rin, rout), {"code": code, "rin": rin, "rout": rout}


def test_moe_block_ops_vs_ref_synthetic():
    """Full block (ops path) vs the NumPy reference on synthetic weights."""
    import mlx.core as mx
    from escha_mlx import moe, ref

    rng = np.random.default_rng(5)
    E, H, I, top_k, N = 8, 256, 128, 4, 3
    gu, gu_np = _synth_experts(rng, E, H, 2 * I, 2)
    dn, dn_np = _synth_experts(rng, E, I, H, 3)
    gate_w = (rng.standard_normal((E, H)) * 0.1).astype(np.float16)
    shg_w = (rng.standard_normal((1, H)) * 0.1).astype(np.float16)
    shared = {}
    for p, (o, k) in (("gate", (I, H)), ("up", (I, H)), ("down", (H, I))):
        shared[f"{p}_w8"] = rng.integers(-128, 128, size=(o, k), dtype=np.int8)
        shared[f"{p}_scale"] = (rng.random(o) * 0.01 + 1e-3).astype(np.float16)

    import os
    from escha_mlx.models.qwen3_5_moe import EschaSparseMoeBlock
    os.environ["ESCHA_MLX_MOE"] = "ops"
    blk = EschaSparseMoeBlock(H, E, top_k, gu, dn, gate_w, shg_w, shared)
    x = (rng.standard_normal((1, N, H)) * 0.5).astype(np.float16)
    got = np.array(blk(mx.array(x)))[0]

    weights = {
        "gate_w": gate_w, "shg_w": shg_w,
        "gu_code": gu_np["code"], "gu_rin": gu_np["rin"], "gu_rout": gu_np["rout"],
        "dn_code": dn_np["code"], "dn_rin": dn_np["rin"], "dn_rout": dn_np["rout"],
        "sh_gate_w8": shared["gate_w8"], "sh_gate_scale": shared["gate_scale"],
        "sh_up_w8": shared["up_w8"], "sh_up_scale": shared["up_scale"],
        "sh_down_w8": shared["down_w8"], "sh_down_scale": shared["down_scale"],
    }
    want = ref.moe_block(x[0], weights, top_k=top_k)
    d = np.abs(got.astype(np.float32) - want.astype(np.float32))
    scale = max(np.abs(want.astype(np.float32)).max(), 1e-6)
    assert d.max() / scale < 5e-2, (d.max(), scale)


@pytest.mark.usefixtures("moeblk_golden")
def test_moe_block_ops_vs_ckpt_golden(moeblk_golden):
    """Layer-0 block from the real checkpoint vs the committed block golden.

    Requires the checkpoint (ESCHA_MODEL). Slow on CPU (~minutes): 8 tokens
    through 8-way top-k with numpy decode.
    """
    from .conftest import DEFAULT_CKPT
    from pathlib import Path
    if not Path(DEFAULT_CKPT).exists():
        pytest.skip("checkpoint not available")

    import json
    import os
    import mlx.core as mx
    from safetensors.numpy import load_file
    from escha_mlx import moe
    from escha_mlx.models.qwen3_5_moe import EschaSparseMoeBlock

    x, want, ids_g, scores_g = moeblk_golden
    ckpt = Path(DEFAULT_CKPT)
    idx = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    pref = "model.language_model.layers.0.mlp."
    cache: dict[str, dict] = {}

    def get(name):
        shard = idx[pref + name]
        if shard not in cache:
            cache[shard] = load_file(ckpt / shard)
        return cache[shard][pref + name]

    gu = moe.EschaExperts(get("experts.gate_up_proj.escha_code"),
                          get("experts.gate_up_proj.escha_rin"),
                          get("experts.gate_up_proj.escha_rout"))
    dn = moe.EschaExperts(get("experts.down_proj.escha_code"),
                          get("experts.down_proj.escha_rin"),
                          get("experts.down_proj.escha_rout"))
    shared = {}
    for p in ("gate", "up", "down"):
        shared[f"{p}_w8"] = get(f"shared_expert.{p}_proj.weight_int8")
        shared[f"{p}_scale"] = get(f"shared_expert.{p}_proj.weight_scale")
    os.environ["ESCHA_MLX_MOE"] = "ops"
    blk = EschaSparseMoeBlock(2048, 256, 8, gu, dn,
                              get("gate.weight"), get("shared_expert_gate.weight"),
                              shared)
    got = np.array(blk(mx.array(x[None])))[0]

    # routing agreement first (set-wise; order differs by convention)
    logits = np.array(blk.gate(mx.array(x)).astype(mx.float32))
    my_ids = np.sort(np.argsort(-logits, axis=-1)[:, :8], axis=-1)
    assert np.array_equal(my_ids, np.sort(ids_g, axis=-1)), "router disagrees with golden"

    d = np.abs(got.astype(np.float32) - want.astype(np.float32))
    denom = np.abs(want.astype(np.float32)).max()
    print(f"moeblk golden: max abs {d.max():.4f}  (out max {denom:.2f})")
    assert d.max() / denom < 3e-2, (d.max(), denom)
