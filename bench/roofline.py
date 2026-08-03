"""Roofline / where-do-the-milliseconds-go analysis for escha-mlx.

Decode on Apple Silicon is memory-bandwidth bound, so the ceiling is

    tok/s = achieved_DRAM_bandwidth / bytes_read_per_token

This script measures BOTH terms instead of assuming either:

  1. peak achievable read bandwidth (pure streaming reduction)      -> roofline
  2. MLX affine-Q8 `quantized_matmul` at the model's real dense shapes
  3. our trellis `moe_gemv` at the model's real expert shapes, vs rows
  4. the exact per-token byte ledger, computed from the LOADED model
  5. a measured decode step, so component times can be reconciled with it

The output tells you which of the three streams (Q8 dense / trellis experts /
everything else) is costing what, and what fraction of peak each achieves.

    python -m escha_mlx.bench.roofline --model ~/models/escha-w2
"""
from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import mlx.nn as nn

PEAK_GBPS = {  # advertised DRAM bandwidth, GB/s
    "Apple M4": 120.0, "Apple M4 Pro": 273.0, "Apple M4 Max": 546.0,
    "Apple M5": 153.0, "Apple M5 Pro": 307.0, "Apple M5 Max": 614.0,
    "Apple M1 Max": 400.0, "Apple M3 Max": 400.0, "Apple M2 Max": 400.0,
}


def timeit(fn, iters: int = 20, warmup: int = 5) -> float:
    """Time one call of `fn`.

    MLX is LAZY: `for _ in range(n): r = fn()` followed by a single eval builds
    n graphs and computes ONE of them -- that mistake reports ~17x the physical
    DRAM bandwidth.  Every iteration must be forced to completion.
    """
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def dispatch_floor() -> float:
    """Per-call eval+dispatch overhead, so tiny-op numbers can be read honestly."""
    a = mx.zeros((16,), dtype=mx.float16)
    mx.eval(a)
    return timeit(lambda: a + 1, iters=200, warmup=50)


# ---------------------------------------------------------------- 1. roofline

def measure_peak_bw() -> dict:
    """Pure streaming read.  sum() over a large fp16 buffer is read-bound."""
    out = {}
    best = 0.0
    for mb in (256, 512, 1024, 2048):
        n = mb * 1024 * 1024 // 2          # fp16 elements
        a = mx.random.normal((n,), dtype=mx.float16)
        mx.eval(a)
        t = timeit(lambda: mx.sum(a, stream=mx.gpu), iters=20)
        gbps = (n * 2) / t / 1e9
        out[f"read_{mb}MB"] = round(gbps, 1)
        best = max(best, gbps)
        del a
        mx.clear_cache()
    out["peak_read_gbps"] = round(best, 1)
    return out


# ------------------------------------------------------- 2. Q8 dense (qmv/qmm)

def measure_q8(shapes: list[tuple[str, int, int]], batches: list[int],
               group: int = 64, bits: int = 8) -> dict:
    """MLX affine-Q8 matmul at the model's real dense shapes.

    Bytes = weights (1 B/elt at 8 bit) + scales&biases (2 x fp16 per group).
    """
    res = {}
    for name, ic, oc in shapes:
        w = mx.random.normal((oc, ic), dtype=mx.float16)
        wq, scales, biases = mx.quantize(w, group_size=group, bits=bits)
        mx.eval(wq, scales, biases)
        wbytes = oc * ic * bits / 8 + 2 * 2 * (oc * ic // group)
        for b in batches:
            x = mx.random.normal((b, ic), dtype=mx.float16)
            mx.eval(x)
            t = timeit(lambda: mx.quantized_matmul(
                x, wq, scales, biases, transpose=True, group_size=group, bits=bits))
            res[f"{name}_b{b}"] = {
                "ms": round(t * 1000, 3),
                "gbps": round(wbytes / t / 1e9, 1),
                "mb": round(wbytes / 1e6, 1),
            }
            del x
        del w, wq, scales, biases
        mx.clear_cache()
    return res


# ------------------------------------------------------- 3. trellis moe_gemv

def measure_trellis(rows_list: list[int]) -> dict:
    from escha_mlx import msl

    res = {}
    # real shapes: gate_up K=2 [2048 -> 1024], down K=3 [512 -> 2048]
    for name, K, ic, oc in (("gate_up_K2", 2, 2048, 1024), ("down_K3", 3, 512, 2048)):
        E = 256
        tk, tn, wpt = ic // 16, oc // 16, 8 * K
        code = mx.random.randint(0, 2**31 - 1, shape=(E, tk, tn, wpt)).astype(mx.uint32)
        mx.eval(code)
        expert_bytes = tk * tn * wpt * 4          # bytes of ONE expert's stream
        for rows in rows_list:
            xh = mx.random.normal((rows, ic), dtype=mx.float16)
            re = (mx.arange(rows) % E).astype(mx.int32)
            mx.eval(xh, re)
            t = timeit(lambda: msl.moe_gemv(xh, code, re, K, ic, oc), iters=10, warmup=3)
            streamed = rows * expert_bytes        # kernel reads 1 stream PER ROW
            res[f"{name}_rows{rows}"] = {
                "ms": round(t * 1000, 3),
                "gbps": round(streamed / t / 1e9, 1),
                "mb_streamed": round(streamed / 1e6, 1),
                "mb_unique": round(min(rows, E) * expert_bytes / 1e6, 1),
            }
            del xh, re
        del code
        mx.clear_cache()
    return res


# ---------------------------------------------------- 4. exact byte ledger

def byte_ledger(model, top_k: int) -> dict:
    """Walk the LOADED model and total the bytes read for ONE decode token."""
    from escha_mlx import moe as moe_mod

    lm = model.language_model
    args = lm.args
    n_layers = args.num_hidden_layers

    from mlx.utils import tree_flatten

    def mod_bytes(mod) -> int:
        """Every leaf array of a module: weight + (for Q8) scales and biases.

        nn.Module.parameters() returns a NESTED dict, so it must be flattened --
        iterating it directly silently yields 0 for every composite module.
        """
        if mod is None:
            return 0
        return sum(v.nbytes for _k, v in tree_flatten(mod.parameters())
                   if isinstance(v, mx.array))

    layers = lm.model.layers
    dense_attn = dense_gdn = shared = gate = norms = 0
    expert_per_token = 0
    n_gdn = n_attn = 0
    for layer in layers:
        blk = layer.mlp
        if isinstance(blk, moe_mod.EschaSparseMoeBlock):
            gu, dn = blk._gu, blk._dn
            # ONE expert's stream, both projections, times top_k active experts
            per_expert = (gu.code.nbytes + dn.code.nbytes) / gu.E
            expert_per_token += per_expert * top_k
            expert_per_token += (gu.rin.nbytes + gu.rout.nbytes
                                 + dn.rin.nbytes + dn.rout.nbytes) / gu.E * top_k
            shared += mod_bytes(blk.sh_gate) + mod_bytes(blk.sh_up) + mod_bytes(blk.sh_down)
            gate += mod_bytes(blk.gate) + mod_bytes(blk.shared_expert_gate)
        if getattr(layer, "linear_attn", None) is not None:
            dense_gdn += mod_bytes(layer.linear_attn)
            n_gdn += 1
        elif getattr(layer, "self_attn", None) is not None:
            dense_attn += mod_bytes(layer.self_attn)
            n_attn += 1
        norms += mod_bytes(layer.input_layernorm) + mod_bytes(layer.post_attention_layernorm)
    head = mod_bytes(getattr(lm, "lm_head", None))

    total = expert_per_token + dense_attn + dense_gdn + shared + gate + head + norms
    return {
        "n_layers": n_layers, "top_k": top_k,
        "n_gdn_layers": n_gdn, "n_attn_layers": n_attn,
        "experts_MB": round(expert_per_token / 1e6, 1),
        "attn_dense_MB": round(dense_attn / 1e6, 1),
        "gdn_dense_MB": round(dense_gdn / 1e6, 1),
        "shared_expert_MB": round(shared / 1e6, 1),
        "router_gate_MB": round(gate / 1e6, 1),
        "norms_MB": round(norms / 1e6, 1),
        "lm_head_MB": round(head / 1e6, 1),
        "TOTAL_GB_per_token": round(total / 1e9, 3),
        "_dense_GB": (total - expert_per_token) / 1e9,
        "_experts_GB": expert_per_token / 1e9,
    }


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="if given, compute the byte ledger + step time")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    info = mx.device_info()
    chip = info.get("device_name", "?")
    peak = PEAK_GBPS.get(chip)
    print(f"=== roofline — {chip} | mlx {mx.__version__} ===")
    print(f"advertised peak DRAM bandwidth: {peak} GB/s"
          if peak else "advertised peak: UNKNOWN for this chip")
    print(f"wired limit {info['max_recommended_working_set_size']/1e9:.2f} GB\n")

    report = {"chip": chip, "advertised_peak_gbps": peak}

    floor = dispatch_floor()
    print(f"per-call dispatch floor: {floor*1e6:.1f} us "
          f"(ops faster than this are dispatch-bound, not bandwidth-bound)\n")
    report["dispatch_floor_us"] = round(floor * 1e6, 1)

    print("--- 1. peak achievable read bandwidth ---")
    bw = measure_peak_bw()
    report["bandwidth"] = bw
    for k, v in bw.items():
        if k != "peak_read_gbps":
            print(f"  {k:<16} {v:8.1f} GB/s" + (f"  ({100*v/peak:.0f}% of peak)" if peak else ""))
    achieved = bw["peak_read_gbps"]
    print(f"  ROOFLINE         {achieved:8.1f} GB/s"
          + (f"  ({100*achieved/peak:.0f}% of advertised)" if peak else ""))

    print("\n--- 2. MLX affine-Q8 quantized_matmul (the dense stream) ---")
    shapes = [
        ("lm_head_248320x2048", 2048, 248320),
        ("gdn_in_proj_2048x12288", 2048, 12288),
        ("gdn_out_proj_4096x2048", 4096, 2048),
        ("attn_qkv_2048x4096", 2048, 4096),
        ("shared_gate_2048x512", 2048, 512),
    ]
    q8 = measure_q8(shapes, [1, 8, 16, 32])
    report["q8"] = q8
    for k, v in q8.items():
        frac = f"  ({100*v['gbps']/achieved:.0f}% of roofline)"
        print(f"  {k:<32} {v['ms']:8.3f} ms  {v['gbps']:7.1f} GB/s{frac}")

    print("\n--- 3. trellis moe_gemv (the expert stream) ---")
    tr = measure_trellis([8, 16, 32, 64, 128, 256])
    report["trellis"] = tr
    for k, v in tr.items():
        frac = f"  ({100*v['gbps']/achieved:.0f}% of roofline)"
        print(f"  {k:<24} {v['ms']:8.3f} ms  {v['gbps']:7.1f} GB/s"
              f"  streamed {v['mb_streamed']:7.1f} MB (unique {v['mb_unique']:6.1f}){frac}")

    if args.model:
        print("\n--- 4. exact byte ledger from the loaded model ---")
        from escha_mlx.loader import load
        model, _tok = load(args.model)
        top_k = model.language_model.args.num_experts_per_tok
        led = byte_ledger(model, top_k)
        report["ledger"] = led
        for k, v in led.items():
            print(f"  {k:<24} {v}")
        bpt = led["TOTAL_GB_per_token"]
        print(f"\n  CEILING at roofline {achieved:.0f} GB/s : {achieved/bpt:6.1f} tok/s (bs1)")
        if peak:
            print(f"  CEILING at advertised {peak:.0f} GB/s: {peak/bpt:6.1f} tok/s (bs1)")

        print("\n--- 5. measured decode step ---")
        from mlx_lm.models.cache import make_prompt_cache
        V = model.language_model.args.vocab_size
        for b in (1, 8, 16):
            cache = make_prompt_cache(model)
            ids = mx.random.randint(1000, 60000, shape=(b, 8)).astype(mx.int32)
            lg = model(ids, cache=cache)
            mx.eval(lg)
            tok = mx.argmax(lg[:, -1, :], axis=-1)[:, None].astype(mx.int32)
            mx.eval(tok)
            mx.clear_cache()

            def step():
                nonlocal tok
                lg2 = model(tok, cache=cache)
                tok = mx.argmax(lg2[:, -1, :], axis=-1)[:, None].astype(mx.int32)
                return tok
            t = timeit(step, iters=16, warmup=6)
            # bytes: dense amortizes over the batch, experts scale with rows
            # (capped: at B*top_k >= n_experts every expert is read at most once)
            dense = led["_dense_GB"]
            exp_b = led["_experts_GB"] * b
            gb = dense + exp_b                       # already GB
            eff = gb / t                             # GB/s
            print(f"  B={b:<3} {t*1000:8.2f} ms/step  {b/t:7.2f} tok/s aggregate"
                  f"  | modeled {gb:5.2f} GB/step -> {eff:6.1f} GB/s"
                  f" ({100*eff/achieved:.0f}% of roofline)")
            report.setdefault("step", {})[f"B{b}"] = {
                "ms": round(t * 1000, 2), "tps": round(b / t, 2),
                "modeled_gb": round(dense + exp_b, 3), "eff_gbps": round(eff, 1),
            }
            del cache
            mx.clear_cache()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
