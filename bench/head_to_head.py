"""Paired head-to-head: escha W2 vs a stock mlx-lm build, same box, same harness.

Every published Apple-Silicon comparison we could find is cross-report --
different chips, prompt lengths, thermal states, chassis (Air vs Pro) and engine
versions -- so the only defensible comparison is running both builds through
IDENTICAL measurement code on one machine in one session.

That is what this does.  The two models differ only in how weights are stored:

    A  escha W2       trellis-coded 2-bit experts + int8 dense (escha_mlx.loader)
    B  stock 4-bit    mlx-lm's own affine 4-bit          (mlx_lm.load)

Same base model (Qwen3.6-35B-A3B), same mlx/mlx-lm, same GPU, back to back.

Fairness rules, all of which cost us something:
  * identical prefill chunking for both (mlx-lm's default chunk differs from
    ours; using ours for both removes that as a variable),
  * identical warmup per shape -- Metal specialises kernels per shape and the
    first call otherwise lands in the timing (doc §10.4),
  * mx.clear_cache() at the same points (without it decode after a long prefill
    reads 20x slow -- doc §5),
  * A/B/A ordering: A is measured first AND last, so thermal drift over the run
    is visible instead of being attributed to B,
  * identical wired limit, set before either model loads.

Reports prefill tok/s, decode tok/s, and peak memory -- the three axes where the
published numbers disagree with each other.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from escha_mlx.benchmark_metadata import (
    annotate_report,
    benchmark_metadata,
    model_hf_revision,
)

WARMUP_DECODE = 8
PREFILL_CHUNK = 256


def load_any(path: str):
    """escha checkpoint -> escha_mlx.loader; anything else -> mlx_lm.load."""
    from escha_mlx.loader import is_escha_checkpoint, load as escha_load

    if is_escha_checkpoint(path):
        return escha_load(path), "escha-W2"
    from mlx_lm import load as mlx_load
    return mlx_load(path), "stock-mlx"


def chunked_prefill(model, ids, cache, chunk):
    logits = None
    for i in range(0, ids.shape[1], chunk):
        del logits
        logits = model(ids[:, i:i + chunk], cache=cache)
        mx.eval(logits)
    return logits


def measure(model, isl: int, decode_steps: int, chunk: int, batch: int = 1):
    from mlx_lm.models.cache import make_prompt_cache

    mx.random.seed(4242)
    ids = mx.random.randint(1000, 60000, shape=(batch, isl)).astype(mx.int32)
    mx.eval(ids)

    # ---- prefill ---------------------------------------------------------
    mx.clear_cache()
    mx.reset_peak_memory()
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = chunked_prefill(model, ids, cache, chunk)
    mx.synchronize()
    t_prefill = time.perf_counter() - t0
    peak_prefill = mx.get_peak_memory() / 1e9

    # ---- decode ----------------------------------------------------------
    tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None].astype(mx.int32)
    mx.eval(tok)
    mx.clear_cache()          # mandatory: prefill transients otherwise thrash

    def step(t):
        lg = model(t, cache=cache)
        return mx.argmax(lg[:, -1, :], axis=-1)[:, None].astype(mx.int32)

    for _ in range(WARMUP_DECODE):
        tok = step(tok)
        mx.eval(tok)
    mx.synchronize()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(decode_steps):
        tok = step(tok)
        mx.eval(tok)
    mx.synchronize()
    t_decode = (time.perf_counter() - t0) / decode_steps
    peak_decode = mx.get_peak_memory() / 1e9

    del cache, logits
    mx.clear_cache()
    return {
        "isl": isl, "batch": batch,
        "prefill_s": round(t_prefill, 3),
        "prefill_tok_s": round(isl * batch / t_prefill, 1),
        "decode_tok_s": round(batch / t_decode, 2),
        "ms_per_token": round(t_decode * 1000, 2),
        "peak_prefill_gb": round(peak_prefill, 2),
        "peak_decode_gb": round(peak_decode, 2),
    }


def run_one(path: str, args) -> dict:
    (model, tokenizer), kind = load_any(path)
    resident = mx.get_active_memory() / 1e9
    print(f"    loaded [{kind}] resident {resident:.2f} GB")
    out = {"path": path, "kind": kind, "resident_gb": round(resident, 2),
           "points": []}
    for isl in [int(x) for x in args.isls.split(",")]:
        r = measure(model, isl, args.decode_steps, args.chunk)
        out["points"].append(r)
        print(f"    ISL {isl:5d}  prefill {r['prefill_tok_s']:8.1f} tok/s  "
              f"decode {r['decode_tok_s']:6.2f} tok/s  "
              f"peak {r['peak_prefill_gb']:5.2f}/{r['peak_decode_gb']:5.2f} GB")
    for B in [int(x) for x in args.batches.split(",")] if args.batches else []:
        try:
            r = measure(model, 128, args.decode_steps, args.chunk, batch=B)
            out["points"].append(r)
            print(f"    B={B:3d}      aggregate decode {r['decode_tok_s']:7.2f} "
                  f"tok/s  peak {r['peak_decode_gb']:5.2f} GB")
        except Exception as e:
            print(f"    B={B:3d}      FAILED {type(e).__name__}: {str(e)[:80]}")
            out["points"].append({"batch": B, "error": f"{type(e).__name__}"})
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="escha checkpoint")
    ap.add_argument("--b", required=True, help="stock mlx-lm model dir/repo")
    ap.add_argument("--isls", default="512,2048")
    ap.add_argument("--batches", default="")
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=PREFILL_CHUNK)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from escha_mlx.loader import apply_wired_limit
    apply_wired_limit()          # identical envelope for both arms

    # set_wired_limit both sets AND returns the previous value, so querying it
    # means immediately putting it back.
    cur = mx.set_wired_limit(0)
    mx.set_wired_limit(cur)
    cap = mx.device_info().get("max_recommended_working_set_size", 0) / 1e9
    print(f"chunk={args.chunk}, decode_steps={args.decode_steps}, "
          f"wired={cur/1e9:.2f} GB, cap={cap:.2f} GB")

    results = []
    metadata = benchmark_metadata(args.a)
    metadata["model_hf_revision"] = {
        args.a: metadata["model_hf_revision"],
        args.b: model_hf_revision(args.b),
    }
    # A/B/A so drift over the run is visible rather than charged to B
    for label, path in (("A (escha W2)", args.a),
                        ("B (stock 4-bit)", args.b),
                        ("A again [drift control]", args.a)):
        print(f"\n=== {label}: {path}")
        try:
            r = run_one(path, args)
        except Exception as e:
            print(f"    LOAD/RUN FAILED: {type(e).__name__}: {str(e)[:200]}")
            r = {"path": path, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        r["label"] = label
        results.append(r)

    if args.out:
        Path(args.out).write_text(json.dumps(annotate_report(results, metadata), indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
