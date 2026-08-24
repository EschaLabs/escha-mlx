"""Baseline + batching characterization for escha-mlx on Apple Silicon.

Loads the model ONCE (~25 s) and runs, in order:

  A. correctness battery      — greedy anchors (Paris / 391 / thinking)
  B. bs1 decode + prefill     — tok/s vs prompt length, peak memory
  C. batched decode           — replication invariance (B=1..16) + aggregate tok/s
  D. cache accounting         — measured bytes/token for KV and the GDN state

Phase C is the load-bearing one: continuous batching is only worth building if
B>1 decode is (a) numerically identical to B=1 for a replicated prompt and
(b) sublinear in wall-time.  Run with --phases to select, e.g. `--phases AB`.

    python bench/baseline.py --model ~/models/escha-w2 --phases ABCD
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
    model_display_name,
)


def _gb(x: float) -> float:
    """Bytes -> decimal GB (1e9), matching every other harness in bench/.

    This divided by 1024**3 (GiB) until 2026-08-09 while printing "GB", which
    is where the historical 11.41-"GB"-resident figure came from: it is the
    same residency as head_to_head.py's 12.25 GB. B/C-phase peak_gb values in
    JSONs committed before that date are GiB.
    """
    return x / 1e9


def _sync(*arrays) -> None:
    mx.eval(*arrays)
    mx.synchronize()


PREFILL_CHUNK = 256
WARMUP_STEPS = 8


def chunked_prefill(model, ids: mx.array, cache, chunk: int = PREFILL_CHUNK) -> mx.array:
    """Prefill in fixed chunks, returning only the LAST position's logits.

    Two reasons this is not optional on a 24 GB box:
      * the full-sequence logits tensor is [B, S, 248320] -- 2.03 GB at S=4096,
        and prefill only ever needs row -1;
      * per-layer MoE transients scale with the chunk, so an unchunked 4k
        prefill peaks at 20.74 GB, past the 19.07 GB wired limit.
    """
    s = ids.shape[1]
    logits = None
    for i in range(0, s, chunk):
        del logits
        logits = model(ids[:, i:i + chunk], cache=cache)
        mx.eval(logits)
    return logits


# --------------------------------------------------------------------------
# A. correctness battery
# --------------------------------------------------------------------------

def phase_a(model, tokenizer) -> dict:
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    out = {}
    cases = [
        ("paris_raw", "The capital of France is", True, False, 16),
        ("mult_chat", "What is 17*23? Answer directly.", False, False, 32),
        ("haiku_think", "Write a haiku about Tokyo.", False, True, 200),
    ]
    for name, prompt, raw, thinking, ntok in cases:
        if raw:
            p = prompt
        else:
            p = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
                enable_thinking=thinking)
        text, last = "", None
        for r in stream_generate(model, tokenizer, p, max_tokens=ntok, sampler=sampler):
            text += r.text
            last = r
        out[name] = {
            "text": text,
            "gen_tps": round(last.generation_tps, 2) if last else None,
            "prompt_tps": round(last.prompt_tps, 2) if last else None,
            "peak_gb": round(last.peak_memory, 2) if last else None,
        }
        print(f"  [{name}] {last.generation_tokens} tok @ {last.generation_tps:.1f} tok/s")
        print(f"      {text[:220]!r}")
    return out


# --------------------------------------------------------------------------
# B. bs1 decode + prefill scaling
# --------------------------------------------------------------------------

def phase_b(model, tokenizer, isls: list[int], decode_steps: int) -> dict:
    from mlx_lm.models.cache import make_prompt_cache

    # A deterministic pseudo-prompt of an EXACT token length: sample ids from
    # the middle of the vocab (avoids specials) so ISL is exact, not approximate.
    vocab = model.language_model.args.vocab_size

    def make_ids(n: int) -> mx.array:
        rng = mx.random.key(1234)
        ids = mx.random.randint(1000, min(vocab, 100000), shape=(1, n), key=rng)
        return ids.astype(mx.int32)

    results = {}
    for isl in isls:
        mx.clear_cache()
        mx.reset_peak_memory()
        ids = make_ids(isl)
        cache = make_prompt_cache(model)

        _sync(ids)
        t0 = time.perf_counter()
        logits = chunked_prefill(model, ids, cache)
        _sync(logits)
        t_prefill = time.perf_counter() - t0

        # decode loop.  clear_cache() is MANDATORY here: the prefill transient
        # otherwise sits in MLX's buffer cache and pushes the working set past
        # the wired limit, costing 12-24x decode throughput (measured).
        tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None].astype(mx.int32)
        del logits
        mx.clear_cache()
        _sync(tok)
        t0 = time.perf_counter()
        for _ in range(decode_steps):
            logits = model(tok, cache=cache)
            tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None].astype(mx.int32)
        _sync(tok)
        t_decode = time.perf_counter() - t0

        peak = _gb(mx.get_peak_memory())
        results[isl] = {
            "prefill_s": round(t_prefill, 3),
            "prefill_tps": round(isl / t_prefill, 1),
            "decode_tps": round(decode_steps / t_decode, 2),
            "ms_per_token": round(1000 * t_decode / decode_steps, 2),
            "peak_gb": round(peak, 2),
        }
        print(f"  ISL={isl:>6}  prefill {t_prefill:7.2f}s ({isl/t_prefill:8.1f} tok/s)"
              f"   decode {decode_steps/t_decode:6.2f} tok/s"
              f"   ({1000*t_decode/decode_steps:6.2f} ms/tok)  peak {peak:.2f} GB")
        del cache
        gc.collect()
    return results


# --------------------------------------------------------------------------
# C. batched decode: replication invariance + aggregate throughput
# --------------------------------------------------------------------------

def phase_c(model, tokenizer, batches: list[int], isl: int, decode_steps: int) -> dict:
    """Replicate ONE prompt B times.

    Correctness: every row must produce the identical token sequence, and it
    must match the B=1 sequence.  This is the R6 bring-up gate (batch>=3
    hybrid-cache corruption is a known ecosystem failure class) and it is the
    only thing standing between us and continuous batching.
    """
    from mlx_lm.models.cache import make_prompt_cache

    vocab = model.language_model.args.vocab_size
    rng = mx.random.key(7)
    base_ids = mx.random.randint(1000, min(vocab, 100000), shape=(1, isl), key=rng).astype(mx.int32)

    results, ref_seq = {}, None
    for b in batches:
        mx.clear_cache()
        mx.reset_peak_memory()
        ids = mx.repeat(base_ids, b, axis=0)          # [B, isl] identical rows
        try:
            cache = make_prompt_cache(model)
            _sync(ids)
            t0 = time.perf_counter()
            logits = chunked_prefill(model, ids, cache)
            _sync(logits)
            t_prefill = time.perf_counter() - t0

            tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None].astype(mx.int32)
            seq = [tok]
            del logits
            mx.clear_cache()
            _sync(tok)

            # WARMUP: every distinct batch shape triggers fresh Metal kernel
            # specialization; timing the first steps of a new shape measures the
            # compiler, not the GPU.  Warm up, then clear the transient, then time.
            for _ in range(WARMUP_STEPS):
                logits = model(tok, cache=cache)
                tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None].astype(mx.int32)
                seq.append(tok)
            _sync(tok)
            mx.clear_cache()

            t0 = time.perf_counter()
            for _ in range(decode_steps):
                logits = model(tok, cache=cache)
                tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None].astype(mx.int32)
                seq.append(tok)
            _sync(tok)
            t_decode = time.perf_counter() - t0

            toks = mx.concatenate(seq, axis=1)         # [B, steps+1]
            mx.eval(toks)
            rows = toks.tolist()
            all_same = all(r == rows[0] for r in rows)
            matches_b1 = (ref_seq is None) or (rows[0] == ref_seq)
            if ref_seq is None:
                ref_seq = rows[0]

            agg = b * decode_steps / t_decode
            peak = _gb(mx.get_peak_memory())
            results[b] = {
                "prefill_s": round(t_prefill, 3),
                "prefill_tps": round(b * isl / t_prefill, 1),
                "aggregate_decode_tps": round(agg, 2),
                "per_seq_decode_tps": round(agg / b, 2),
                "ms_per_step": round(1000 * t_decode / decode_steps, 2),
                "rows_identical": all_same,
                "matches_b1": matches_b1,
                "peak_gb": round(peak, 2),
            }
            flag = "OK " if (all_same and matches_b1) else "FAIL"
            print(f"  B={b:>3} [{flag}] aggregate {agg:7.2f} tok/s"
                  f"   per-seq {agg/b:6.2f}   step {1000*t_decode/decode_steps:7.2f} ms"
                  f"   prefill {b*isl/t_prefill:8.1f} tok/s   peak {peak:.2f} GB")
            if not all_same:
                print(f"       !! rows diverge: row0[:8]={rows[0][:8]} row1[:8]={rows[1][:8]}")
            if not matches_b1:
                print(f"       !! differs from B=1: b1={ref_seq[:8]} got={rows[0][:8]}")
            del cache, toks
        except Exception as e:  # OOM or shape failure — record and continue
            results[b] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  B={b:>3} [ERR ] {type(e).__name__}: {str(e)[:160]}")
        gc.collect()
    return results


# --------------------------------------------------------------------------
# D. cache accounting
# --------------------------------------------------------------------------

def phase_d(model, tokenizer, isl: int = 512) -> dict:
    from mlx_lm.models.cache import make_prompt_cache

    vocab = model.language_model.args.vocab_size
    ids = mx.random.randint(1000, min(vocab, 100000), shape=(1, isl),
                            key=mx.random.key(3)).astype(mx.int32)
    cache = make_prompt_cache(model)
    logits = model(ids, cache=cache)
    _sync(logits)

    kinds: dict[str, dict] = {}
    for i, c in enumerate(cache):
        name = type(c).__name__
        nb = c.nbytes() if callable(getattr(c, "nbytes", None)) else getattr(c, "nbytes", 0)
        d = kinds.setdefault(name, {"count": 0, "bytes": 0, "trimmable": None})
        d["count"] += 1
        d["bytes"] += int(nb)
        if d["trimmable"] is None:
            it = getattr(c, "is_trimmable", None)
            d["trimmable"] = bool(it()) if callable(it) else bool(it)
    total = sum(v["bytes"] for v in kinds.values())
    out = {"isl": isl, "total_mb": round(total / 1e6, 2), "kinds": {}}
    print(f"  cache after ISL={isl}: total {total/1e6:.1f} MB")
    for name, v in kinds.items():
        # attribute per-token growth to KV kinds only; recurrent state is constant
        per_tok = v["bytes"] / isl
        out["kinds"][name] = {
            "layers": v["count"],
            "mb": round(v["bytes"] / 1e6, 2),
            "bytes_per_token": round(per_tok, 1),
            "trimmable": v["trimmable"],
        }
        print(f"    {name:<24} x{v['count']:<3} {v['bytes']/1e6:8.1f} MB"
              f"  ({per_tok:8.1f} B/tok if KV)  trimmable={v['trimmable']}")
    return out


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--phases", default="ABCD")
    ap.add_argument("--isls", default="128,512,2048")
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--batch-isl", type=int, default=128)
    global PREFILL_CHUNK
    ap.add_argument("--prefill-chunk", type=int, default=PREFILL_CHUNK)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    PREFILL_CHUNK = args.prefill_chunk
    metadata = benchmark_metadata(args.model)

    from escha_mlx.loader import load

    print(f"=== escha-mlx baseline — mlx {mx.__version__} ===")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"model loaded in {time.time()-t0:.1f}s, resident {_gb(mx.get_active_memory()):.2f} GB\n")

    # The repo id, not args.model: these reports are committed, and an absolute
    # path publishes the operator's home directory while adding nothing to
    # model_hf_revision. See benchmark_metadata.model_display_name.
    report: dict = {"mlx": mx.__version__, "model": model_display_name(args.model)}
    if "A" in args.phases:
        print("--- A. correctness battery ---")
        report["A_correctness"] = phase_a(model, tokenizer)
        print()
    if "B" in args.phases:
        print("--- B. bs1 decode + prefill scaling ---")
        isls = [int(x) for x in args.isls.split(",")]
        report["B_bs1"] = phase_b(model, tokenizer, isls, args.decode_steps)
        print()
    if "C" in args.phases:
        print(f"--- C. batched decode (ISL={args.batch_isl}) ---")
        batches = [int(x) for x in args.batches.split(",")]
        report["C_batch"] = phase_c(model, tokenizer, batches, args.batch_isl, args.decode_steps)
        print()
    if "D" in args.phases:
        print("--- D. cache accounting ---")
        report["D_cache"] = phase_d(model, tokenizer)
        print()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(annotate_report(report, metadata), f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
