"""Paired in-model A/B/A for the fused expert output Hadamard transform.

The shipped input transform stays fused in every arm.  Only ``_output_rows``
changes:

* A: native MLX Hadamard + scale gather + cast
* B: one custom Metal kernel for the same operations
* A: native again as the drift control

The harness covers prompt prefill and step-synchronized aggregate decode, keeps
the model loaded once, warms every shape/arm, records raw repeat samples and
requires identical output hashes across every arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.baseline import chunked_prefill
from escha_mlx import moe, ref
from escha_mlx.benchmark_metadata import annotate_report, benchmark_metadata


def _git_diff_sha256() -> str | None:
    result = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=Path(__file__).parent.parent,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _hardware() -> dict[str, object]:
    ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return {
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "ram_gb": round(ram / 1e9, 2),
        "mlx": mx.__version__,
        "device": mx.device_info(),
    }


def _hash_array(x: mx.array) -> str:
    mx.eval(x)
    return hashlib.sha256(np.array(x).tobytes()).hexdigest()


def _native_output(self, mid, row_expert, ex):
    return (moe.had_blocks(mid) * ref.RS * ex.rout[row_expert]).astype(mx.float16)


def _set_output_method(layers, method) -> None:
    for layer in layers:
        layer.mlp._output_rows = types.MethodType(method, layer.mlp)


def _prefill_once(model, ids: mx.array, chunk: int) -> tuple[float, str, float]:
    from mlx_lm.models.cache import make_prompt_cache

    mx.clear_cache()
    mx.reset_peak_memory()
    cache = make_prompt_cache(model)
    mx.synchronize()
    t0 = time.perf_counter()
    logits = chunked_prefill(model, ids, cache, chunk)
    mx.eval(logits)
    mx.synchronize()
    elapsed = time.perf_counter() - t0
    peak = mx.get_peak_memory() / 1e9
    digest = _hash_array(logits)
    del cache, logits
    mx.clear_cache()
    return elapsed, digest, peak


def _decode_once(
    model, batch: int, prefill: int, warmup: int, steps: int
) -> tuple[float, str, float]:
    from mlx_lm.models.cache import make_prompt_cache

    mx.clear_cache()
    cache = make_prompt_cache(model)
    ids = mx.random.randint(
        1000, 60000, shape=(batch, prefill), key=mx.random.key(1234)
    ).astype(mx.int32)
    logits = model(ids, cache=cache)
    mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None].astype(mx.int32)
    mx.eval(tok)
    del logits
    mx.clear_cache()

    sequence = [tok]
    for _ in range(warmup):
        logits = model(tok, cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None].astype(mx.int32)
        mx.eval(tok)
        sequence.append(tok)
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()

    t0 = time.perf_counter()
    for _ in range(steps):
        logits = model(tok, cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None].astype(mx.int32)
        mx.eval(tok)
        sequence.append(tok)
    mx.synchronize()
    elapsed = time.perf_counter() - t0
    peak = mx.get_peak_memory() / 1e9
    tokens = mx.concatenate(sequence, axis=1)
    digest = _hash_array(tokens)
    del cache, tokens
    mx.clear_cache()
    return batch * steps / elapsed, digest, peak


def _summarize(samples: list[float]) -> dict[str, object]:
    return {
        "median": round(statistics.median(samples), 3),
        "min": round(min(samples), 3),
        "max": round(max(samples), 3),
        "spread_pct": round(100 * (max(samples) - min(samples)) / min(samples), 2),
        "samples": [round(x, 3) for x in samples],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--isls", default="128,512,2048")
    ap.add_argument(
        "--batches",
        default="1,8,16,32",
        help="comma-separated decode batches (B=128 should run in a fresh process)",
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prefill-chunk", type=int, default=256)
    ap.add_argument("--prefill-warmup", type=int, default=3)
    ap.add_argument("--decode-prefill", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=24)
    ap.add_argument("--memory-limit-gb", type=float, default=19.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mx.set_memory_limit(int(args.memory_limit_gb * 1e9))
    metadata = benchmark_metadata(args.model)
    metadata.update(
        {
            "working_tree_diff_sha256": _git_diff_sha256(),
            "hardware": _hardware(),
        }
    )

    from escha_mlx.loader import load

    print(f"loading {args.model} ...")
    model, _ = load(args.model)
    wired = mx.set_wired_limit(0)
    mx.set_wired_limit(wired)
    metadata["hardware"]["wired_limit_gb"] = round(wired / 1e9, 2)
    layers = model.language_model.model.layers
    fused_output = type(layers[0].mlp)._output_rows
    arms = (
        ("native A1", _native_output),
        ("fused B", fused_output),
        ("native A2 (drift control)", _native_output),
    )

    report: dict[str, object] = {
        "settings": {
            "repeats": args.repeats,
            "prefill_chunk": args.prefill_chunk,
            "prefill_warmup": args.prefill_warmup,
            "decode_prefill": args.decode_prefill,
            "warmup": args.warmup,
            "decode_steps": args.decode_steps,
            "memory_limit_gb": args.memory_limit_gb,
        },
        "prefill": [],
        "decode": [],
    }

    def checkpoint() -> None:
        if args.out:
            path = Path(args.out)
            path.write_text(json.dumps(annotate_report(report, metadata), indent=2))

    try:
        vocab = model.language_model.args.vocab_size
        for isl in [int(v) for v in args.isls.split(",")]:
            ids = mx.random.randint(
                1000, min(vocab, 100000), shape=(1, isl), key=mx.random.key(7)
            ).astype(mx.int32)
            mx.eval(ids)
            print(f"\nprefill ISL={isl}")
            hashes: set[str] = set()
            for label, method in arms:
                _set_output_method(layers, method)
                for _ in range(args.prefill_warmup):
                    _prefill_once(model, ids, args.prefill_chunk)
                raw = [
                    _prefill_once(model, ids, args.prefill_chunk)
                    for _ in range(args.repeats)
                ]
                tps = [isl / elapsed for elapsed, _, _ in raw]
                arm_hashes = {digest for _, digest, _ in raw}
                hashes.update(arm_hashes)
                summary = _summarize(tps)
                row = {
                    "isl": isl,
                    "arm": label,
                    "tok_s": summary,
                    "peak_gb": round(max(peak for _, _, peak in raw), 3),
                    "hashes": sorted(arm_hashes),
                }
                report["prefill"].append(row)
                print(
                    f"  {label:27s} {summary['median']:8.1f} tok/s  "
                    f"spread {summary['spread_pct']:5.2f}%  peak {row['peak_gb']:.2f} GB"
                )
            if len(hashes) != 1:
                raise RuntimeError(f"prefill ISL={isl} output hashes differ: {hashes}")
            checkpoint()

        for batch in [int(v) for v in args.batches.split(",")]:
            print(f"\ndecode B={batch}")
            hashes: set[str] = set()
            for label, method in arms:
                _set_output_method(layers, method)
                raw = [
                    _decode_once(
                        model,
                        batch,
                        args.decode_prefill,
                        args.warmup,
                        args.decode_steps,
                    )
                    for _ in range(args.repeats)
                ]
                tps = [value for value, _, _ in raw]
                arm_hashes = {digest for _, digest, _ in raw}
                hashes.update(arm_hashes)
                summary = _summarize(tps)
                row = {
                    "batch": batch,
                    "arm": label,
                    "aggregate_tok_s": summary,
                    "peak_gb": round(max(peak for _, _, peak in raw), 3),
                    "hashes": sorted(arm_hashes),
                }
                report["decode"].append(row)
                print(
                    f"  {label:27s} {summary['median']:8.2f} tok/s  "
                    f"spread {summary['spread_pct']:5.2f}%  peak {row['peak_gb']:.2f} GB"
                )
            if len(hashes) != 1:
                raise RuntimeError(f"decode B={batch} token hashes differ: {hashes}")
            checkpoint()
    finally:
        _set_output_method(layers, fused_output)

    result = annotate_report(report, metadata)
    if args.out:
        path = Path(args.out)
        path.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
