"""Reproduce Qwen3.8 native-MTP one-shot and continuous-batch benchmarks.

The public result uses two separate measurements:

* ``one-shot``: one loaded model, both paths warmed, then AR/MTP/MTP/AR.
  Timing includes prefill and 256 generated tokens.
* ``continuous``: decode-only after prefill at each batch shape.  Every ABBA
  arm runs in a fresh subprocess so MLX allocator or cache state from one path
  cannot affect the other path.

Examples::

    python bench/mtp.py --model ~/models/Qwen3.8-27B-Escha-W2 \
        --mode one-shot --out /tmp/mtp-one-shot.json
    python bench/mtp.py --model ~/models/Qwen3.8-27B-Escha-W2 \
        --mode continuous --batches 1,2,4,8 --out /tmp/mtp-batch.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from escha_mlx.benchmark_metadata import (  # noqa: E402
    annotate_report,
    benchmark_metadata,
    model_display_name,
)


def _digest_tokens(tokens: list[int]) -> str:
    payload = b"".join(int(token).to_bytes(4, "little") for token in tokens)
    return hashlib.sha256(payload).hexdigest()[:16]


def _one_shot(args: argparse.Namespace) -> dict:
    from mlx_lm.generate import generate_step

    from escha_mlx.loader import load
    from escha_mlx.mtp import mtp_generate_step

    mx.set_memory_limit(19_000_000_000)
    model, tokenizer = load(args.model, load_mtp=True)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt = mx.array(tokenizer.encode(text), dtype=mx.uint32)

    def run_ar(max_tokens: int) -> tuple[float, list[int], float | None]:
        started = time.perf_counter()
        tokens = [
            int(token)
            for token, _ in generate_step(prompt, model, max_tokens=max_tokens)
        ]
        mx.synchronize()
        return time.perf_counter() - started, tokens, None

    def run_mtp(max_tokens: int) -> tuple[float, list[int], float | None]:
        started = time.perf_counter()
        output = list(mtp_generate_step(prompt, model, max_tokens=max_tokens))
        mx.synchronize()
        tokens = [int(token) for token, _, _ in output]
        target_rounds = sum(not accepted for _, _, accepted in output)
        mean_emit = len(tokens) / target_rounds if target_rounds else None
        return time.perf_counter() - started, tokens, mean_emit

    run_ar(args.warmup_tokens)
    run_mtp(args.warmup_tokens)
    mx.clear_cache()

    results: dict[str, list[dict]] = {"ar": [], "mtp": []}
    order = (("ar", run_ar), ("mtp", run_mtp),
             ("mtp", run_mtp), ("ar", run_ar))
    for name, runner in order:
        elapsed, tokens, mean_emit = runner(args.output_tokens)
        row = {
            "elapsed_s": elapsed,
            "output_tokens": len(tokens),
            "reached_max_tokens": len(tokens) == args.output_tokens,
            "ms_per_token": 1000 * elapsed / len(tokens),
            "tokens_per_s": len(tokens) / elapsed,
            "mean_emit": mean_emit,
            "digest": _digest_tokens(tokens),
        }
        results[name].append(row)
        mx.clear_cache()

    if not all(r["reached_max_tokens"] for rows in results.values() for r in rows):
        raise RuntimeError("one-shot generation stopped before --output-tokens")

    medians = {
        name: {
            "ms_per_token": statistics.median(r["ms_per_token"] for r in rows),
            "tokens_per_s": statistics.median(r["tokens_per_s"] for r in rows),
            "mean_emit": rows[0]["mean_emit"],
        }
        for name, rows in results.items()
    }
    return {
        "model": model_display_name(args.model),
        "workload": {
            "prompt": args.prompt,
            "prompt_tokens": int(prompt.size),
            "output_tokens": args.output_tokens,
            "warmup_tokens": args.warmup_tokens,
            "sampling": "greedy",
            "order": "ABBA",
            "timing": "end-to-end including prefill",
        },
        "runs": results,
        "medians": medians,
        "median_speedup": (
            medians["ar"]["ms_per_token"] / medians["mtp"]["ms_per_token"]
        ),
    }


def _batch_arm(args: argparse.Namespace) -> dict:
    from mlx_lm.generate import BatchGenerator

    from escha_mlx.loader import load
    from escha_mlx.mtp_batch import MTPBatchGenerator

    model, tokenizer = load(args.model, load_mtp=True)
    seed = tokenizer.encode(
        "Analyze the performance of speculative decoding on Apple silicon. " * 12
    )[:args.isl]
    if len(seed) != args.isl:
        raise ValueError(f"could construct only {len(seed)} of {args.isl} prompt tokens")
    prompts = [seed[:-1] + [seed[-1] - (i % 7)] for i in range(args.batch)]
    generator_type = MTPBatchGenerator if args.path == "mtp" else BatchGenerator

    def run(max_tokens: int) -> dict:
        generator = generator_type(
            model,
            max_tokens=max_tokens,
            stop_tokens=None,
            prefill_batch_size=args.batch,
            completion_batch_size=args.batch,
            prefill_step_size=args.prefill_step_size,
        )
        uids = generator.insert(prompts)
        live = set(uids)
        outputs = {uid: [] for uid in uids}
        finish_reasons: dict[int, str] = {}
        try:
            while len(generator._generation_batch) == 0:
                generator.next()
            start_steps = generator._steps_counter
            mx.reset_peak_memory()
            started = time.perf_counter()
            while live:
                _, responses = generator.next()
                for response in responses:
                    outputs[response.uid].append(int(response.token))
                    if response.finish_reason is not None:
                        live.discard(response.uid)
                        finish_reasons[response.uid] = response.finish_reason
            mx.synchronize()
            elapsed = time.perf_counter() - started
            steps = generator._steps_counter - start_steps
        finally:
            generator.close()

        output_lists = [outputs[uid] for uid in uids]
        lengths = [len(tokens) for tokens in output_lists]
        output_total = sum(lengths)
        return {
            "path": args.path,
            "batch": args.batch,
            "seconds": elapsed,
            "output_tokens_per_request": lengths,
            "finish_reasons": [finish_reasons.get(uid) for uid in uids],
            "reached_max_tokens": all(n == max_tokens for n in lengths),
            "output_tps": output_total / elapsed,
            "steps": steps,
            "emission_per_request_round": output_total / (steps * args.batch),
            "peak_gb": mx.get_peak_memory() / 1e9,
            "digest": hashlib.sha256(repr(output_lists).encode()).hexdigest(),
        }

    run(args.warmup_tokens)
    mx.clear_cache()
    result = run(args.output_tokens)
    if not result["reached_max_tokens"]:
        raise RuntimeError("continuous-batch generation stopped before --output-tokens")
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return result


def _continuous(args: argparse.Namespace) -> dict:
    rows = []
    script = str(Path(__file__).resolve())
    for batch in (int(value) for value in args.batches.split(",")):
        for path in ("ar", "mtp", "mtp", "ar"):
            command = [
                sys.executable, script,
                "--mode", "_batch-arm",
                "--model", args.model,
                "--path", path,
                "--batch", str(batch),
                "--isl", str(args.isl),
                "--output-tokens", str(args.output_tokens),
                "--warmup-tokens", str(args.warmup_tokens),
                "--prefill-step-size", str(args.prefill_step_size),
            ]
            print(f"running batch={batch} path={path}", file=sys.stderr, flush=True)
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.arm_timeout,
            )
            if result.returncode != 0:
                detail = result.stderr.strip().splitlines()[-1:] or ["no stderr"]
                raise RuntimeError(
                    f"batch={batch} path={path} failed: {detail[0]}"
                )
            rows.append(json.loads(result.stdout))

    summary = []
    for batch in (int(value) for value in args.batches.split(",")):
        ar = [r for r in rows if r["batch"] == batch and r["path"] == "ar"]
        mtp = [r for r in rows if r["batch"] == batch and r["path"] == "mtp"]
        ar_mean = statistics.mean(r["output_tps"] for r in ar)
        mtp_mean = statistics.mean(r["output_tps"] for r in mtp)
        summary.append({
            "batch": batch,
            "ar_tps_mean": ar_mean,
            "mtp_tps_mean": mtp_mean,
            "speedup": mtp_mean / ar_mean,
            "mtp_emission_mean": statistics.mean(
                r["emission_per_request_round"] for r in mtp
            ),
            "ar_peak_gb_max": max(r["peak_gb"] for r in ar),
            "mtp_peak_gb_max": max(r["peak_gb"] for r in mtp),
            "ar_samples": [r["output_tps"] for r in ar],
            "mtp_samples": [r["output_tps"] for r in mtp],
        })
    return {
        "model": model_display_name(args.model),
        "workload": {
            "isl": args.isl,
            "osl": args.output_tokens,
            "warmup_tokens": args.warmup_tokens,
            "order": "ABBA, fresh process per arm",
            "stop_tokens": None,
        },
        "rows": rows,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode", choices=("one-shot", "continuous", "_batch-arm"),
        default="one-shot",
    )
    parser.add_argument("--out")
    parser.add_argument("--prompt", default="Explain speculative decoding in one paragraph.")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--path", choices=("ar", "mtp"))
    parser.add_argument("--batch", type=int)
    parser.add_argument("--isl", type=int, default=128)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--warmup-tokens", type=int)
    parser.add_argument("--prefill-step-size", type=int, default=256)
    parser.add_argument("--arm-timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.output_tokens is None:
        args.output_tokens = 96 if args.mode in ("continuous", "_batch-arm") else 256
    if args.warmup_tokens is None:
        args.warmup_tokens = 16 if args.mode in ("continuous", "_batch-arm") else 8

    if args.mode == "_batch-arm":
        if args.path is None or args.batch is None:
            parser.error("_batch-arm requires --path and --batch")
        print(json.dumps(_batch_arm(args)))
        return

    report = _one_shot(args) if args.mode == "one-shot" else _continuous(args)
    report = annotate_report(report, benchmark_metadata(args.model))
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(payload)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
