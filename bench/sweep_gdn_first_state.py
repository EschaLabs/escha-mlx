"""Peak memory of the FIRST forward on a fresh cache.

Every decode harness here -- bench/sweep_kernel_variants.run,
bench/sweep_output_had.py::_decode_once, bench/baseline.py Phase C -- calls
mx.reset_peak_memory() AFTER the prompt and the warmup steps, because that is
what makes steady-state decode peaks comparable.  That ordering makes them
structurally blind to the allocation this probe exists to measure: the
full-sized f32 initial recurrent state that upstream's gated_delta_update
materialises on the first recurrence and then casts to the cache dtype.  By the
time those harnesses arm the counter, the transient is already freed.

Here the counter is armed BEFORE a single forward on a fresh cache, which is the
only window in which that state exists.  Both arms run against the same loaded
weights in one process:

  main -- upstream gated_delta_update: mx.zeros(f32) state, then cast to fp16
  PR   -- escha_mlx.gdn_cache's register-zero Metal kernel, writing fp16 direct

Logits and recurrent states are compared for bit equality across the arms, so a
memory saving that changed numerics would fail rather than read as a win.

    python bench/sweep_gdn_first_state.py --model <dir> [--batches 1,8,16,32,64]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from escha_mlx import gdn_cache
from escha_mlx.benchmark_metadata import annotate_report, benchmark_metadata


def first_forward(model, batch: int, tokens: int):
    """One forward on a fresh cache, with peak tracking armed beforehand."""
    from mlx_lm.models.cache import make_prompt_cache

    ids = mx.random.randint(
        1000, 60000, shape=(batch, tokens), key=mx.random.key(1234)
    ).astype(mx.int32)
    mx.eval(ids)

    mx.clear_cache()
    mx.synchronize()
    mx.reset_peak_memory()

    cache = make_prompt_cache(model)
    logits = model(ids, cache=cache)
    # Only the linear (GDN) layers carry a recurrent state, and KVCache does not
    # support indexing at all -- select on the cache type, not on the slot.
    states = [c[1] for c in cache
              if isinstance(c, gdn_cache.GDNStateCache) and c[1] is not None]
    mx.eval(logits, states)
    mx.synchronize()

    peak = mx.get_peak_memory() / 1e9
    logits_np = np.array(logits.astype(mx.float32))
    state_dtype = str(states[0].dtype) if states else "none"
    state_np = np.array(states[0].astype(mx.float32)) if states else None

    del cache, logits, states
    mx.clear_cache()
    return peak, logits_np, state_np, state_dtype


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batches", default="1,8,16,32,64")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--memory-limit-gb", type=float, default=19.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mx.set_memory_limit(int(args.memory_limit_gb * 1e9))
    metadata = benchmark_metadata(args.model)

    from escha_mlx.loader import load

    print(f"loading {args.model} ...")
    model, _ = load(args.model)
    dtype = gdn_cache.state_dtype()
    if dtype == mx.float32:
        raise SystemExit(
            "ESCHA_MLX_GDN_STATE=fp32 disables the zero-state kernel entirely; "
            "there is no A/B to run at f32.")

    rows: list[dict] = []
    for batch in [int(v) for v in args.batches.split(",") if v]:
        record: dict[str, object] = {"batch": batch, "tokens": args.tokens}
        captured = {}
        for label, install in (
            ("main", lambda: gdn_cache._restore_upstream_patch()),
            ("PR", lambda: gdn_cache._install_zero_state_patch(dtype)),
        ):
            install()
            peak, logits, state, state_dtype = first_forward(
                model, batch, args.tokens)
            captured[label] = (logits, state)
            record[f"{label}_peak_gb"] = round(peak, 3)
            record[f"{label}_state_dtype"] = state_dtype

        saved = record["main_peak_gb"] - record["PR_peak_gb"]
        record["saved_gb"] = round(saved, 3)
        record["saved_pct"] = round(100 * saved / record["main_peak_gb"], 2)
        record["logits_bit_identical"] = bool(
            np.array_equal(captured["main"][0], captured["PR"][0]))
        record["state_bit_identical"] = bool(
            np.array_equal(captured["main"][1], captured["PR"][1]))
        rows.append(record)
        print(
            f"  B={batch:<4d} main {record['main_peak_gb']:6.3f} GB   "
            f"PR {record['PR_peak_gb']:6.3f} GB   "
            f"saved {saved:5.3f} GB ({record['saved_pct']:5.2f}%)   "
            f"logits_identical={record['logits_bit_identical']} "
            f"state_identical={record['state_bit_identical']}"
        )
        if args.out:
            Path(args.out).write_text(json.dumps(annotate_report({
                "description": "peak memory of the first forward on a fresh "
                               "cache: the only window in which the f32 "
                               "initial GDN state exists",
                "settings": {
                    "prefill_tokens": args.tokens,
                    "memory_limit_gb": args.memory_limit_gb,
                    "gdn_state_dtype": str(dtype),
                },
                "results": rows,
            }, metadata), indent=2))

    gdn_cache._install_zero_state_patch(dtype)


if __name__ == "__main__":
    main()
