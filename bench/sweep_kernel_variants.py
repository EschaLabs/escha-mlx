"""A/B the three kernel/glue levers in-model, with a drift control.

Levers (bring-up doc §9, items 1/6/8):
  splitk   -- partition the trellis kt chain across S threadgroups (msl)
  shuffle  -- broadcast the code tile intra-simdgroup instead of 2 loads/lane
Both are togglable inside one process -- splitk/shuffle are read from the
environment per call -- so every variant is measured against the same loaded
model, avoiding the ~22 s reload and the across-process drift it brings.

(A third lever, mx.compile on the elementwise glue, was measured and REMOVED:
throughput wash, and it shifted full-model logits by 5.6% relative. See the
bring-up doc §11.3.)

Ordering: `base` is measured FIRST and LAST. The gap between those two readings
bounds thermal/position drift for the whole block, so a variant delta smaller
than that gap is not a result. (One earlier session showed 7.7% between two
computationally identical configs -- see doc §10.4.)

    python bench/sweep_kernel_variants.py --model <dir> \
        [--batches 1,8,16,32] [--repeats 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

PREFILL = 16
WARMUP = 8
STEPS = 24

# name -> (env overrides, unused)
VARIANTS: dict[str, tuple[dict[str, str], str | None]] = {
    "base (shipped defaults)": ({}, None),
    "splitk auto":             ({"ESCHA_MLX_SPLITK": "auto"}, None),
    "shuffle fetch":           ({"ESCHA_MLX_FETCH": "shuffle"}, None),
    "splitk auto + shuffle":   ({"ESCHA_MLX_SPLITK": "auto",
                                 "ESCHA_MLX_FETCH": "shuffle"}, None),
}

# Focused table for the split-K / fetch decision.  Only meaningful at B<=4:
# split_k_for returns 1 for gate_up once m>=64, and from B=16 the MoE takes the
# row-blocked GEMM path so moe_gemv is not called at all.  Measuring "no splitk"
# at B=8/16 compares identical code, which is what made the first pass read as
# uniform noise.
# Pre-sorted-x staging (doc §15.3): borrowed from mlx-lm's SwitchGLU.
# NB: set the flag EXPLICITLY on both arms. Relying on "unset == the variant I
# mean" silently turns the A/B into A/A the moment the default flips.
VARIANTS_SORTX: dict[str, tuple[dict[str, str], str | None]] = {
    "rows_idx gather (default)": ({"ESCHA_MLX_SORTX": "0"}, None),
    "sorted-x":                  ({"ESCHA_MLX_SORTX": "1"}, None),
}

VARIANTS_PREFETCH: dict[str, tuple[dict[str, str], str | None]] = {
    "per-kt fetch (default)": ({"ESCHA_MLX_PREFETCH": "0"}, None),
    "code prefetch":          ({"ESCHA_MLX_PREFETCH": "1"}, None),
}

VARIANTS_SPLITK: dict[str, tuple[dict[str, str], str | None]] = {
    "S=auto,   shuffle": ({"ESCHA_MLX_SPLITK": "auto",
                          "ESCHA_MLX_FETCH": "shuffle"}, None),
    "S=1,      shuffle": ({"ESCHA_MLX_SPLITK": "1"}, None),
    "S=2,      shuffle": ({"ESCHA_MLX_SPLITK": "2"}, None),
    "S=4,      shuffle": ({"ESCHA_MLX_SPLITK": "4"}, None),
    "S=policy, load":    ({"ESCHA_MLX_FETCH": "load"}, None),
    "S=1,      load":    ({"ESCHA_MLX_SPLITK": "1", "ESCHA_MLX_FETCH": "load"}, None),
    "S=4,      load":    ({"ESCHA_MLX_SPLITK": "4", "ESCHA_MLX_FETCH": "load"}, None),
}


def apply_env(over: dict[str, str],
              keys=("ESCHA_MLX_SPLITK", "ESCHA_MLX_FETCH", "ESCHA_MLX_SORTX",
                    "ESCHA_MLX_PREFETCH")) -> None:
    for k in keys:
        os.environ.pop(k, None)
    os.environ.update(over)


def run(model, B: int, steps: int) -> tuple[float, float]:
    from mlx_lm.models.cache import make_prompt_cache

    mx.random.seed(1234)
    cache = make_prompt_cache(model)
    ids = mx.random.randint(1000, 60000, shape=(B, PREFILL)).astype(mx.int32)
    lg = model(ids, cache=cache)
    mx.eval(lg)
    tok = mx.argmax(lg[:, -1, :], axis=-1)[:, None].astype(mx.int32)
    mx.eval(tok)
    mx.clear_cache()

    def step(t):
        lg = model(t, cache=cache)
        return mx.argmax(lg[:, -1, :], axis=-1)[:, None].astype(mx.int32)

    for _ in range(WARMUP):
        tok = step(tok)
        mx.eval(tok)
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()

    t0 = time.perf_counter()
    for _ in range(steps):
        tok = step(tok)
        mx.eval(tok)
    mx.synchronize()
    dt = (time.perf_counter() - t0) / steps
    peak = mx.get_peak_memory() / 1e9
    del cache
    mx.clear_cache()
    return B / dt, peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batches", default="1,8,16,32")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--table", choices=("levers", "splitk", "sortx", "prefetch"), default="levers")
    args = ap.parse_args()

    table = {"levers": VARIANTS, "splitk": VARIANTS_SPLITK,
             "sortx": VARIANTS_SORTX,
             "prefetch": VARIANTS_PREFETCH}[args.table]

    from escha_mlx.loader import load

    print(f"loading {args.model} ...")
    model, _ = load(args.model)

    names = list(table) + [next(iter(table))]            # base again at the end
    rows = []
    for B in [int(b) for b in args.batches.split(",")]:
        print(f"\nB={B}")
        base = None
        for i, name in enumerate(names):
            over, mode = table[name]
            apply_env(over)
            res = [run(model, B, args.steps) for _ in range(args.repeats)]
            tps = sorted(r[0] for r in res)
            peak = max(r[1] for r in res)
            med = tps[len(tps) // 2]
            label = name if i < len(names) - 1 else name + " [drift control]"
            if base is None:
                base = med
                delta = ""
            else:
                delta = f"{100*(med-base)/base:+6.1f}%"
            spread = 100 * (tps[-1] - tps[0]) / tps[0]
            print(f"  {label:38s} {med:7.2f} tok/s {delta:>8s}  "
                  f"peak {peak:5.2f} GB  (n={len(tps)}, spread {spread:.1f}%)")
            rows.append({"batch": B, "variant": label, "tok_s": round(med, 2),
                         "peak_gb": round(peak, 2), "spread_pct": round(spread, 1),
                         "all": [round(x, 2) for x in tps]})
        apply_env({})

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")
    print("\nA variant delta smaller than the base-vs-drift-control gap is NOT "
          "a result.")


if __name__ == "__main__":
    main()
