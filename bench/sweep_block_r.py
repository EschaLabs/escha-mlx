"""Sweep rows-per-group (R) at DECODE row counts, plus the MLX wired limit.

Why this exists (bring-up doc §9 lever 5).  `_blocked_R` was tuned on PREFILL
row counts, where every expert is hit many times and sharing one decoded stream
across R rows is a clear 4.8x win.  Decode is a different distribution: B
sequences x top_k=8 slots draw m = 8B rows from E=256 experts, so the expected
number of DISTINCT experts is

    E * (1 - (1 - 1/E)^m)

    B= 1  m=  8 -> 7.9 distinct (1.01 rows/expert)  -- nothing to amortize
    B= 8  m= 64 -> 56.2         (1.06)
    B=16  m=128 -> 100.9        (1.27)  -> 21% of stream bytes are redundant
    B=32  m=256 -> 161.9        (1.58)  -> 37%
    B=64  m=512 -> 221.1        (2.32)  -> 57%

So collisions become worth harvesting somewhere in B=16..32 -- exactly the
concurrency band that matters for aggregate throughput -- while the current
policy uses R=1 below m=256 and jumps straight to R=4 at m>=256.  R=2/3 are
never used, and R=4 at m=256 pads ~1.6 rows/expert up to 4.

The trade is explicit: grouping cuts trellis STREAM BYTES by
(distinct/m) but inflates ROW WORK by (R*groups/m) because partial groups are
padded to R.  The kernel is ~53% of roofline at B=16, i.e. neither purely
bandwidth- nor purely ALU-bound, so which side wins is a measurement, not a
derivation.  Hence this sweep.

Also sweeps `mx.set_wired_limit`, which defaults to **0** -- MLX wires nothing,
so at high B the resident set is evictable.  Free to set, capped by
`max_recommended_working_set_size` unless the iogpu.wired_limit_mb sysctl is
raised.

    python bench/sweep_block_r.py --model <dir> [--batches 8,16,24]
                                            [--rs 1,2,3,4] [--wired 18]

Correctness gate: the GEMM output itself (`_gemv`), R=1 vs R>1 on one fixed
input inside a single process — bit-identity is the direct kernel contract and
holds exactly, independent of everything downstream.  Historical note: this
gate was chosen when greedy decode was still run-to-run NONDETERMINISTIC —
`_expert_path` used to end in `y.at[row_token].add(contrib)`, an atomic f32
scatter-add with top_k=8 duplicate indices per token, so summation order varied
per run (measured: R=1 vs R=1, same seed, same prompt -> different tokens).
docs/BRINGUP_AND_PERF.md §10.5 replaced that with a fixed-order segmented sum,
so decode is now bit-reproducible; the GEMM gate is retained as the sharper,
cheaper check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from escha_mlx.benchmark_metadata import annotate_report, benchmark_metadata

PREFILL = 16          # short prompt: we are measuring decode, not prefill
WARMUP = 8            # per-shape Metal specialization must not land in timing
STEPS = 24


def set_R(model, r: int | None) -> None:
    for layer in model.language_model.model.layers:
        blk = layer.mlp
        if hasattr(blk, "_block_env"):
            blk._block_env = r


def expected_distinct(m: int, E: int = 256) -> float:
    return E * (1.0 - (1.0 - 1.0 / E) ** m)


def check_gemm_equiv(model, B: int, rs: list[int]) -> dict[int, bool]:
    """Bit-identity of the trellis GEMM across R, on a DECODE-shaped routing draw.

    The unit tests (tests/test_blocked.py) cover random and extreme-skew
    distributions; this exercises the real thing -- m = 8B rows drawn from
    E=256 with ~1.1-2.3 rows/expert -- against the real packed weights, in one
    process so there is nothing nondeterministic in the comparison.
    """
    import numpy as np

    blk = model.language_model.model.layers[0].mlp
    m = 8 * B
    rng = np.random.default_rng(7)
    row_expert = mx.array(rng.integers(0, blk.num_experts, size=m).astype(np.int32))
    xh = mx.array(rng.standard_normal((m, blk._gu.IC)).astype(np.float16))
    mx.eval(row_expert, xh)

    ref = None
    out = {}
    # `None` means "let the policy choose"; resolve it so the identity check
    # covers whatever R the shipped policy actually picks at this m.
    saved_env = blk._block_env
    blk._block_env = None                    # so _blocked_R reports the POLICY
    concrete = sorted({(blk._blocked_R(m) if r is None else r) for r in rs})
    blk._block_env = saved_env
    for r in concrete:
        saved = blk._block_env
        blk._block_env = r
        groups = None
        if r > 1:
            from escha_mlx.moe import build_groups, n_groups_bound
            ng = n_groups_bound(m, blk.num_experts, r)
            rows_idx, group_expert = build_groups(row_expert, blk.num_experts, r, ng)[:2]
            # _gemv's groups tuple carries the optional sorted-x arrays too
            groups = (rows_idx, group_expert, r, None, None, None)
        mid = blk._gemv(xh, row_expert, blk._gu, groups)
        mx.eval(mid)
        a = np.array(mid)
        blk._block_env = saved
        if ref is None:
            ref, out[r] = a, True
        else:
            out[r] = bool(np.array_equal(a, ref))
    return out


def run(model, B: int, r: int | None, steps: int = STEPS) -> dict:
    """Time `steps` decode steps at batch B with rows-per-group r."""
    set_R(model, r)
    from mlx_lm.models.cache import make_prompt_cache

    mx.random.seed(1234)
    cache = make_prompt_cache(model)
    ids = mx.random.randint(1000, 60000, shape=(B, PREFILL)).astype(mx.int32)
    logits = model(ids, cache=cache)
    mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1)[:, None].astype(mx.int32)
    mx.eval(tok)
    # MLX's buffer cache holds the prefill transients; without this the decode
    # loop thrashes the wired limit and reads 20x slow (bring-up doc §5).
    mx.clear_cache()

    def step(t):
        lg = model(t, cache=cache)
        return mx.argmax(lg[:, -1, :], axis=-1)[:, None].astype(mx.int32)

    emitted = []
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
        emitted.append(tok)
    mx.synchronize()
    dt = (time.perf_counter() - t0) / steps
    peak = mx.get_peak_memory() / 1e9

    out = {
        "batch": B,
        "R": r if r is not None else "policy",
        "ms_per_step": round(dt * 1000, 2),
        "tok_s": round(B / dt, 2),
        "peak_gb": round(peak, 2),
    }
    del cache
    mx.clear_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batches", default="8,16,24")
    ap.add_argument("--rs", default="1,2,3,4")
    ap.add_argument("--wired", type=float, default=None,
                    help="GB to pass to mx.set_wired_limit (default: leave at 0)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--repeats", type=int, default=1,
                    help="measurements per config; the median is reported")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    metadata = benchmark_metadata(args.model)

    if args.wired:
        info = mx.device_info()
        cap = info["max_recommended_working_set_size"] / 1e9
        if args.wired > cap:
            print(f"!! requested wired {args.wired:.1f} GB > cap {cap:.2f} GB; "
                  f"raise it with: sudo sysctl iogpu.wired_limit_mb="
                  f"{int(args.wired * 1000)}")
            return
        prev = mx.set_wired_limit(int(args.wired * 1e9))
        print(f"wired limit {prev/1e9:.2f} -> {args.wired:.2f} GB (cap {cap:.2f})")

    from escha_mlx.loader import load

    print(f"loading {args.model} ...")
    t0 = time.time()
    model, _ = load(args.model)
    print(f"loaded in {time.time()-t0:.1f}s")

    batches = [int(b) for b in args.batches.split(",")]
    # "p" = leave _blocked_R's own policy in charge (verifies the shipped default)
    rs = [None if r.strip() in ("p", "policy") else int(r) for r in args.rs.split(",")]
    rows = []

    for B in batches:
        m = 8 * B
        d = expected_distinct(m)
        print(f"\nB={B}  (m={m} rows, E[distinct experts]={d:.1f}, "
              f"{m/d:.2f} rows/expert, stream-byte floor {d/m:.2f}x)")
        equiv = check_gemm_equiv(model, B, rs)
        bad_r = [r for r, ok in equiv.items() if not ok]
        print(f"  GEMM bit-identity across R={sorted(equiv)}: "
              f"{'ALL EQUAL' if not bad_r else f'**DIVERGED at R={bad_r}**'}")
        set_R(model, None)
        policy_R = model.language_model.model.layers[0].mlp._blocked_R(8 * B)
        base = None
        for r in rs:
            reps = []
            for _ in range(args.repeats):
                try:
                    reps.append(run(model, B, r, args.steps))
                except Exception as e:                  # OOM at high B is data
                    print(f"  R={r}: FAILED {type(e).__name__}: {str(e)[:90]}")
                    break
            if not reps:
                continue
            tps = sorted(x["tok_s"] for x in reps)
            res = dict(reps[0])
            res["tok_s"] = tps[len(tps) // 2]           # median over repeats
            res["tok_s_all"] = tps
            res["ms_per_step"] = round(1000.0 * B / res["tok_s"], 2)
            res["rows_per_expert"] = round(m / d, 2)
            eff_R = policy_R if r is None else r
            res["effective_R"] = eff_R
            res["gemm_bit_identical"] = equiv.get(eff_R, None)
            rows.append(res)
            if base is None:
                base = res["tok_s"]
            delta = 100.0 * (res["tok_s"] - base) / base
            spread = 100.0 * (tps[-1] - tps[0]) / tps[0] if len(tps) > 1 else 0.0
            label = f"policy(R={eff_R})" if r is None else f"R={r}"
            print(f"  {label:>13s}: {res['ms_per_step']:7.2f} ms/step  "
                  f"{res['tok_s']:7.2f} tok/s  peak {res['peak_gb']:5.2f} GB  "
                  f"{delta:+6.1f}%   (n={len(tps)}, spread {spread:.1f}%)")

    bad = [r for r in rows if r["gemm_bit_identical"] is False]
    print(f"\n{len(rows)} configs, {len(bad)} with non-bit-identical GEMM"
          f"{' — INVESTIGATE' if bad else ' (all bit-identical)'}")
    if args.out:
        Path(args.out).write_text(json.dumps(annotate_report(rows, metadata), indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
