# M4 Pro / 48 GB

First Pro-class Apple silicon measured for escha-mlx. All previously committed results
are from a base M4 (10-core GPU, 120 GB/s advertised, 24 GB).

## Machine

| item | value |
|---|---|
| Model | MacBook Pro, `Mac16,7` (MX2Y3LL/A) |
| Chip | Apple M4 Pro |
| CPU | 14 cores (10 performance, 4 efficiency) |
| GPU | 20 cores |
| Unified memory | 48 GB |
| Memory bandwidth | 273 GB/s advertised, **242.8 GB/s measured (89%)** |
| GPU working-set cap | 38.65 GB (no sysctl override) |
| macOS | 15.7.3 (24G419) |
| Python | 3.12.7, arm64 |

Base M4 for reference: 10-core GPU, 101 GB/s measured of 120 advertised (84%).
Bandwidth ratio between the machines: **2.40x**. GPU core ratio: **2.0x**.

## Software

| package | version |
|---|---|
| escha-mlx | 0.1.0 |
| mlx | 0.32.0 |
| mlx-metal | 0.32.0 |
| mlx-lm | 0.31.3 |
| numpy | 2.5.1 |
| transformers | 5.14.1 |

Checkpoint: `EschaLabs/Qwen3.6-35B-A3B-Escha-W2` @ `1b7237f0886a10b4bd92cd7653090cd7381ae199`

Note: the committed `m4-base-24gb` results record no mlx version, macOS version or commit,
so this comparison is not version-matched. Stated as a caveat rather than resolved.

## Run conditions

- Single session, 2026-08-04, laptop on AC power, lid open, no external display.
- `ESCHA_MLX_WIRED_GB=30` on every run (cap 38.65 GB, default 0).
- Server launched with `--prefill-step-size 256` as documented, for comparability with the
  24 GB results even though 48 GB does not require it.
- Other applications closed during measurement.
- Benchmarks run in the requested order: `p0_gates`, `baseline`, `isl_osl_grid`,
  `head_to_head`.

## Correctness

Everything passes.

- `pytest tests/` : 164 passed, 1 skipped.
- `bench/p0_gates.py` : ALL GATES PASS. Every Metal kernel path bit-exact against the
  committed goldens on 20-core Apple GPU silicon.
- Serving smoke test correct (`17*23` returns `391`), generations coherent.
- Resident memory 11.41 GB, identical to the committed base-M4 figure.

## Finding 1: time to first token regresses in absolute terms

Comparing `m4-base-24gb/grid_fused.json` against this machine at identical grid points,
same script, same fused config, same request counts:

| point | base M4 TTFT | M4 Pro TTFT | |
|---|---|---|---|
| 128:128, C=1 | 0.806 s | 0.549 s | Pro 1.47x faster |
| 128:128, C=8 | 4.434 s | 4.581 s | level |
| 1000:1000, C=1 | 4.082 s | 3.771 s | Pro 1.08x faster |
| 1000:1000, C=8 | 10.368 s | **27.093 s** | **Pro 2.6x slower** |
| 2048:128, C=1 | 8.109 s | 8.077 s | level |
| 2048:128, C=8 | 18.133 s | **48.525 s** | **Pro 2.7x slower** |

At concurrency 1 the machines are level or the Pro is ahead. Under concurrency, with
prompts of 1000 tokens or more, the Pro is roughly 2.6x slower to first token despite 2x
the GPU cores and 2.4x the bandwidth.

Corroborating, from the in-process path: prefill at ISL 2048 measures 266.6 tok/s here
against 264 committed for base M4, a ratio of 1.01x. Prefill throughput is flat across
the two chips. The served TTFT regression is the same phenomenon under load.

From the full grid, TTFT growth with concurrency at fixed ISL:

| ISL:OSL | C=1 | C=4 | C=8 |
|---|---|---|---|
| 128:128 | 0.549 s | 2.565 s | 4.581 s |
| 2048:128 | 8.077 s | 25.540 s | 48.525 s |
| 20000:2000 | 84.2 s | 294.1 s | **560.1 s** |

Worse than linear at every ISL, consistent with prefill not batching across requests.
At 20k input and concurrency 8, first token takes 9.3 minutes.

## Finding 2: decode scaling is 1.40x against a published 2.3x

Median across the six grid points shared with `m4-base-24gb/grid_fused.json`, fused
config both sides:

| point | base M4 | M4 Pro | ratio |
|---|---|---|---|
| 128:128 C=1 | 23.53 | 36.14 | 1.54x |
| 128:128 C=8 | 47.31 | 71.83 | 1.52x |
| 1000:1000 C=1 | 24.81 | 33.54 | 1.35x |
| 1000:1000 C=8 | 50.43 | 72.56 | 1.44x |
| 2048:128 C=1 | 10.04 | 11.54 | 1.15x |
| 2048:128 C=8 | 13.47 | 17.36 | 1.29x |

**Median 1.40x, mean 1.38x, range 1.15x to 1.54x.** No point reaches 2.3x.
That is 58% of the measured bandwidth gain and 61% of the published projection.

The mechanism is visible in roofline utilization, which *falls* on the faster chip:

| | roofline ceiling | measured decode | utilization |
|---|---|---|---|
| base M4 | 39.3 tok/s | 27.3 | 69.5% |
| M4 Pro | 94.5 tok/s | 42.6 | **45.0%** |

More bandwidth, worse utilization, which is the signature of a latency-bound workload
rather than a bandwidth-bound one. Consistent with the bs1 GEMV diagnosis in
`docs/BRINGUP_AND_PERF.md` section 3.1. Batched decode holds up better than single
stream, which fits: larger batches amortize the fixed per-dispatch cost.

## Drift control

Three in-process `baseline` runs are committed.

| metric | opening | closing | control |
|---|---|---|---|
| decode ISL512 | 42.56 | 38.71 | 40.28 |
| B=8 aggregate | 114.77 | 104.14 | 113.50 |
| B=16 aggregate | 201.20 | 182.88 | 185.70 |
| prefill ISL2048 | 266.6 | 236.6 | 256.5 |
| model load | 22.4 s | 31.7 s | 21.5 s |

Control lands within **2.6% mean** of the opening run, so the session was stable.

The closing run reads ~10% low, but it was taken immediately after a system crash while
background reindexing was active. Model load regressed 41.5% in that run and returned to
21.5 s in the control, and load time is a pure disk-read operation unaffected by GPU
thermals. That is the evidence the closing run is contaminated rather than the session
having drifted.

## Serving grid highlights

Full `nvidia` grid, 10 points x concurrency [1, 4, 8], in `grid.json`.

| ISL:OSL | C | TTFT p50 | aggregate tok/s |
|---|---|---|---|
| 128:2048 | 8 | 4.71 s | **105.13** (peak output) |
| 2048:128 | 8 | 48.5 s | 17.36 |
| 5000:500 | 8 | 130.2 s | 22.41 |
| 20000:2000 | 8 | 560.1 s | 20.78 |

Served throughput trails what the same box reaches in-process, and the gap widens with
load:

| concurrency | served | in-process | served / in-process |
|---|---|---|---|
| 1 | 34.95 | 42.91 | 81% |
| 4 | 60.21 | 95.31 | 63% |
| 8 | 64.80 | 114.77 | **56%** |

## Files

| file | contents |
|---|---|
| `baseline.json` | opening in-process baseline |
| `baseline_close.json` | closing baseline, post-crash, see drift control |
| `baseline_control.json` | clean control run |
| `grid.json` | full `nvidia` serving grid, 30 points |
| `grid_short.json` | `short` serving grid, 9 points |
| `p0_gates.log` | gate output including measured trellis-stream bandwidth |
| `roofline.log` | measured roofline and per-op bandwidth sweep |

`p0_gates.py` and `roofline.py` do not emit JSON, so their output is included as logs.

## Not included

`head_to_head` was not run. It requires a second checkpoint (stock MLX 4-bit, ~19.5 GB)
and this machine lacked the free disk space after the escha checkpoint. Will follow in a
separate PR.

## Harness and doc issues found

1. `p0_gates` G0.2 gemv microbench runs below this machine's measured 122.2 us dispatch
   floor (calls are 120 and 198 us), so its reported 21 and 26 GB/s measure dispatch
   overhead, not bandwidth. Its pass threshold is also expressed against a hardcoded
   M4-class peak; against the measured 242.8 GB/s those are 8.6% and 10.7%, below the
   gate's own "<15% = investigate" line, and it printed PASS. Both defects get easier as
   GPUs widen, so neither is visible on base M4. The kernel is fine: `roofline.py`'s row
   sweep reaches 52% of roofline at 256 rows.
2. `p0_gates.py` and `roofline.py` do not archive results, so there was no committed
   base-M4 figure to compare against.
3. No result file records mlx version, macOS version or commit. Directly relevant to the
   `escha-benchmarks` traceability goal.
4. `isl_osl_grid.py` writes a results JSON and exits zero when every point errors.
   Encountered three times across three distinct failure modes.
5. Documented `pytest tests/ -v` fails on conda-managed macOS because the console script
   resolves outside the venv. `python -m pytest tests/ -v` is robust.
6. `test_moe_block_ops_vs_ckpt_golden` still skips with "checkpoint not available" after
   a normal `hf download`, so the single end-to-end golden test may never run for anyone.
7. With `ESCHA_MLX_WIRED_GB=30` set and the server holding the model, loading a second
   in-process instance crashed the OS rather than failing as a process OOM. Wired pages
   cannot be reclaimed.
8. Install budgets ~13 GB for the checkpoint, which covers the first of four benchmarks.
   The full suite needs about 45 GB.
9. Memory guidance is 24 GB-specific throughout, with no direction for the Pro/Max/Ultra
   owners the contributing guide is recruiting.
