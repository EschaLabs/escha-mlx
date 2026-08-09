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
| Memory bandwidth | 273 GB/s advertised, **243.7 GB/s measured (89%)** |
| GPU working-set cap | 38.65 GB (no sysctl override) |
| macOS | 15.7.3 (24G419) |
| Python | 3.12.7, arm64 |

Base M4 for reference: 10-core GPU, 101 GB/s measured of 120 advertised (84%).
Bandwidth ratio between the machines: **2.41x**. GPU core ratio: **2.0x**.

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

The base-M4 machine table in `docs/PERFORMANCE.md:12` records macOS 26.5.2,
`mlx` 0.32.0 and `mlx-lm` 0.31.3. This machine used the same MLX stack but macOS
15.7.3. That major OS-version gap may affect GPU scheduling and remains an uncontrolled
difference.

## Run conditions

- Original session on 2026-08-04; matched-flags rerun on 2026-08-05.
- Laptop on AC power, lid open, no external display.
- Serving and in-process model runs used `ESCHA_MLX_WIRED_GB=30` (cap 38.65 GB).
  `p0_gates.log` records the default wired limit of 0 for that gate run.
- Other applications closed during measurement.
- Benchmarks ran in the requested order: `p0_gates`, `baseline`, `isl_osl_grid`.

### Serving commands

The original `grid.json` and `grid_short.json` runs used the mlx-lm defaults,
`--decode-concurrency 32` and `--prompt-concurrency 8`:

```bash
ESCHA_MLX_WIRED_GB=30 escha-mlx-server \
    --model /Users/<user>/models/escha-w2 --port 8080 --prefill-step-size 256
python bench/isl_osl_grid.py --model /Users/<user>/models/escha-w2 \
    --grid nvidia --concurrency 1,4,8 \
    --out bench/results/m4-pro-48gb/grid.json
python bench/isl_osl_grid.py --model /Users/<user>/models/escha-w2 \
    --grid short --concurrency 1,4,8 \
    --out bench/results/m4-pro-48gb/grid_short.json
```

The no-step control kept those defaults and omitted only `--prefill-step-size 256`:

```bash
ESCHA_MLX_WIRED_GB=30 escha-mlx-server \
    --model /Users/<user>/models/escha-w2 --port 8080
python bench/isl_osl_grid.py --model /Users/<user>/models/escha-w2 \
    --grid 2048:128 --concurrency 8 \
    --out bench/results/m4-pro-48gb/grid_2048x128_c8_nostep.json
```

The matched-flags rerun used the base-M4 server settings recorded in
`docs/BRINGUP_AND_PERF.md` lines 220-223 and 324-327:

```bash
ESCHA_MLX_WIRED_GB=30 escha-mlx-server \
    --model /Users/<user>/models/escha-w2 --port 8080 \
    --prefill-step-size 256 --decode-concurrency 16 --prompt-concurrency 2
.venv/bin/python -u bench/isl_osl_grid.py \
    --model /Users/<user>/models/escha-w2 \
    --grid 128:128,1000:1000,2048:128 --concurrency 1,8 \
    --out bench/results/m4-pro-48gb/grid_matched_flags.json
```

For the headline repeatability check, the server stayed on those matched flags. After
one unrecorded warm-up at the same shape, the harness measured five consecutive
`128:128` trials at C=1 and C=8:

```bash
.venv/bin/python -u bench/isl_osl_grid.py \
    --model /Users/<user>/models/escha-w2 \
    --grid 128:128,128:128,128:128,128:128,128:128 --concurrency 1,8 \
    --out bench/results/m4-pro-48gb/grid_matched_flags_repeats.json
```

## Correctness

Everything passes.

- `python -m pytest tests/` : 169 passed, 1 skipped.
- `bench/p0_gates.py` : ALL GATES PASS. Every Metal kernel path bit-exact against the
  committed goldens on 20-core Apple GPU silicon.
- Serving smoke test correct (`17*23` returns `391`), generations coherent.
- Resident memory 11.41 GiB = 12.25 GB, identical to the committed base-M4 figure.
  (The harness that printed this divided by 1024³ while labeling the result "GB";
  fixed 2026-08-09.)

## Headline

Aggregate output tok/s at 128:128. The M4 Pro row is the median of five consecutive
trials with the baseline-matched server flags in `grid_matched_flags_repeats.json`; the
M5 Pro result from PR #1 is included for context:

| chip | bandwidth | C=1 | vs base | C=8 | vs base |
|---|---|---|---|---|---|
| base M4 | 101 GB/s | 23.53 | 1.00x | 47.31 | 1.00x |
| M4 Pro (matched flags, median of 5) | 243.7 GB/s | 31.90 | **1.36x** | 60.07 | 1.27x |
| M5 Pro (#1) | 270.6 GB/s | 36.20 | **1.54x** | 108.17 | 2.29x |

The matched M4 Pro medians reach 1.36x the base M4 at C=1 and 1.27x at C=8. The five
M4 Pro measurements span 31.75-32.04 tok/s at C=1 and 59.91-60.12 tok/s at C=8. The
M5 Pro scales much further at C=8, but this is not a paired hardware A/B: the runs
differ in macOS version, runtime revision, GPU configuration and memory capacity.

## Detail: time to first token under concurrency

Comparing `m4-base-24gb/grid_fused.json` with the matched-flags M4 Pro rerun at
identical grid points, server flags, script and request counts:

| point | base M4 TTFT | M4 Pro TTFT | |
|---|---|---|---|
| 128:128, C=1 | 0.806 s | 0.619 s | Pro 1.30x faster |
| 128:128, C=8 | 4.434 s | 4.888 s | level, within session spread |
| 1000:1000, C=1 | 4.082 s | 3.674 s | Pro 1.11x faster |
| 1000:1000, C=8 | 10.368 s | 9.766 s | level, within session spread |
| 2048:128, C=1 | 8.109 s | 8.112 s | level |
| 2048:128, C=8 | 18.133 s | 23.793 s | Pro 1.31x slower |

The original M4 Pro server used mlx-lm's defaults of decode concurrency 32 and prompt
concurrency 8, while the base M4 used 16 and 2. Matching those flags cuts the M4 Pro
2048:128 C=8 median TTFT from 48.525 s to 23.793 s and the 1000:1000 C=8 median from
27.093 s to 9.766 s. The original claim that the M4 Pro was about 2.7x slower under
concurrency was therefore a configuration artifact.

The matched 2048:128 C=8 p50/p99 split is 23.793/56.850 s, close to the base M4's
18.133/61.6 s queueing pattern. In-process prefill at ISL 2048 is 258.9 tok/s here
against 264 tok/s on the base M4, so single-stream prefill remains roughly flat.

The two 1.1x rows above sit inside this machine's measured session-to-session spread
(see "Repeated 128:128 measurements") and are reported as level rather than directional.

### Prefill step size control

`grid_2048x128_c8_nostep.json` repeats 2048:128 at C=8 under the original 32/8 server
settings with `--prefill-step-size 256` omitted and nothing else changed:

| 2048:128, C=8, 32/8 server | TTFT p50 | TTFT p99 | aggregate tok/s |
|---|---:|---:|---:|
| with `--prefill-step-size 256` (`grid.json`) | 48.525 s | 48.564 s | 17.36 |
| without (`grid_2048x128_c8_nostep.json`) | 84.646 s | 84.737 s | 13.47 |

Omitting the flag raises TTFT by 1.74x and costs 22.4% of aggregate throughput, so the
documented setting helps on this machine rather than being 24 GB-specific. Both runs
completed 16 requests with no errors and hit the exact OSL.

### Where prefill time goes

`prefill_profile.json` ablates the MoE block from a single-stream prefill at four
sequence lengths. Share of prefill attributable to the MoE block, computed as
`(full_ms - no_moe_ms) / full_ms`:

| S | full prefill | MoE block share |
|---|---:|---:|
| 128 | 270.2 ms | 68.9% |
| 256 | 928.4 ms | 83.3% |
| 512 | 1657.9 ms | 81.5% |
| 1024 | 3080.3 ms | 77.8% |

The MoE block accounts for 77.8-83.3% of prefill at S=256-1024, falling to 68.9% at
S=128 where fixed per-dispatch cost is a larger fraction of a shorter run. Ablating the
expert FFNs alone (`no_experts_ms`) gives 63.9 / 81.5 / 79.4 / 75.7% across the same
lengths, so router, dispatch and combine account for roughly 2 to 5 points. The LM head
is negligible: `no_head_ms` is 261.9 ms against 270.2 ms full at S=128, about 3%.

Prefill on this checkpoint is therefore an expert-path problem, which is consistent with
the latency-bound decode picture below.

## Detail: single-run scaling sweep across shared grid points

The original matched-flags file contains one measurement at each of six points shared
with `m4-base-24gb/grid_fused.json`. It is retained for workload-shape context, but the
128:128 headline above uses the five-trial medians instead:

| point | base M4 | M4 Pro | ratio |
|---|---|---|---|
| 128:128 C=1 | 23.53 | 33.46 | 1.42x |
| 128:128 C=8 | 47.31 | 65.94 | 1.39x |
| 1000:1000 C=1 | 24.81 | 34.29 | 1.38x |
| 1000:1000 C=8 | 50.43 | 75.76 | 1.50x |
| 2048:128 C=1 | 10.04 | 11.32 | 1.13x |
| 2048:128 C=8 | 13.47 | 15.57 | 1.16x |

Across this single sweep, the median ratio is 1.39x, the mean is 1.33x and the range is
1.13x to 1.50x. No point reaches 2.3x. Each row is one observation, so the 1.13x to
1.50x spread is not separable from the ~10% session variation quantified below; read the
sweep as workload-shape context, not as six measured ratios.

The mechanism is visible in roofline utilization, which *falls* on the faster chip:

| | roofline ceiling | measured decode | utilization |
|---|---|---|---|
| base M4 | 39.3 tok/s | 27.3 | 69.5% |
| M4 Pro | 94.8 tok/s | 41.29 | **43.6%** |

More bandwidth, worse utilization, which is the signature of a latency-bound workload
rather than a bandwidth-bound one. Consistent with the bs1 GEMV diagnosis in
`docs/BRINGUP_AND_PERF.md` section 3.1. Batched decode holds up better than single
stream, which fits: larger batches amortize the fixed per-dispatch cost.

## Drift control

Three in-process `baseline` JSON files are committed. Every row below maps directly to
the named key in `baseline.json`, `baseline_close.json` and `baseline_control.json`:

| metric | JSON key | opening | closing | control |
|---|---|---:|---:|---:|
| decode ISL512 | `B_bs1["512"].decode_tps` | 41.29 | 38.71 | 40.28 |
| B=8 aggregate | `C_batch["8"].aggregate_decode_tps` | 104.67 | 104.14 | 113.50 |
| B=16 aggregate | `C_batch["16"].aggregate_decode_tps` | 179.64 | 182.88 | 185.70 |
| prefill ISL2048 | `B_bs1["2048"].prefill_tps` | 258.9 | 236.6 | 256.5 |

The absolute opening-to-control deltas are 2.4%, 8.4%, 3.4% and 0.9%. Their arithmetic
mean is **3.8%**. This is small enough not to overturn the broad conclusions, but the
8.4% row means differences near 10% should be treated as run-to-run variation rather
than described as stable. Model-load time was removed because it is not present in the
committed JSON or logs.

## Serving grid highlights

The original full `nvidia` grid, 10 points x concurrency [1, 4, 8], is in `grid.json`.
It used the server's default decode/prompt concurrency of 32/8, so it is retained as raw
data but is not used for the matched headline comparison.

| ISL:OSL | C | TTFT p50 | aggregate tok/s |
|---|---|---|---|
| 128:2048 | 8 | 4.71 s | **105.13** (peak output) |
| 2048:128 | 8 | 48.5 s | 17.36 |
| 5000:500 | 8 | 130.2 s | 22.41 |
| 20000:2000 | 8 | 560.1 s | 20.78 |

### Repeated 128:128 measurements

The two original C=8 files disagree materially despite using the same 32/8 server
settings. The initial matched-flags result is also a single observation:

| file | server decode/prompt concurrency | C=8 aggregate tok/s |
|---|---|---:|
| `grid_short.json` | 32/8 | 64.80 |
| `grid.json` | 32/8 | 71.83 |
| `grid_matched_flags.json` | 16/2 | 65.94 |
| `grid_matched_flags_repeats.json` | 16/2 | **60.07 median**, 59.91-60.12 range |

The two original same-configuration runs span 64.80 to 71.83 tok/s, a 10.8% increase
relative to the lower measurement. The committed artifacts do not establish the cause,
so neither value is used for the headline. The single 65.94 tok/s matched result is also
excluded from the headline.

The spread is not confined to the 32/8 runs. `grid_matched_flags.json` reads 65.94 tok/s
and the five-trial median is 60.07 tok/s at **identical** 16/2 flags, a 9.8% gap between
sessions, against a 0.35% spread within the repeat block. Session-to-session variation
therefore dominates run-to-run variation on this machine by more than an order of
magnitude. Only the 128:128 headline carries a measured error bar; every other figure in
this directory is a single observation and should be read with roughly 10% uncertainty.
That is wide enough to make differences below about 1.15x uninformative, but not wide
enough to affect the headline conclusion, since even a 10% upward correction on the
largest observed ratio falls well short of the 2.3x projection. In the controlled five-trial matched-flags repeat, C=8 is
60.07 / 60.12 / 59.91 / 59.96 / 60.11 tok/s: median 60.07 and range 59.91-60.12.
C=1 is 31.94 / 32.04 / 31.79 / 31.75 / 31.90 tok/s: median 31.90 and range
31.75-32.04. Every trial hit the exact OSL with zero cached prompt tokens. The warm-up
was excluded. This establishes a repeatable same-session headline under matched flags;
it does not identify the cause of the earlier 32/8 discrepancy.

Using `grid_short.json` for served throughput and the opening `baseline.json` for the
same box's in-process throughput:

| concurrency | served | in-process | served / in-process |
|---|---|---|---|
| 1 | 34.95 | 41.68 | 84% |
| 4 | 60.21 | 93.64 | 64% |
| 8 | 64.80 | 104.67 | **62%** |

## Files

| file | contents |
|---|---|
| `baseline.json` | opening in-process baseline |
| `baseline_close.json` | closing baseline, post-crash, see drift control |
| `baseline_control.json` | clean control run |
| `grid.json` | full `nvidia` serving grid, 30 points |
| `grid_short.json` | `short` serving grid, 9 points |
| `grid_matched_flags.json` | six shared points with base-M4 server flags |
| `grid_matched_flags_repeats.json` | five measured 128:128 trials at matched flags |
| `grid_2048x128_c8_nostep.json` | default prefill-step-size control |
| `prefill_profile.json` | MoE-block prefill ablation at S=128, 256, 512, 1024 |
| `p0_gates.log` | gate output including measured trellis-stream bandwidth |
| `roofline.log` | measured roofline and per-op bandwidth sweep |

`p0_gates.py` and `roofline.py` do not emit JSON, so their output is included as logs.

## Not included

`head_to_head` was not run. It requires a second checkpoint (stock MLX 4-bit, ~19.5 GB)
and this machine lacked the free disk space after the escha checkpoint. Will follow in a
separate PR.

## Harness and doc issues found

1. `p0_gates` reports 21 GB/s for K2 and 65 GB/s for K3. The independent roofline log
   measures a 129 us dispatch floor and reaches 52% of roofline at 256 rows, so the
   single-shape gate output should not be read as the kernel's bandwidth ceiling.
2. `p0_gates.py` and `roofline.py` do not archive results, so there was no committed
   base-M4 figure to compare against.
3. The original serving files do not record runtime/model revisions or command flags.
   `grid_matched_flags.json` and `grid_matched_flags_repeats.json` record both revisions;
   the full commands remain in this README because the harness does not yet embed server
   flags.
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
