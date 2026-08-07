# Performance — escha MLX runtime

All numbers below were measured on one machine, in one session, with the same
harness. Nothing here is scaled, extrapolated, or taken from a third-party report.

**Test system**

| | |
|---|---|
| Machine | MacBook Pro, **Apple M4 (base)** — 10-core GPU, 24 GB unified memory |
| Memory bandwidth | 120 GB/s advertised; **101 GB/s measured** (84%, streaming read) |
| macOS / MLX | 26.5.2 / `mlx` 0.32.0, `mlx-lm` 0.31.3 |
| Model | `Qwen3.6-35B-A3B-Escha-W2` — 35B total, 3B active, 12.3 GB resident |

> This is the **entry-level** M4. An M4 Pro has 2.3× the bandwidth, an M4 Max
> 4.5×, an M3 Ultra 6.8×. Decode is bandwidth-bound, so it scales close to
> linearly with that number — read these as a floor for the product line, not a
> typical result.

## Single stream

| | |
|---|---|
| Decode | **27.0 tok/s** |
| Prefill | **264 tok/s** (ISL 512) · 264 (2048) |
| Peak memory | 12.1 GB (short context) · 12.2 GB (2k context) |

Decode is memory-bound: 2.57 GB moved per token against 101 GB/s gives a hard
ceiling of 39.3 tok/s on this chip, and we reach 69% of it. **100 tok/s from a
single stream is not physically possible here** — it would need 250 GB/s.

## Concurrency

Aggregate decode throughput, in-process:

| batch | tok/s | peak memory |
|---|---|---|
| 1 | 27.0 | 12.3 GB |
| 8 | 60.3 | 12.6 GB |
| 16 | 108.5 | 13.0 GB |
| 32 | 151.6 | 13.7 GB |
| 64 | 162.5 | 15.2 GB |
| 96 | 181.2 | 16.6 GB |
| **128** | **185.6** | **18.1 GB** |

Batch 128 fits inside the default ~19 GB working-set cap. Requires
`ESCHA_MLX_WIRED_GB` — see [INSTALL.md](INSTALL.md).

## Served endpoint — NVIDIA ISL/OSL grid

Through the OpenAI-compatible server, `ignore_eos` so every request emits exactly
OSL tokens. ISL = input length, OSL = output length, C = concurrency.

| ISL | OSL | C | TTFT p50 (s) | TTFT p99 (s) | TPOT (ms) | output tok/s | total tok/s | vs pre-fusion |
|---|---|---|---|---|---|---|---|---|
| 128 | 128 | 1 | 0.81 | 0.9 | 36.4 | **23.5** | 49.4 | +13% |
| 128 | 128 | 8 | 4.43 | 6.3 | 136.0 | **47.3** | 99.1 | +12% |
| 128 | 128 | 16 | 3.87 | 13.4 | 204.2 | **65.3** | 136.8 | +26% |
| 128 | 1024 | 1 | 0.85 | 0.9 | 35.6 | **27.4** | 31.2 | +13% |
| 128 | 1024 | 8 | 4.48 | 6.4 | 130.9 | **59.7** | 67.9 | +1% |
| 128 | 1024 | 16 | 3.99 | 13.5 | 155.3 | **99.5** | 113.2 | +22% |
| 1000 | 1000 | 1 | 4.08 | 4.2 | 36.0 | **24.8** | 49.9 | +13% |
| 1000 | 1000 | 8 | 10.37 | 30.9 | 147.7 | **50.4** | 101.5 | +5% |
| 1000 | 1000 | 16 | 11.10 | 64.0 | 201.7 | **74.3** | 149.5 | +24% |
| 2048 | 128 | 1 | 8.11 | 8.4 | 36.2 | **10.0** | 171.5 | +20% |
| 2048 | 128 | 8 | 18.13 | 61.6 | 449.9 | **13.5** | 230.3 | +24% |
| 2048 | 128 | 16 | 19.29 | 127.3 | 933.9 | **14.3** | 244.5 | +28% |

Peak served output is **99.5 tok/s** (128/1024, C=16); peak total is **244.5
tok/s** (2048/128, C=16). Every point ran with `ignore_eos`, 0 errors and an
OSL hit rate of 1.00, so these are steady-state numbers rather than a decaying
batch.

TTFT still rises with concurrency on prompt-heavy rows — prefill is the
bottleneck on a 10-core GPU and requests queue behind it — but far less than
before: 1000/1000 at C=16 went from 50.5 s to 11.1 s p50. On prompt-heavy
workloads, keep concurrency low.

The final column compares against the same grid before the fused transform
kernel. Decode-dominated rows at low concurrency gain ~12%; the gains reach
+21-25% wherever many expert rows are in flight, and the one flat row
(128/1024 C=8) is the 64-row point where there is little transform work to
fuse.

---

## Head to head: escha W2 vs stock MLX 4-bit

Same base model, same machine, same harness, back to back, with the
escha arm measured first *and* last as a drift control.

* **A** — `EschaLabs/Qwen3.6-35B-A3B-Escha-W2` (this runtime)
* **B** — `mlx-community/Qwen3.6-35B-A3B-4bit` (stock mlx-lm)

### Memory

| | escha W2 | MLX 4-bit |
|---|---|---|
| resident | **12.25 GB** | 19.51 GB |
| peak @ 2048-token prompt | **13.14 GB** | 20.05 GB |

### Throughput

Prefill and single-stream decode:

| ISL | escha prefill | 4-bit prefill | escha decode | 4-bit decode |
|---|---|---|---|---|
| 512 | 264.4 / 262.4 | **344.2** | 28.5 / 28.1 | **43.5** |
| 2048 | 263.6 / 263.7 | **341.3** | 27.7 / 27.8 | **41.7** |

Aggregate decode by batch (escha shown as first run / drift control):

| batch | escha W2 | MLX 4-bit | |
|---|---|---|---|
| 1 | 27.3 / 27.9 | **42.9** | 4-bit 1.55× |
| 2 | 46.5 / 46.4 | **66.1** | 4-bit 1.42× |
| 4 | 58.2 / 58.1 | **86.4** | 4-bit 1.49× |
| 8 | 59.6 / 64.8 | **101.5** | 4-bit 1.63× |
| 16 | **104.0 / 113.3** | out of memory | — |
| 32 | **162.1 / 155.7** | out of memory | — |

The escha arm was measured first *and* last. Drift is ≤1.5% at ISL and low
batch; B=8 and B=16 carry ~8% run-to-run spread on this box and should be read
as approximate.

**Read this honestly in both directions.**

*Below batch 16, MLX 4-bit is faster.* At low batch the dense weights dominate
per-token traffic, and 4-bit dense moves fewer bytes than escha's int8 dense.
escha's 2-bit applies to the routed experts, which are only ~12% of single-stream
traffic because just 8 of 256 experts are read per token.

*From batch 16 up, only escha runs at all.* The 4-bit build exhausts memory
(20.66 GB peak at batch 8 against a 22.02 GB cap) — reproduced in two independent
sessions. This is the regime the footprint buys, and it is where a Mac serving
more than one user actually lives.

The crossover in *bytes moved* comes earlier than the crossover in speed: from
batch 4 escha already reads fewer bytes per step, reaching 1.63× fewer at batch
32, and the throughput advantage lags that.

The reason is not the trellis kernel. Benchmarked in isolation at matched shapes,
it is **faster than MLX's own fused `gather_qmm`** at every decode row count —
0.305 ms vs 0.365 at 8 rows, 2.136 vs 2.490 at 256, 5.000 vs 6.402 at 1024 — and
within 8% at prefill row counts, while moving about half the bytes.

The difference is the codec. Every escha expert call applies a random Hadamard
transform and per-expert input/output scale vectors; affine 4-bit quantization
does neither. In-model that transform pipeline costs **~20% of both prefill and
decode**, and it is work the 4-bit build never performs. Combined with escha's
int8 dense weights against their 4-bit dense, it accounts for the gap without any
kernel being slow.

That ~20% is the open work item, and it is a memory-traffic problem rather than
an arithmetic one — the transforms materialise several f32 intermediates per
layer that a fused kernel would keep in registers.

**Choose 4-bit if** you run one stream at a time on a machine with ≥32 GB.
**Choose escha W2 if** you serve concurrent requests, run on 24 GB, or need room
for long context and other applications alongside the model.

---

## Apple M5 Pro 24 GB — original-code baseline

This is a separate machine section, not a scaled projection from the base M4. All
committed measurements below were repeated after an unrelated 7.3 GB model service
was stopped and disabled; the earlier M5 files are not used.

| | |
|---|---|
| Machine | MacBook Pro (Mac17,9), **Apple M5 Pro**, 16-core GPU, 24 GB unified memory |
| Power | AC attached; battery 80% |
| macOS / MLX | 26.5.2 (25F84) / `mlx` 0.32.0, `mlx-lm` 0.31.3 |
| Metal | `applegpu_g17s`; recommended working set 19.07 GB |
| Model | local `Qwen3.6-35B-A3B-Escha-W2` revisions `32016b7946fa1a1965c40deed9daac071b512a64` / `1b7237f0886a10b4bd92cd7653090cd7381ae199` (manifests differ only in `README.md`); 11.41 GB resident |
| Runtime revision | `79ba35e84517b2770ca00a1fe76091ff4144de37` |
| Runtime settings | repository defaults; MLX memory limit 19.0 GB; wired limit 0 except 19.0 GB for B=128 |

### Correctness status

The Paris, 17×23=391 and thinking-text anchors pass. Replicated rows are identical
and match the B=1 sequence at B=1/8/16/32. `bench/p0_gates.py` reports `ALL GATES
PASS`, including hash/LUT decode, GEMV values and Q8 repack; its K2/K3 DRAM-side
microbench varied across three consecutive runs: K2 24–49 GB/s and K3 33–54 GB/s.
The value gates passed every time; these short microbench figures are retained as a
stability observation and are not used for the throughput headlines below.

At the benchmarked runtime revision, the complete suite was **165 passed, 4 failed,
1 skipped**. All four failures were the former
`tests/test_fused_had.py::test_fused_matches_op_chain`: the primary
`2e-3` relative-error bound passes, but 60.35–61.11% of fp16 bit patterns differ
from the dense-matmul op chain on this M5 Pro, exceeding the test's 1% secondary
bound. The fused path remains bit-reproducible and its independent NumPy-reference
test passes. No gate was excluded or loosened, so this is machine-characterization
data rather than a merge-green validation result. This was subsequently resolved
by replacing the dense-matmul test oracle and production op chain with MLX's native
butterfly; the performance tables below still describe the recorded revision and
have not been rerun.

#### M5 Pro resolved issue: dense-matmul test oracle and TF32

On this M5 Pro, MLX 0.32.0 selected its NAX/TF32 path for the FP32 dense matmul in
the old comparison op chain when `MLX_ENABLE_TF32` was unset. The fused Metal kernel
performed an explicit FP32 butterfly. The test therefore combined different
operation order with different FP32 matmul precision, making the raw fp16 comparison
fail widely even though the maximum-error gate remained green.

The four test shapes were measured again in fresh processes with TF32 at its
default and with `MLX_ENABLE_TF32=0`. Counts compare the raw fp16 bit patterns;
the test requires the op-chain mismatch rate to be below 1%.

| rows × IC | default TF32: fused vs op chain | TF32=0: fused vs op chain | TF32=0: fused vs NumPy |
|---:|---:|---:|---:|
| 8 × 512 | 2,472 / 4,096 (60.3516%) | 5 / 4,096 (0.1221%) | 0 / 4,096 |
| 96 × 2048 | 119,555 / 196,608 (60.8088%) | 222 / 196,608 (0.1129%) | 0 / 196,608 |
| 300 × 1024 | 187,741 / 307,200 (61.1136%) | 369 / 307,200 (0.1201%) | 0 / 307,200 |
| 2048 × 2048 | 2,557,851 / 4,194,304 (60.9839%) | 4,933 / 4,194,304 (0.1176%) | 0 / 4,194,304 |

In that historical implementation,
`MLX_ENABLE_TF32=0 .venv/bin/python -m pytest tests/ -v` reported **169 passed,
1 skipped** and the targeted subset reported **8 passed**. This was not a bit-exact
resolution: 0.113–0.122% of fused/dense-matmul outputs still differed, and the test
passed only because that tail was below 1%.

The current implementation instead uses `mx.hadamard_transform` for the native
op chain. Both paths now execute the same radix-2 butterfly order, and the fused
test requires zero differing FP16 bits across the four shapes above with TF32 left
at its default. `ESCHA_MLX_FUSED_HAD=0` selects this native chain. The independent
NumPy comparison remains tolerance-based because its matrix multiplication has a
different reduction order. The complete current suite reports **176 passed,
1 skipped** both under default TF32 and with `MLX_ENABLE_TF32=0`. No correctness
gate is loosened or skipped.

Turning off TF32 also has a measurable cost. A same-session default/off/default
A/B/A used `bench.baseline` phases B and C at runtime revision `aec1ea8` and model
revision `1b7237f`; the three model safetensor LFS SHA-256 values are identical to
revision `32016b7`, so this is not a weight change.

| metric | default A1 | TF32=0 B | default A2 | B vs mean(A1,A2) |
|---|---:|---:|---:|---:|
| prefill, ISL 512 | 686.6 | 550.3 | 689.2 | −20.0% |
| prefill, ISL 2048 | 675.6 | 542.8 | 663.2 | −18.9% |
| decode, ISL 512 | 46.91 | 45.54 | 46.31 | −2.3% |
| decode, ISL 2048 | 47.47 | 45.81 | 46.06 | −2.0% |
| aggregate decode, B=1 | 47.96 | 46.28 | 46.72 | −2.2% |
| aggregate decode, B=8 | 180.51 | 178.05 | 179.04 | −1.0% |
| aggregate decode, B=16 | 239.57 | 228.24 | 237.94 | −4.4% |
| aggregate decode, B=32 | 349.34 | 336.81 | 350.49 | −3.7% |

Peak memory was unchanged at every corresponding point. The first ISL-128 arm
was cold and the batched-prefill closing arm drifted substantially, so neither is
used for a performance claim. The stable observations are a roughly 19–20%
long-prompt prefill loss and a 1–4% decode loss; disabling TF32 was therefore a
useful diagnosis of the former dense-matmul oracle, not a current workaround or
default-performance recommendation. Raw A/B/A JSON:
[tf32_aba.json](../bench/results/m5-pro-24gb/tf32_aba.json).

### In-process ABCD baseline

```bash
.venv/bin/python -u -c \
    'import mlx.core as mx; mx.set_memory_limit(19_000_000_000); from bench.baseline import main; main()' \
    --model ./escha-w2 --phases ABCD \
    --isls 128,512,2048 --batches 1,8,16,32 --decode-steps 32 \
    --batch-isl 128 --prefill-chunk 256 \
    --out bench/results/m5-pro-24gb/baseline_abcd.json
```

| ISL | prefill tok/s | decode tok/s | peak memory |
|---:|---:|---:|---:|
| 128 | 624.0 | **46.66** | 11.95 GB |
| 512 | **684.2** | 46.56 | 12.25 GB |
| 2048 | 677.3 | 46.30 | 12.28 GB |

| batch | prefill tok/s | aggregate decode tok/s | per-sequence tok/s | peak memory | correctness |
|---:|---:|---:|---:|---:|---|
| 1 | 623.2 | 46.63 | 46.63 | 11.95 GB | OK |
| 8 | 843.6 | 179.08 | 22.38 | 13.71 GB | OK |
| 16 | **852.1** | 239.10 | 14.94 | 14.67 GB | OK |
| 32 | 765.1 | **347.72** | 10.87 | 17.16 GB | OK |

After an ISL-512 prefill the caches total 43.42 MB: 32.93 MB across 30
`GDNStateCache` layers and 10.49 MB across 10 trimmable `KVCache` layers.

### Step-synchronized decode and repeatability

`bench.sweep_kernel_variants.run` uses a 16-token prefill, eight warmups, 24 timed
steps and an `mx.eval` synchronization after every decode step. Every batch was called
five times without changing any runtime strategy. B=128 used the same workload and
repository defaults with a 19 GB MLX memory limit and a 19 GB wired limit.

| batch | runs | median tok/s | min–max | spread | peak memory |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 45.52 | 45.39–46.02 | 1.39% | 12.30 GB |
| 8 | 5 | 178.97 | 178.25–179.63 | 0.77% | 12.65 GB |
| 16 | 5 | 226.23 | 225.64–226.83 | 0.53% | 13.00 GB |
| 32 | 5 | 328.81 | 326.97–329.50 | 0.77% | 13.75 GB |
| 128 | 5 | **537.08** | 522.83–553.61 | 5.89% | **18.04 GB** |

The ABCD Phase C loop and this step-synchronized loop have different synchronization
semantics, so their absolute throughput figures are reported separately. The five-run
rows establish within-session stability but do not run Phase C replication-invariance
checks. This is not a candidate-vs-baseline A/B.

### Current output Hadamard fusion A/B/A

The summary and repeatability tables above are historical measurements from runtime
revision `79ba35e`, whose expert output transform was a dense 128x128 matmul. They are
not measurements of the native butterfly now on `main`. The current output-fusion
candidate was therefore compared directly with that native `main` path in one loaded
model process: native A1, fused B, then native A2 as the drift control. Each decode
point uses a 16-token prefill, eight warmup steps, 24 timed steps, per-step evaluation
and five repeats. The wired limit remained at zero; every completed arm produced the
same token hash.

```bash
.venv/bin/python -u bench/sweep_output_had.py --model ./escha-w2 \
    --isls 512 --batches 1,8,16,32 --repeats 5 --decode-steps 24 \
    --out bench/results/m5-pro-24gb/output_had_ab.json
```

| workload | native A1 | output-fused | native A2 | fused vs mean(native) | fused spread | peak memory |
|---|---:|---:|---:|---:|---:|---:|
| prefill, ISL 512 | 551.45 | **583.09** | 550.69 | **+5.8%** | 0.42% | 13.09 GB |
| decode, B=1 | 42.00 | **44.61** | 42.48 | **+5.6%** | 1.30% | 12.30 GB |
| decode, B=8 | 147.25 | **151.25** | 147.51 | **+2.6%** | 0.74% | 12.64 GB |
| decode, B=16 | 186.65 | **188.23** | 186.62 | **+0.9%** | 0.98% | 12.98 GB |
| decode, B=32 | 267.67 | **271.48** | 264.90 | **+2.0%** | 1.16% | 13.72 GB |

This isolates the current change: output fusion is faster than the native butterfly
at every tested batch and is bit exact with it. It does not establish a comparison
with the old dense-matmul table, whose NAX/TF32 execution and reduction order differ.
An attempted B=128 extension was discarded after macOS panicked in `IOGPUFamily`
(`IOGPUMemory.cpp:550`, `completeMemory() prepare count underflow`) during the first
native arm. No B=128 result is claimed, and the rerun was deliberately limited to
B<=32. Raw A/B/A JSON: [output_had_ab.json](../bench/results/m5-pro-24gb/output_had_ab.json).

### Roofline

The committed roofline harness measured 270.6 GB/s peak streaming read bandwidth,
88% of the chip's advertised 307 GB/s, and a 187.3 us dispatch floor. The loaded
model's byte ledger is 2.386 GB/token.

```bash
.venv/bin/python -u bench/roofline.py --model ./escha-w2 \
    --out bench/results/m5-pro-24gb/roofline.json
```

| batch | measured tok/s | modeled GB/step | effective GB/s | roofline efficiency |
|---:|---:|---:|---:|---:|
| 1 | 44.87 | 2.386 | 107.1 | 40% |
| 8 | 177.14 | 4.492 | 99.5 | 37% |
| 16 | 221.15 | 6.899 | 95.4 | 35% |

### Served endpoint

The M4 result set contains both a historical unfused baseline and the current
default-fused grid, so the same two artifacts were captured here. Both use the
same server settings: decode concurrency 16, prompt concurrency 2 and prefill
step size 256. `baseline.json` fixes every point at 16 requests with
`ESCHA_MLX_FUSED_HAD=0`; `grid_fused.json` uses the harness default of 4/16/32
requests at C=1/8/16. Because request counts differ and no closing drift-control
arm was run, these columns are reported side by side, not as a causal A/B claim.

```bash
# Terminal 1: prepend ESCHA_MLX_FUSED_HAD=0 for baseline.json; omit it for grid_fused.json.
ESCHA_MLX_FUSED_HAD=0 .venv/bin/python -m escha_mlx.server \
    --model ./escha-w2 --port 8080 --decode-concurrency 16 \
    --prompt-concurrency 2 --prefill-step-size 256

# Terminal 2: the unfused historical-baseline request shape.
.venv/bin/python bench/isl_osl_grid.py --model ./escha-w2 \
    --grid 128:128,128:1024,1000:1000,2048:128 --concurrency 1,8,16 \
    --requests-per-point 16 --out bench/results/m5-pro-24gb/baseline.json

# Restart Terminal 1 without ESCHA_MLX_FUSED_HAD, then use the harness request defaults.
.venv/bin/python bench/isl_osl_grid.py --model ./escha-w2 \
    --grid 128:128,128:1024,1000:1000,2048:128 --concurrency 1,8,16 \
    --out bench/results/m5-pro-24gb/grid_fused.json
```

| ISL | OSL | C | fused TTFT p50/p99 | fused TPOT | unfused output tok/s | fused output tok/s | fused total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 1 | 0.451 / 0.633 s | 24.01 ms | 35.10 | 36.20 | 75.94 |
| 128 | 128 | 8 | 3.273 / 3.480 s | 51.76 ms | 107.20 | 108.17 | 226.59 |
| 128 | 128 | 16 | 2.596 / 7.233 s | 101.62 ms | 122.65 | 124.70 | 261.36 |
| 128 | 1024 | 1 | 0.506 / 0.510 s | 24.65 ms | 37.73 | 39.73 | 45.16 |
| 128 | 1024 | 8 | 3.272 / 3.476 s | 49.41 ms | 149.95 | 152.85 | 173.78 |
| 128 | 1024 | 16 | 3.213 / 7.300 s | 81.16 ms | 177.62 | **188.01** | 213.76 |
| 1000 | 1000 | 1 | 1.817 / 1.914 s | 25.35 ms | 35.85 | 36.79 | 74.04 |
| 1000 | 1000 | 8 | 8.730 / 13.119 s | 54.16 ms | 124.56 | 127.80 | 257.17 |
| 1000 | 1000 | 16 | 6.253 / 27.297 s | 99.42 ms | 142.97 | 147.82 | 297.44 |
| 2048 | 128 | 1 | 3.347 / 3.640 s | 24.46 ms | 18.66 | 19.62 | 335.43 |
| 2048 | 128 | 8 | 11.422 / 25.109 s | 153.26 ms | 29.49 | 33.06 | 565.17 |
| 2048 | 128 | 16 | 9.099 / 52.203 s | 379.17 ms | 29.88 | 33.68 | **575.86** |

All 24 served rows completed with zero errors, no cached prompt tokens and OSL hit
rate 1.00. The stock-4-bit `head_to_head` artifacts were intentionally skipped for
this machine and are not inferred from the M4 data. B=128 was measured separately with
the step-synchronized in-process harness above, not with the served endpoint.

Raw JSON: [ABCD baseline](../bench/results/m5-pro-24gb/baseline_abcd.json),
[decode repeats](../bench/results/m5-pro-24gb/baseline_repeats.json),
[roofline](../bench/results/m5-pro-24gb/roofline.json),
[unfused served baseline](../bench/results/m5-pro-24gb/baseline.json), and
[default-fused served grid](../bench/results/m5-pro-24gb/grid_fused.json).

---

## Quality

Quantization quality is a property of the checkpoint, not this runtime, and is
documented on the [model card](https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2).
What this runtime guarantees is that it reproduces the codec exactly: the Metal
kernels are gated **bit-exact against the committed reference goldens**.

Three intentional deviations, all measured and all documented in-source. Note
what they have in common: each is a **reordering** of a sum, never a reduction in
precision, and each leaves decode bit-reproducible.

1. **Routed-expert accumulation** is a fixed-order reduction rather than an
   atomic scatter order. This is what makes decode bit-reproducible — same input,
   same logits, every run and every process.
2. **GDN recurrent state in fp16.** Over 384 teacher-forced decode steps: 99.74%
   top-1 agreement, mean KL 2.8e-4 versus f32, and the error is flat over the
   run rather than accumulating. For calibration, changing batch size from 1 to
   16 perturbs logits *more* than this does. `ESCHA_MLX_GDN_STATE=fp32` restores
   the f32 behaviour at ~10% throughput cost above batch 32.
3. **Fused expert transforms.** Both the fused kernels and the native op chains
   use a radix-2 butterfly instead of a dense 128×128 matmul. The input kernel
   combines the row/scale gather, transform, output scale and cast; the output
   kernel combines the transform, output-scale gather and cast. Both preserve the
   native chain's f32 operation order and produce bit-exact final f16 outputs while
   avoiding the intermediate tensors. `ESCHA_MLX_FUSED_HAD=0` selects the native
   input and output chains.

## Reproducing

From the source tree:

```bash
python bench/baseline.py       --model ./escha-w2 --phases ABCD
python bench/isl_osl_grid.py   --model ./escha-w2 --grid nvidia --concurrency 1,8,16
python bench/head_to_head.py   --a ./escha-w2 --b ./qwen36-4bit --isls 512,2048 --batches 1,2,4,8,16,32
```

Every JSON-producing benchmark records `escha_mlx_git_revision` and
`model_hf_revision`. The model revision is read locally from Hugging Face
download metadata (or a Hub snapshot path), so collecting it does not require
network access. Reports that were previously arrays store their measurement rows
under `results`, leaving exactly one top-level set of revision fields.

Measurement notes, learned the hard way and worth repeating if you benchmark
this yourself: warm up per prompt shape (Metal specialises kernels per shape),
call `mx.clear_cache()` after prefill (otherwise decode reads ~20× slow), and pair
A/B runs within one session with the baseline measured first *and* last — on this
box thermal drift alone moves results by 1–3%, and once by 7%.
