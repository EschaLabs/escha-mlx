# MTP coded-projection experiments

This log records the small-row coded-linear optimizations tried for Qwen3.8
native MTP and then removed because they did not improve the complete target
verification enough to justify production complexity.  It exists so the same
ideas are not reintroduced from convincing-looking single-kernel results.

## Measurement contract

- Date: 2026-09-05
- Runtime base: `84dd1860bf5ce89e7f8fd6a5a31faa98099807ac`
- Model: `/Users/tu/Dev/repos/models/Qwen3.8-27B-Escha-W2`
- Machine: Apple M5 Pro, 16-core GPU, 24 GB unified memory
- Software: macOS 26.6.2, MLX 0.32.0, mlx-lm 0.31.3
- Target shapes: fixed tree M4 and the prospective/batched M8 shape
- Timing: one model load per experiment, both arms compiled and warmed, ABBA
  interleaving, `mx.eval` plus `mx.synchronize` for every sample

An implementation must improve complete M4/M8 target verification by at least
3%, exceed twice the measured drift, and preserve M1.  A final candidate must
also improve a 256-token, multi-prompt end-to-end run by at least 2%.  A
reassociated kernel must be deterministic and tolerance-gated; a kernel which
keeps operation order must be bit-exact.

Absolute times drift with temperature, so decisions use the paired ratios from
one process rather than comparing numbers from separate runs.

## Results

| Experiment | Scope | Reference | Candidate | Speedup | Decision |
|---|---|---:|---:|---:|---|
| Parallel output WHT across rows | full target M4 | 100.771 ms | 100.622 ms | 1.001x | remove |
| Parallel output WHT across rows | full target M8 | 146.928 ms | 147.370 ms | 0.997x | remove |
| Native R8 matrix core + fused output WHT | full target M8 | 146.594 ms | 145.283 ms | 1.009x | remove |
| Fused SwiGLU + down input WHT | one complete MLP, M4 | 1.036 ms | 1.040 ms | 0.996x | remove |
| Fused SwiGLU + down input WHT | one complete MLP, M8 | 1.509 ms | 1.499 ms | 1.006x | remove |
| `down_proj` on-chip split-K=2 | eight layers, M4 | 2.283 ms | 2.264 ms | 1.009x | remove |
| `down_proj` on-chip split-K=2 | eight layers, M8 | 3.344 ms | 3.234 ms | 1.034x | remove |
| Code-stream prefetch PF2 | full target M8 | 146.839 ms | 145.137 ms | 1.012x | remove |
| Code-stream prefetch PF4 | full target M8 | 147.047 ms | 148.380 ms | 0.991x | remove |
| Code-stream prefetch PF8 | full target M8 | 146.748 ms | 267.678 ms | 0.548x | remove |
| `mx.compile` fixed M4 target body | full target M4 | runs normally | cannot capture stateful body | n/a | reject |

No experimental kernel, switch, allowlist, or alternate runtime path from this
campaign remains in the code.

After removing the candidates, an 11-sample profile measured 101.080 ms for
the complete M4 target body plus vocabulary head (M1 72.462 ms, M8 146.992
ms). The final post-fix measurement at revision `b30bb9b` used a warmed ABBA
run with a 20-token chat prompt and 256 output tokens, including prefill. It
measured autoregressive generation at 60.128 ms/token and fixed-tree M4 at
46.598 ms/token, or 1.290x, with mean emission 2.844 tokens per target round;
both arms had the same token digest. As documented for this checkpoint, greedy
digests can still diverge at shape-dependent near ties, so this is a
performance regression check rather than a general bit-identity assertion.
Continuous-batch and C=8 server results from the same revision are recorded in
`bench/results/m5-pro-24gb/dense27b_mtp_20260907.json`; `bench/mtp.py`
reproduces the one-shot and local continuous-batch measurements.

## What each result means

### Native R8 simdgroup matrix

The prototype used one native 8-row fragment, loaded A directly from device
memory, partitioned decoded B storage by simdgroup, used simdgroup barriers,
and fused output Hadamard/rout/f16 writeback.  Single-projection error versus
the scalar kernel was at most 0.001953125 after f16 output.  It was sometimes
slightly faster for K3 `up_proj`, but `down_proj` was neutral or slower; the
complete target gained only 0.9%.

M4 cannot use a native four-row matrix fragment.  It would still compute eight
rows, so an M4 half-tile was not pursued after the exact M8 tile failed the
whole-target gate.

Target ABBA samples in milliseconds:

```text
scalar: 146.5995 147.9442 148.1532 146.3863 146.3147 146.1426
        146.6900 146.4425 146.6915 147.5520 146.5887 146.1415
R8:     146.0271 146.5791 145.0580 145.1033 145.8522 144.8204
        145.3618 145.1500 145.2638 145.3015 145.3869 145.2020
```

### Output-WHT row parallelism

The existing fused epilogue runs the seven H128 stages one row at a time.  The
prototype processed all R rows concurrently, reducing full-threadgroup
barriers from `14*R` to 14 while keeping every butterfly operation identical.
Projection output was bit-exact.  Complete-target performance was unchanged at
M4 and slightly worse at M8, proving those epilogue barriers are not the
limiter.

```text
M4 serial:   100.9823 101.1422 101.0420 100.9549 100.8902 100.3438
             100.6279 101.0714 100.6520 100.3253 100.6103 100.5978
M4 parallel: 101.5909 100.6944 100.7268 100.5874 100.6846 100.3405
             100.5218 100.8767 100.4645 100.2293 100.0357 100.6567
M8 serial:   147.2411 146.9076 146.5590 146.6109 147.1075 146.9516
             146.5423 146.9223 146.7439 147.8696 148.0703 146.9335
M8 parallel: 147.3960 147.5506 146.9998 146.7429 146.8641 147.0288
             146.9539 147.6056 147.7096 148.5661 147.8122 147.3440
```

### MLP fusion

The low-risk fusion combined the FP16 SwiGLU result with `down_proj`'s input
scale and H128 transform.  It measured the useful boundary of a larger
gate/up/SwiGLU kernel while reusing the existing coded down GEMM.  It saved an
intermediate tensor but did not reduce coded-stream reads or decoding, and was
a wash at both row counts.  A same-threadgroup gate/up kernel would double
accumulator pressure and still decode the same two streams, so it was stopped
at this gate rather than adding a larger negative-risk kernel.

The custom activation differed by at most 0.00390625 at final f16 MLP output;
that numerical cost is not justified by a 0.4--0.6% timing change.

### `down_proj` on-chip split-K

The prototype used a 512-thread group with 16 simdgroups: eight output shards
times two fixed K halves.  Both partials stayed in threadgroup memory, were
reduced in a fixed order, and then used the existing output transform.  It
created no global partial buffer and no reduction dispatch.  Results were
deterministic; maximum f16 difference versus the sequential accumulator was
0.001953125.

Across eight different layer weights it gained 0.9% at M4 and 3.4% at M8.
`down_proj` would need about 1.16x locally to move the full target by the 3%
retention threshold, so the target integration was not warranted.

### Code-stream prefetch and repacking

PF2 kept two independent code-word pairs in flight and preserved accumulation
order exactly.  It gained 3--7% on isolated M8 projections but only 1.2% on the
complete target.  PF4 regressed slightly, while PF8 hit a register/occupancy
cliff and made target verification 82% slower.

```text
PF1 vs PF2 target samples:
PF1: 146.7773 146.5055 147.0867 148.6519 146.5348 146.6247
     147.0617 146.8997 146.4813 148.1547 147.8389 146.7019
PF2: 145.3805 145.1296 144.8250 144.8893 145.9391 145.1440
     144.8702 144.6462 144.8525 146.0629 146.3814 145.7685
```

The checkpoint already stores each compressed tile contiguously as
`[TK, TN, 8*K]`.  A same-size permutation cannot remove the overlapping
bit-window decode.  Expanding 24.3 billion coded weights to f16 would require
about 48.6 GB just for the coded body, versus the 19 GB runtime working-set
budget.  Streaming an expanded projection per layer would add decode and
allocation work every token.  Consequently there is no memory-feasible offline
repack left to retain after the PF experiment.

### Fixed-shape graph capture

Wrapping `_target_tree_forward` in `mx.compile` failed with:

```text
RuntimeError: [eval] Attempting to eval an array without a primitive.
If you are compiling a function, make sure all the inputs and outputs are captured.
```

This target body mutates heterogeneous KV, GDN, and convolution caches.  MLX
compile requires a pure captured input/output graph; it is not CUDA graph-style
command-buffer capture.  Rewriting every cache as explicit functional inputs
would be a large alternative runtime, not a small optimization, and earlier
in-model compile experiments also found no throughput gain and unacceptable
logit drift.  The current runtime therefore keeps MLX lazy scheduling plus the
already measured projection-pair dispatch fusion.

## Revisit conditions

These paths should be reconsidered only when one of their premises changes:

- a Metal/MLX release exposes smaller matrix fragments or command-buffer graph
  capture;
- a wider Apple GPU makes the measured split-K or PF2 ratios materially larger;
- the checkpoint ships a matrix-friendly compressed layout without increasing
  its working set;
- profiling identifies a specific stall counter rather than another inferred
  kernel bottleneck.
