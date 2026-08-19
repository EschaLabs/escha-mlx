# M4 Max / 64 GB — optimization baseline

First measurements of escha-mlx on an M4 Max. This machine was brought up to
re-tune the runtime's wide-GPU settings: the M4 10-core tuning in
`docs/BRINGUP_AND_PERF.md` predicted the decode GEMV is latency-bound below a
core-count threshold that this chip (40 GPU cores, 1.6 threadgroups/core at
bs1) crosses.

## Machine

| item | value |
|---|---|
| Model | `Mac16,6` |
| Chip | Apple M4 Max |
| GPU | 40 cores (per `mx.device_info`: `applegpu_g16s`) |
| Unified memory | 64 GB |
| Memory bandwidth | 546 GB/s advertised, **482 GB/s measured** (88%, `roofline.py`) |
| GPU working-set cap | 55.66 GB (no sysctl override) |
| macOS | 26.5 |
| Python | 3.12.8 (uv venv) |

## Software

| package | version |
|---|---|
| escha-mlx | 0.1.0 |
| mlx | 0.32.0 |
| mlx-lm | 0.31.3 |
| numpy | 2.5.1 |
| safetensors | 0.8.0 |
| transformers | 5.14.1 |

Checkpoint: `EschaLabs/Qwen3.6-35B-A3B-Escha-W2` @
`0641c2dc8be5c96741a1e2bdeda99e81e3b15062` (HF snapshot cache, 3 shards,
format `eschamoe` 2.0, fp16 residuals, `weight_int8`+`weight_scale` packs).

## Run conditions

- Desktop Mac, AC power, other applications closed.
- In-process harnesses, no wiring needed (peak 15.1 GB vs 55.66 GB cap).

## Optimization-branch result (2026-08-11)

All committed changes active (R=8 prefill policy, shared-expert gate+up
regroup, barrier-free had).  `p0_gates.py` all PASS, `pytest tests/` 181
pass / 1 skip.  `bench/baseline.py --phases ABCD`:

```
A. correctness battery  PASS (paris / 391 / haiku-thinking all generate)
B. bs1 decode + prefill
   ISL= 128  prefill 1007.3 tok/s   decode  67.37 tok/s   (14.84 ms/tok)
   ISL= 512  prefill 1182.6 tok/s   decode  66.92 tok/s   (14.94 ms/tok)
   ISL=2048  prefill 1155.8 tok/s   decode  69.77 tok/s   (14.33 ms/tok)
C. batched decode (ISL=128)  -- all rows identical, match B=1
   B= 1  70.37 tok/s   B=2 118.13   B=4 165.97   B=8 205.74   B=16 333.73
D. cache accounting (ISL=512): 43.4 MB (GDNStateCache x30 32.9, KVCache x10 10.5)
```

vs the pre-optimization baseline (commit `ec4e88d`): prefill ISL=512
1072.7 -> 1182.6 (+10.2%), ISL=2048 1050.1 -> 1155.8 (+10.1%); decode B=1
68.31 -> 70.37 (+3.0%) and B=8 202.5 -> 205.7; correctness and
rows-identical invariance preserved throughout.  See the matrix below for
the measured-negative levers.

## Baseline (2026-08-11, `bench/baseline.py --phases ABCD`)

```
A. correctness battery     PASS (all three anchors generate)
B. bs1 decode + prefill
   ISL=  128  prefill  443.5 tok/s   decode  59.43 tok/s   (16.83 ms/tok)  peak 12.71 GB
   ISL=  512  prefill 1072.7 tok/s   decode  67.57 tok/s   (14.80 ms/tok)  peak 13.07 GB
   ISL= 2048  prefill 1050.1 tok/s   decode  64.81 tok/s   (15.43 ms/tok)  peak 13.12 GB
C. batched decode (ISL=128)
   B=  1  aggregate  68.31 tok/s   step  14.64 ms   prefill  808.3 tok/s
   B=  2  aggregate 112.80 tok/s   step  17.73 ms   prefill  783.4 tok/s
   B=  4  aggregate 160.96 tok/s   step  24.85 ms   prefill 1151.4 tok/s
   B=  8  aggregate 202.54 tok/s   step  39.50 ms   prefill 1334.6 tok/s
   B= 16  aggregate 341.77 tok/s   step  46.82 ms   prefill 1264.6 tok/s
D. cache accounting (ISL=512): 43.4 MB total — GDNStateCache x30 32.9 MB,
   KVCache x10 10.5 MB
```

`roofline.py` (same session): ceiling **201.9 tok/s bs1 at 482 GB/s**; measured
decode step 14.47 ms (69.11 tok/s) = 34% of ceiling; trellis GEMV at bs1 row
counts only 6-12% of roofline (28 GB/s gate_up rows8) while the Q8 dense stream
(2.09 GB/token) and trellis stream (0.30 GB/token) together sweep 2.386
GB/token.

`prefill_profile.py`: MoE expert path is 55-60% of prefill wall time at
chunk 128-1024 (e.g. S=256: 146.1 ms of 254.4 ms); lm_head is negligible
(-2 to -9% incl. measurement noise; last-logit head shipped).

## Per-machine deltas being re-measured here

All of these were measured a wash or regression on the 10-core M4 and each
share a "starvation grows with GPU width" premise that this chip falsifies:

- `ESCHA_MLX_SPLITK=auto` — 64 threadgroups/step at bs1 is 1.6/core here vs
  6.4/core on the M4.
- `ESCHA_MLX_FETCH=shuffle` — direct-GEMV cooperative broadcast.
- `ESCHA_MLX_PREFETCH=1` — row-blocked GEMM code prefetch.
- `ESCHA_MLX_SORTX=1` — pre-sorted x staging.

## Measured on the M4 Max (drift-controlled where possible, 2026-08-11)

Shipped-policy flags stay.  Every entry below was measured in-model with the
repo's harnesses; isolated-kernel wins that did not transfer are recorded
separately because this box keeps showing the same wall:

| lever | result | note |
|---|---|---|
| `ESCHA_MLX_SPLITK=auto` | wash to -3.1% (B=1) | regression at B=16; the chip still absorbs 64 TGs |
| `ESCHA_MLX_SPLITK=4` (pinned) | wash at B=1 (68.2 vs 69.3) | the wide-GPU starvation premise fails even pinned on 40 cores |
| `ESCHA_MLX_FETCH=shuffle` | -6.2% (B=1), -11.3% (B=8) | load slots were not the stall |
| `ESCHA_MLX_SORTX=1` | wash (delta == drift) | B=8..32 |
| `ESCHA_MLX_PREFETCH=1` (GEMM) | wash-at-noise up to B=8 | isolated KB=16+pf -11% did NOT transfer (in-model -9%) |
| `ESCHA_MLX_KT_BLOCK=16` + prefetch | -9% (S=256 chunk) | registers vs the overlapping pipeline |
| `ESCHA_MLX_GEMV_KB=4/8` (new) | wash at B<=2, noise at B=4/8 | isolated 1.85x GB/s on the rows8 kernel, zero in-model (overlap) |
| `ESCHA_MLX_LUT=1` | **-25 to -46% decode, -42% prefill** | the hash/ALU path is right on this GPU; LUT's dependent load is a regression |
| `ESCHA_MLX_GEMV_HAD` (new, had_in+gemv) | wash at B<=2 | isolated 1.85x on highest 1.85x, +85% rows8 GB/s, zero in-model |
| `ESCHA_MLX_GEMV_OUT` (new, gemv+had_out epilogue) | **-6 to -11% decode** | in-TG butterfly sync outweighs the saved launch on this GPU |
| `build_groups` ablation | isolated 0.32 ms x80 = 26-38 ms/chunk, but a stub is SLOWER in-model | grouping ops overlap; not independently recoverable |
| funnel-shift (32-bit) | neutral in-model | bit-identical ALU trim on the hot extract |
| `ESCHA_MLX_HAD_KB=8` (**new: prefetch code tiles in the fused decode GEMV**) | **decode +3-4%** | code stream ~55% of the gu leg (constant-folding drops it to 22us); prefetch only ever existed on the non-fused moe_gemv, not the moe_gemv_had the decode path uses. Isolated gu 49->31us, dn 18-20us. KB=16 no better in-model. |
| `ESCHA_MLX_HAD_PACK` (**new: 8 output-had blocks per 256-thread TG**) | **~+1% decode (noise), prefill flat** | decode had kernels launched 32-thread TGs (~2560/layer at bs1); butterfly is simdgroup-local so 8 pack one TG. Isolated 10.3->5.3us / 8.0->5.2us; in-model the kernels overlap the gemv chain so the win is small. |
| codic-ALU reduction (drop cba_decode body) | **no change** | decode step 15.6 vs 15.6ms clean — the per-state multiply/mask ALU is hidden under memory/dispatch latency. The MoE block is NOT ALU-bound. |

**Banked wins:**

- `_PREFILL_R["Apple M4 Max"] = 8` (was 12): prefill S=256 +4.7%, S=512 +7.2%
  in-model; end-to-end ISL=2048 prefill 1050 -> 1135 tok/s.  R=8 == m/E at
  m=2048: one full group per expert, zero padding (256 vs 405 groups).
- shared-expert gate+up regroup (one affine-Q8 call per layer): -40 launches
  and one [512, K] stream read per decode step; decode +0.9/+1.9/+0.6/+8% at
  B=1/2/4/8 under thermals.
- 256-thread scaled-had threadgroups (`ESCHA_MLX_HAD_TG`): prefill issues
  32768 32-thread threadgroups per [m,IC]=[2048,2048] transform; packing 8
  (128-block) simdgroups into one 256-thread threadgroup cuts launch count 8x
  with the identical per-block butterfly (partial-last-group guarded out,
  bit-identical, verified across many shapes).  Isolated 1.7x at m=2048,
  2.5x at m=4096; in-model prefill 1123 -> 1158 tok/s (+2.5-3%), eval metric
  1.178 -> 1.207 (win at both ends of an A/B/A bracket).

Decode bs1 remains structure-bound: the whole MoE block is ~49% of the step
(B=1: 14.14 -> 7.23 ms with the block stubbed), the routed-expert path ~29%,
and no kernel-level knob moves it because the 80 trellis GEMVs overlap the
dense stream in-model (isolated 6-12% of roofline at rows8 vs 46-50% for
stock affine gather on the same shape -- omlx#2238 measured the same gap on
an M3 Ultra and reached the same conclusion: at M=1 the missing margin is the
single-token structure, not a better matvec).
## wave3/cache lane — Python graph-build + per-layer fixed ops (2026-08-12)

The decode metric builds all 32 decode graphs BEFORE the single trailing
`mx.eval` (eval_metric.py), so MLX does not overlap Python graph construction
with GPU execution: measured decode time = 32 x (python_build + gpu_step).
Python graph-build for one decode step is ~3.2 ms on this box (pure-CPU,
contention-immune), i.e. roughly 20-25% of the ~14.5 ms step, and it is 1:1
additive on the metric.  MoE blocks are ~57% of the build (~37-42 us/layer),
GDN+attention ~38%, norms ~2%.  The dominant line item inside a MoE layer is
the 4 trellis metal_kernel calls (~5.5 us Python each, irreducible from the
caller's side) plus ~25 small mx ops.

Shipped (bit-identical, logit hash unchanged): cache the per-step constant
row/arange tensors (`_arange32`, `_row_token`) that every MoE layer rebuilt
every step — reusing one immutable array drops ~90 arange/repeat/reshape
nodes from the 32-step lazy graph and equally many Python calls.  Small, but
strictly less work on the additive path; whole-step delta is inside noise.

Measured dead ends (bit-identity hard requirements kill all of these):

| lever | result | note |
|---|---|---|
| silu-tail fusion (output-Had + silu in one kernel) | **NOT bit-identical** | MLX compiles its elementwise Sigmoid with fast-math: its GPU sigmoid differs from the stable 1/(1+exp(\|x\|)) on ~39% of f32 inputs on this build, and a mx.fast.metal_kernel cannot reproduce it.  One sigmoid ulp -> s16/h -> dn GEMV -> logits.  Kernel kept gated (ESCHA_MLX_SILU_TAIL=1) for a build where MLX's sigmoid is bit-exact. |
| k+v projection stacking (one Q8 call, GQA-safe in theory) | **NOT bit-identical** | Standalone [1,2048]x[2048,512] vs stacked [1,2048]x[2048,1024] matched on every sampled layer/input, but the REAL route diverged (n=1024 vs n=512 picks a different MLX matmul tiling -> different K-reduction -> occasional rounding flip).  Same wall as FUSE_ATTN: output-axis stacking is not reliably bit-identical on this MLX build, even under GQA where FUSE_ATTN is documented invalid. |
| k/v -> one call (in-proj fusion for the GDN, dense lane) | neutral (already single in_qkv) | zero bytes saved; bandwidth-bound |
| routers / topk / softmax (precise) | no lever | `precise=True` softmax + fp16 gate are the bit-identity contract; cannot quantize/fuse the gate |
| attention SDPA + KV cache | no lever | KV read 0.33 GB/token is real but its dtype is the bit-identity contract (no fp8), layout is mlx-internal |
| lm_head (0.51 GB/token, ~1.1 ms) | no lever | int8 stream read once/token at ~97% of roofline; any dtype change breaks logits |

Decode stays structure-bound and decode-MoE-heavy at this HEAD (stub=mlp
removes ~8 ms of a 15.5 ms step); the MoE trellis and dense/Q8 streams are the
two lanes that can move the metric, per the README head note.

## PR opt/decode-prefill — paired A/B (2026-08-19)

`bench/baseline.py --phases BC` run back-to-back from `main` then
`opt/decode-prefill` in the same session.  The PR adds `mx.async_eval` per
forward (overlaps GPU with the next step's Python graph-build, ~3.2 ms/step)
and `folded_generate_step` (folds the final prompt token into prefill, saving
~16 ms TTFT per request).

`baseline.json` = main (pre-PR); `baseline_opt.json` = opt/decode-prefill.

**bs1 ISL=512 (the clean comparison — B≥8 on main ran under thermal stress):**

| | main (`baseline.json`) | opt/decode-prefill (`baseline_opt.json`) | delta |
|---|---|---|---|
| decode tok/s | 67.87 | 88.75 | **+30.8%** |
| prefill tok/s | 1076.9 | 1201.2 | **+11.5%** |
| step ms | 14.73 | 11.27 | −23.5% |

```
baseline.json (main):
B. ISL=128  prefill  991.3 tok/s   decode  66.17 tok/s  (15.11 ms/tok)
   ISL=512  prefill 1076.9 tok/s   decode  67.87 tok/s  (14.73 ms/tok)
   ISL=2048 prefill 1048.6 tok/s   decode  68.06 tok/s  (14.69 ms/tok)
C. B=1  69.68 tok/s   B=2 118.18   B=4 159.70
   (B≥8 measured under thermal stress; see note below)

baseline_opt.json (opt/decode-prefill):
B. ISL=128  prefill 1007.7 tok/s   decode  85.62 tok/s  (11.68 ms/tok)
   ISL=512  prefill 1201.2 tok/s   decode  88.75 tok/s  (11.27 ms/tok)
   ISL=2048 prefill 1177.4 tok/s   decode  88.36 tok/s  (11.32 ms/tok)
C. B=1  89.57 tok/s   B=2 151.35   B=4 201.99   B=8 240.92   B=16 271.67
   (all rows_identical=true, matches_b1=true)
```

**Thermal note:** the main B phase ran immediately after a full prior session;
B=8 (94.71) and B=16 (74.85) in `baseline.json` are visibly suppressed
(step 84 ms / 213 ms vs the expected ~35 ms / 47 ms from Aug-11 data above).
B=1–4 are unaffected. The opt branch ran after a 60 s cool-down and is clean
across all batch sizes.

Roofline (`roofline.json`): measured peak 493.8 GB/s (90% of 546 GB/s
advertised), ceiling 207.0 tok/s bs1.  Measured step B=1: 13.77 ms / 72.64
tok/s (35% of roofline).

Correctness: 201 passed, 1 skipped.  `p0_gates.py`: ALL GATES PASS.

## Files

| file | contents |
|---|---|
| `baseline.json` | `main` in-process baseline (pre-PR, phases BC, 2026-08-19) |
| `baseline_opt.json` | `opt/decode-prefill` result (phases BC, 2026-08-19) |
| `roofline.json` | peak bandwidth, Q8 dense, trellis GEMV, byte ledger, measured step |
| `p0_gates.log` | gate output — all PASS on macOS 26.5, mlx 0.32.0, 40-core M4 Max GPU |

`p0_gates.py` does not emit JSON; its output is captured as a log.
