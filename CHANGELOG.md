# Changelog

## Unreleased

- **Dense architecture support (`qwen3_5`)** — serves
  [`EschaLabs/Qwen3.8-27B-Escha-W2`](https://huggingface.co/EschaLabs/Qwen3.8-27B-Escha-W2),
  a 27B dense hybrid (GDN + attention) whose 400 projections are all trellis-coded at a
  **per-tensor rate** (`mlp.{up,down}_proj` at K=3, the rest at K=2 — 2.469 bits/weight).
  New `escha_mlx/dense.py` (single-stream weight container + `EschaLinear`) and
  `escha_mlx/models/qwen3_5.py` (plugin); the skeleton is `mlx_lm.models.qwen3_5`
  untouched, so no architecture code is carried here.
  - **Dense Metal kernels** as a compile-time variant of the expert kernels rather than
    a second implementation: the GEMV (direct / staged / split-K) and both fused
    Hadamard transforms drop the `row_expert` indirection and the per-row transform-vector
    gather. Every MoE kernel source is byte-identical to before, and the dense variants
    are gated bit-identical against the expert kernels on a one-expert stream
    (`tests/test_dense_linear.py`).
  - **End-to-end scales folded at load**: dense exports ship `escha_s_in`/`escha_s_out`
    alongside `escha_rin`/`escha_rout`. They multiply at the same points, so
    `ref.fold_scales` folds them in f32 at load time — no new kernel, no new tensor, and
    one rounding point fewer than applying the scales separately (documented deviation, gated
    against real shipped tensors). Dropping them would be a ~2% silent error; there is a
    test that fails if they are.
  - **Row-blocked dense GEMM** — the dense counterpart of the expert row-blocking, and
    what makes dense prefill viable: the per-row GEMV reads the whole coded stream once
    per row, so a 256-token chunk would decode every projection 256 times. A dense group
    needs no grouping machinery (rows are consecutive, one stream, only the last group
    partly padding), so it carries neither `rows_idx` nor `group_expert`. Bit-identical
    to the per-row kernel at every R. `ESCHA_MLX_DENSE_BLOCK_R` pins R; the default
    thresholds are structural, **not measured on Metal** — the first thing to sweep on
    hardware (flagged as such in docs/INSTALL.md).
  - **Per-linear bias**: the additive correction the end-to-end stage leaves behind is
    applied in f32 after the output transform.
  - **Load-time metadata cross-check**: `escha_config`'s K and shapes are checked against
    the code stream's own shape, and an unknown codebook id is refused — a mismatch would
    otherwise decode into plausible, finite, entirely wrong weights.
  - **Load-time shape guard**: both dimensions of a coded linear must be multiples of
    128. The kernels assume it and do not check — a 16-aligned but not 128-aligned
    dimension would silently drop the remainder of the last block.
  - `escha-mlx-generate --reasoning-effort` passes an effort through to chat templates
    that take one (Qwen3.8: low/medium/xhigh, default xhigh); omitted unless given, so
    every model keeps its own default.
  - New gates: real-data goldens under `tests/data/qwen3_5/` (a 128×128 corner of two
    shipped linears with their reference outputs), a dense synthetic
    end-to-end checkpoint test, and `tests/test_dense_checkpoint.py`, which validates a
    real export's structure from safetensors headers alone (leaf completeness, declared
    vs implied rate, kernel shape preconditions).
  - **Load-time guards** (each verified to be a real silent-failure path before being
    added): K must be in {2, 3} — anything else was decoded by the K=3 unpacker into
    plausible Gaussian weights; `escha_config` must have exactly its 6 fields — a
    shorter header skipped the codebook gate entirely; `escha_s_in`/`escha_s_out` must
    be 1-D of the right length — the fold broadcasts, so a 2-D scale silently applied
    row 0 to every channel and a scalar applied one value everywhere; an unrecognised
    `escha_*` tensor is a clear version error rather than a parameter-name crash inside
    `model.update`. The int8-pair and conv1d-shift checks became raises, since `python
    -O` strips asserts and both failures are silent.
  - `ESCHA_MLX_DENSE_BLOCK_R` is now latched at module construction like every other
    knob, instead of being read on each forward.
  - `gdn_cache.install` reads the GDN geometry off the model instead of assuming the
    MoE's 32 v-heads; the dense 27B saving is 75.5 MB/seq, not the 50.3 it was printing.
  - docs/INSTALL.md gains a dense memory section: 64 KiB/token of KV (3.2× the MoE),
    the real GDN state figure, and the fact that `--max-kv-size` is inert on any hybrid
    GDN model because the architecture defines `make_cache` (true of stock mlx-lm too).
  - **Byte ledger fixed for dense models** (`bench/roofline.py`). `mod_bytes` walked
    `parameters()`, but a coded linear keeps its stream off the parameter tree on
    purpose — so it counted a dense linear at its bias, literally zero when there is
    none — and `layer.mlp` was only read inside the MoE branch, so the dense MLP, the
    majority of the bytes, was invisible. A dense model ledgered at ~0.05 GB instead of
    10.15. Adds `dense.coded_bytes`, an `mlp_dense_MB` term, and a guard that raises if
    the ledger ever again misses coded streams. Verified on a truncated real load:
    0.471 GB counted against 0.470 GB on disk.
  - **Rows-per-group policy replaced.** The first version was a size ladder inherited
    from the MoE's thresholds, which returned R=1 below 4 rows — so a server at
    concurrency 2 or 3 read the whole coded stream once per row, the exact cost the
    row-blocked kernel exists to remove. Now `R = min(rows, 8)` snapped to a power of
    two: never pads, R>1 from two rows, four compiled variants.
  - **The per-linear correction is OFF by default** (`ESCHA_MLX_BIAS=1` applies it).
    A dense export ships a `bias` beside every coded linear; on Qwen3.8-27B all 400 are
    non-zero and applying one moves that linear's output by 6.7-8.3%. The path
    those published numbers come from does not apply them — that method registers no
    bias parameter, every coded linear is built `bias=False`, and unmatched `.bias`
    names are dropped through a GPTQ-era guard without warning — so every published number
    for this checkpoint was produced without it. Defaulting off keeps this runtime
    comparable to those numbers; the loader logs that the tensors are present and
    unused. Which behaviour is correct is a question about the checkpoint, not about
    this runtime.
  - **Not yet run on Metal.** The dense path was developed on Linux with `mlx[cpu]`,
    where the Metal kernels do not execute. The reference, the module, the loader and
    the real-checkpoint structure are all gated and green; the dense kernel sources
    themselves meet the Metal compiler for the first time at RUNBOOK step 1e.
- **Fused expert output Hadamard transform** (companion to 0.1.0's input-side
  fusion): transform + output-scale gather + f16 cast in one Metal kernel,
  bit-exact with the native chain. M5 Pro: +5.6% decode at B=1, +5.8% prefill
  (same-process A/B/A). M4 base: ~+10% prefill, decode unchanged within noise.
- **Native Hadamard transform** in the fallback op chain (`mx.hadamard_transform`
  instead of dense matmul), which also resolved the M5 Pro TF32 test-oracle
  issue; the fused/native gate is now bit-exact under default TF32.
- **Allocation-free GDN first state**: the first recurrence starts from zeros in
  Metal registers and emits the storage dtype directly — first-forward peak
  memory −2.08 GB at B=64 on 24 GB (18.20 → 16.12 GB), logits and state
  bit-identical.
- Benchmarks: refreshed M5 Pro results; first M4 Pro (48 GB) characterization;
  M4 base paired A/B/A for both changes plus a GDN first-state memory probe
  (`bench/sweep_gdn_first_state.py`).
- Fixes: `bench/baseline.py` reported GiB while printing "GB" (all figures
  corrected or relabeled across the docs); `bench/roofline.py` repaired after
  the models/ split; broken line-continuation in INSTALL's serving example;
  a pre-release consistency audit across README/docs/tables.

## 0.1.0 — 2026-08-03

First public release.

- Runtime for `EschaLabs/Qwen3.6-35B-A3B-Escha-W2` (2-bit trellis experts + int8 dense),
  consuming the Hugging Face export directly — no conversion step.
- Architecture plugin registry (`escha_mlx/models/`): the codec engine is shared, each
  supported `model_type` is one plugin module (skeleton, tensor map, router, quirks);
  a synthetic mini-checkpoint test runs the full load-and-forward path per plugin in CI.
- Metal kernels via `mx.fast.metal_kernel`: trellis decode (hash + LUT), staged and
  direct expert GEMV, row-blocked GEMM, fused gather+scale+transform+cast — every path
  gated bit-identical to the committed reference goldens (`np.array_equal`, not a tolerance).
- Exact int8 → MLX affine-Q8 repack for dense/embed/head, validated at load time.
- CLI generation (`escha-mlx-generate`) and an OpenAI-compatible server
  (`escha-mlx-server`) with continuous batching and prefix caching (mlx-lm scheduler).
- Hardware-validated on Apple M4 24 GB: 27.0 tok/s single-stream decode (69% of the
  bandwidth ceiling), 185.6 tok/s in-process @ batch 128, 99.5 tok/s peak served output.
- Full bring-up + perf campaign record in `docs/BRINGUP_AND_PERF.md`, including negative
  results.
