# Changelog

## Unreleased

- **Stock-MLX checkpoints run on the escha runtime** (`escha_mlx/native.py`):
  `load`, `escha-mlx-generate` and `escha-mlx-server` now dispatch on the
  checkpoint's `quant_method`, so an ordinary MLX-quantized `qwen3_5` /
  `qwen3_5_moe` export loads through mlx-lm with escha's storage-agnostic quirks
  installed on top (fp16 GDN state cache + allocation-free first state,
  last-position LM head, wired limit, server continuous batching and
  `ignore_eos`). No codec is involved: nothing on this path is bit-exact-gated
  and the 2-bit footprint does not apply (a 4-bit 35B-class MoE is ~19.5 GB
  resident). Unvalidated architectures are refused by name unless
  `ESCHA_MLX_NATIVE_ANY=1`. Verified on an M5 Max against a 4-bit
  `qwen3_5_moe` export: greedy output **token-identical** to stock
  `mlx_lm.load` with `ESCHA_MLX_GDN_STATE=fp32 ESCHA_MLX_LAST_LOGIT=0`;
  346 tok/s aggregate served at batch 16 / OSL 128. Negative result: at ISL
  2817 on that chip the last-position head is neutral within noise (it is worth
  ~7% of prefill on a base M4).
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
