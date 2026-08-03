# Changelog

## 0.1.0 — 2026-08-03

First public release.

- Runtime for `EschaLabs/Qwen3.6-35B-A3B-Escha-W2` (2-bit trellis experts + int8 dense),
  consuming the Hugging Face export directly — no conversion step.
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
