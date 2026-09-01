# escha-mlx

[![CI](https://github.com/EschaLabs/escha-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/EschaLabs/escha-mlx/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10–3.13-blue.svg)](pyproject.toml)
[![Models](https://img.shields.io/badge/🤗%20models-EschaLabs-yellow.svg)](https://huggingface.co/EschaLabs)

**2-bit LLMs on Apple Silicon.** escha-mlx is the [MLX](https://github.com/ml-explore/mlx)
runtime for [escha](https://huggingface.co/EschaLabs) trellis-quantized models: it loads
the published Hugging Face exports directly — no conversion step — and serves them through
custom Metal kernels that are **bit-exact against the codec's committed reference vectors**.

- **35B-class MoE in 12.3 GB** — generates and serves on a stock 24 GB Mac
- **OpenAI-compatible server** with continuous batching and prefix caching
- **Provable correctness**: every kernel path gated `np.array_equal` against committed
  goldens — not a tolerance. CI runs the suite on every PR (reference/repack gates on
  Linux CPU; the Metal kernel gates run when the runner exposes Metal and self-skip
  otherwise — the workflow's "Metal available?" step records which happened)
- **Honest benchmarks**: measured on hardware, drift-controlled, negative results included

> **This is a reference implementation.** The kernels and defaults are correct on any
> Apple silicon but are tuned only on the hardware we could measure — a base M4, an
> M4 Pro and an M5 Pro. Apple GPUs differ enough (bandwidth, core
> count, Dynamic Caching) that the optimal settings are machine-specific, and every
> tuning knob here is gated bit-exact or has its numeric effect measured and
> documented, so tuning for *your* chip is a measurement exercise, not an
> unquantified numerics risk. **Community contributions are welcome**, per-machine
> tuning and benchmark PRs especially: method and knobs in
> [docs/BRINGUP_AND_PERF.md](docs/BRINGUP_AND_PERF.md) and
> [docs/INSTALL.md](docs/INSTALL.md), per-machine result format in
> [`bench/results/`](bench/results/), process in [CONTRIBUTING.md](CONTRIBUTING.md).

## Supported models

| model | quant | resident | status |
|---|---|---|---|
| [`EschaLabs/Qwen3.6-35B-A3B-Escha-W2`](https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2) | 2-bit experts + int8 dense | 12.3 GB | ✅ supported |
| [`EschaLabs/Qwen3.8-27B-Escha-W2`](https://huggingface.co/EschaLabs/Qwen3.8-27B-Escha-W2) | mixed-rate 2/3-bit dense + int8 embed/head | 10.2 GB | ✅ supported |
| Kimi K3 / GLM / DeepSeek MoE | — | — | 🔭 exploring |

The two are different shapes of the same codec. The 35B is a mixture of experts: only
its expert stacks are coded, and one fused kernel walks them through a router. The 27B
is **dense** — all 400 projections are coded, at a per-tensor rate (`mlp.{up,down}_proj`
at 3 bits, everything else at 2, 2.469 bits/weight overall), and each is an ordinary
matmul. Same decode, same goldens; the Metal kernels take the single-stream case as a
compile-time variant with the expert indirection removed.

Want an architecture that isn't listed? Open a
[model request](https://github.com/EschaLabs/escha-mlx/issues/new?template=new_model.yml).

## Install

Requirements: Apple Silicon (M1–M5), macOS 14+ (15+ recommended), Python 3.10–3.13,
~13 GB disk for the checkpoint. Full details and memory guidance:
[docs/INSTALL.md](docs/INSTALL.md).

```bash
git clone https://github.com/EschaLabs/escha-mlx && cd escha-mlx
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[test]' 'mlx==0.32.*' 'mlx-lm==0.31.3'   # quotes required: zsh globs [test]

hf download EschaLabs/Qwen3.6-35B-A3B-Escha-W2 --local-dir ~/models/escha-w2
```

The `mlx`/`mlx-lm` pins are the hardware-validated combination; newer versions usually
work, and the test gates below tell you for sure.

## Test

The NumPy reference (`escha_mlx/ref.py`) is the semantic contract, and the golden vectors
committed under `tests/data/` pin the codec's exact numerics — every fp16 rounding point
included. Every Metal kernel path — staged GEMV, direct GEMV, row-blocked GEMM, fused
transform, hash and LUT decode — is gated **bit-identical** to them and to each other:

```bash
pytest tests/ -v              # full suite — tests self-skip by capability
python bench/p0_gates.py      # kernel gates + measured DRAM-side GB/s
```

`pytest tests/` is safe anywhere: without Metal the kernel gates skip (the reference and
repack tests still run, even on Linux with `mlx[cpu]`), and without a downloaded
checkpoint the end-to-end tests skip. On a Mac with the model present, everything runs.
First run on a new machine? Follow [RUNBOOK.md](RUNBOOK.md) — gates first, then
generation, then serving.

## Serve

```bash
# one-shot generation
escha-mlx-generate --model ~/models/escha-w2 --prompt "Explain unified memory in one paragraph."

# OpenAI-compatible server (continuous batching + prefix caching)
escha-mlx-server --model ~/models/escha-w2 --port 8080 --prefill-step-size 256
```

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is 17*23?"}],
  "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": false}}'
```

On a 24 GB machine `--prefill-step-size 256` is not optional (the mlx-lm default peaks
past the ~19 GB GPU working-set cap), and close memory-hungry apps for the first session.
Above a ~18 GB working set, set `ESCHA_MLX_WIRED_GB` — unwired, throughput silently
collapses 23× near the cap. Memory settings and the full tuning-knob reference:
[docs/INSTALL.md](docs/INSTALL.md).

## Benchmarks

Measurements on the machines below, all using MLX 0.32.0 and mlx-lm 0.31.3.
The M4 is the entry-level 10-core-GPU model (24 GB, macOS 26.5.2); the M4 Pro is a
20-core-GPU, 48 GB machine (macOS 15.7.3); the M5 Pro has a 16-core
GPU (24 GB, macOS 26.5.2).

### Qwen3.6-35B-A3B MoE

Every M5 Pro result in this summary table used runtime revision
`bf86c10d4d91e5d4aaa7d4046983723e139f47cc`, model revision
`1b7237f0886a10b4bd92cd7653090cd7381ae199`, AC power and High Power mode.

| workload | M4 base, escha W2 | M4 base, stock MLX 4-bit | M4 Pro 48 GB, escha W2 | M5 Pro, escha W2 |
|---|---:|---:|---:|---:|
| resident memory¹ | **12.25 GB** | 19.51 GB | 12.25 GB | **12.25 GB** |
| prefill, ISL 512 | 264 tok/s | **344.2 tok/s** | 263.8 tok/s | **756.5 tok/s** |
| decode, single stream | 27.3 tok/s | **42.9 tok/s** | 41.3 tok/s | **59.68 tok/s** |
| aggregate @ batch 8 | 59.6 tok/s | **101.5 tok/s** | 104.7 tok/s | **193.03 tok/s** |
| aggregate @ batch 16 | **104.0 tok/s** | out of memory | 179.6 tok/s | **239.96 tok/s** |
| aggregate @ batch 128 | **185.6 tok/s** (18.1 GB peak) | out of memory | — | **539.25 tok/s** (18.06 GB peak) |
| served peak output / total | 99.5 / 244.5 tok/s | — | 105.1 / 296.7 tok/s | **206.00 / 589.08 tok/s** |

¹ Identical on every machine, as it must be — same checkpoint, same Q8 group size.
Earlier tables showed 11.41 for the Pro machines: that was the same residency in
GiB (12.25 × 10⁹ bytes = 11.41 GiB) from a harness that divided by 1024³ while
printing "GB" — fixed 2026-08-09. Decimal GB everywhere in this table.

On the base M4, prefill runs ~264 tok/s; single-stream decode reaches ~66% of that
chip's ~41.2 tok/s bandwidth ceiling (2.45 GB/token at the current Q8-128 ledger,
101 GB/s measured). Read the comparison honestly in both directions: below batch 16 the
4-bit build is faster; from batch 16 up **only escha runs at all** on 24 GB — that is the
regime the 1.59× footprint buys, and where a Mac serving more than one user lives.

The M4 Pro column was measured on a second team machine (the first Pro-class
datapoint, [PR #2](https://github.com/EschaLabs/escha-mlx/pull/2)) at
runtime **v0.1.0 — before** the native-Hadamard and fused-output-transform changes
the M5 Pro column includes (the M4 column also predates the fused output transform;
see below). Its in-process rows are the opening baseline of a session whose own
drift controls put ~10% on single observations; only its served 128:128 point
carries a five-trial error bar. B=128 was not run there. Its most interesting
reading: 41.3 tok/s single-stream is ~**42% of that chip's ~99.5 tok/s roofline**
(243.7 GB/s ÷ 2.45 GB/token; the archived report's 43.6%-of-94.8 used the ledger
accounting current at measurement time) vs ~66% on the base M4 — more bandwidth,
lower utilization, the latency-bound signature that motivates per-machine tuning.
Full write-up, drift controls, and harness issues found:
[`bench/results/m4-pro-48gb/README.md`](bench/results/m4-pro-48gb/README.md).

The M5 Pro results are a separate machine characterization, not a paired cross-chip
A/B. Its B=1/8/16/128 entries are five-run medians from the 16-token,
step-synchronized repeatability harness; B=128 alone uses a 19 GB wired limit.
The current P0 gates pass and the complete suite reports **259 passed, 9 skipped**
(the skips are the opt-in real-checkpoint and slow gates).
A same-process native-vs-output-fused A/B/A through B=32 is reported in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md#current-output-hadamard-fusion-aba).

The M4 column predates the fused expert **output** transform. Re-measured on that
chip as a paired A/B/A, the kernel raises prefill ~10% and leaves decode unchanged
within noise — the opposite split from the M5 Pro — while the allocation-free GDN
first state cuts first-forward peak memory 2.08 GB at batch 64 (18.20 → 16.12 GB
against a 19.07 GB cap) with bit-identical logits and state:
[docs/PERFORMANCE.md](docs/PERFORMANCE.md#apple-m4-base-24-gb--output-hadamard-fusion-and-gdn-first-state).

### Qwen3.8-27B dense

The M5 Pro publish run used runtime revision
`b373dc353e8190965d0ec47b1d77cd6ae3336da5` and model revision
`f0eadefa2f9679f7c04a115214c1cd883979a529`, with repository defaults, a
256-token prefill chunk and no memory or wired-limit override. The full suite,
including the slow real-checkpoint dense gates, and `bench/p0_gates.py` passed
before measurement. Each M5 baseline row is the median of three fresh processes;
the M4 rows are the previously committed hardware run and are not a paired
cross-chip A/B.

| workload | M4 base, escha W2 | M5 Pro, escha W2 |
|---|---:|---:|
| prefill, ISL 128 | 39.8 tok/s | 80.6 tok/s |
| prefill, ISL 512 | 39.8 tok/s | 82.0 tok/s |
| prefill, ISL 2048 | 37.7 tok/s | 81.8 tok/s |
| decode, single stream, ISL 512 | 7.18 tok/s | 14.55 tok/s |
| aggregate @ batch 4 | 18.96 tok/s | 34.75 tok/s |
| aggregate @ batch 16 | 30.79 tok/s | 60.68 tok/s |

The shipped, bit-exact scalar prefill path reaches ~82 tok/s on the M5 Pro. The
opt-in simdgroup-matrix path reaches 103.64 tok/s at chunk 256 (+28.3% against
the same-process scalar mean), but remains off by default because it reassociates
the f32 reduction and is not bit-identical. Dense single-stream decode reaches
about 47% of its nominal memory roofline on this chip; together with the M4
diagnostics, that points to instruction issue rather than DRAM bandwidth as the
operative limit. Method, repeatability ranges, strategy controls, cache accounting
and raw artifacts are in the
[full M5 Pro 27B section](docs/PERFORMANCE.md#apple-m5-pro-24-gb--qwen38-27b-dense-w2).

Full tables (ISL/OSL serving grid, prefill scaling, drift controls):
[docs/PERFORMANCE.md](docs/PERFORMANCE.md). The complete bring-up and optimization
campaign — including every negative result, so you don't repeat them:
[docs/BRINGUP_AND_PERF.md](docs/BRINGUP_AND_PERF.md). Raw result JSONs per machine:
[`bench/results/`](bench/results/).

## Resolved issues

- **M5 Pro, MLX 0.32.0, fused-Hadamard comparison:** the old test compared an
  explicit FP32 butterfly with a dense matmul that selected MLX's TF32/NAX path.
  It therefore measured two reduction algorithms and precisions, not the production
  fused/native equivalence. `moe.had_blocks` now uses `mx.hadamard_transform`, and
  the fused test requires its final FP16 output to be bit exact with that native
  butterfly under the default TF32 setting. The independent NumPy/reference check
  remains as a tolerance test. Global `MLX_ENABLE_TF32=0` is no longer needed for
  this gate; the complete suite now reports **259 passed, 9 skipped** with
  default TF32 and with `MLX_ENABLE_TF32=0`. Historical counts and the diagnostic
  TF32 A/B are retained in
  [docs/PERFORMANCE.md](docs/PERFORMANCE.md#m5-pro-resolved-issue-dense-matmul-test-oracle-and-tf32).

## Repository layout

| path | what |
|---|---|
| `escha_mlx/ref.py` | NumPy bit-exact codec reference — **the semantic contract** |
| `escha_mlx/msl.py` | the Metal kernels (`mx.fast.metal_kernel`): decode, GEMV ×2, row-blocked GEMM, fused transform |
| `escha_mlx/quant.py` / `moe.py` / `dense.py` / `loader.py` | int8→Q8 repack · expert toolkit · dense-linear toolkit · streaming loader — all architecture-agnostic |
| `escha_mlx/models/` | one plugin per architecture (`qwen3_5_moe`, `qwen3_5`): skeleton, tensor map, router, quirks |
| `escha_mlx/gdn_cache.py` | recurrent-state cache (fp16 state) + allocation-free first-state Metal kernel |
| `escha_mlx/{generate,server}.py` | CLI / OpenAI-compatible server |
| `tests/` + `tests/data/` | golden-gated suite + the committed reference vectors |
| `bench/` | gates, roofline, serving grid, head-to-head; results per machine under `bench/results/` |
| `docs/` | install & tuning · performance · full campaign record |

Everything outside the Metal kernels is pure MLX; attention and the KV caches come
from mlx-lm untouched. The one exception is the GDN recurrent state: by default it is
stored in escha-mlx's fp16 cache and initialized by an allocation-free first-state
kernel (`escha_mlx/gdn_cache.py`, measured in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md)); subsequent steps use mlx-lm's kernel
unchanged, and `ESCHA_MLX_GDN_STATE=fp32` restores the stock mlx-lm path exactly.

## Contributing

Contributions are welcome — kernels, new model architectures, serving features, and
especially broader perf campaigns on Pro/Max/Ultra GPUs; the M5 Pro results above are an
initial machine characterization. The short version of [CONTRIBUTING.md](CONTRIBUTING.md):

- **Correctness bar**: kernel changes stay bit-identical to the goldens, or the deviation
  is measured, documented, and behind a flag with the old behavior recoverable.
- **Measurement bar**: paired same-session A/B with a drift control; report chip, RAM,
  macOS and mlx versions. Negative results are contributions.
- **New chips**: one tree, no per-chip branches — strategies are flags, results are
  folders (`bench/results/<machine>/`). **New architectures**: goldens first, then
  kernels; we supply goldens for any published escha export.

No Mac required for real work: the codec reference, repack and loader all run on Linux
with `mlx[cpu]` — this entire runtime was written blind on Linux and was bit-exact on
first hardware contact.

## Scope

This repository is the **runtime**: it consumes finished escha checkpoints from Hugging
Face. The encoder — calibration, trellis code search, transform construction, the recovery
fine-tune — is what produces those checkpoints, and it is not part of this project.
Runtimes for other platforms are linked from the
[model cards](https://huggingface.co/EschaLabs).

## Related projects — higgs

If you want to *serve* escha models rather than embed this runtime,
[**higgs**](https://github.com/panbanda/higgs) is worth a look. It's an MIT-licensed Rust
server for **free local MLX inference on Apple Silicon**: a single static binary with no
external dependencies that runs open-weight MLX models locally and exposes local and
remote providers (OpenAI, Anthropic, Ollama) behind one endpoint, with API translation
between the OpenAI and Anthropic formats, `higgs shellenv` / `higgs exec` for dropping it
into existing tools, and a daemon mode with a metrics dashboard. It covers Qwen 3.6 and
Qwen 3.x alongside Llama, Mistral, Gemma 2, Phi-3, Starcoder2, DeepSeek-V2 and
LLaVA-Qwen2.

[**@dusterbloom's fork**](https://github.com/dusterbloom/higgs) is where the escha-facing
work lives: the [`nightly`](https://github.com/dusterbloom/higgs/commits/nightly/) branch
has been adding **EschaLabs trellis-quantized checkpoint support** and its own trellis
GEMM kernels, so escha users get a second, independent implementation of the codec to run
against.

**If you use escha models, give both a try** — it's free and it installs as one binary:

```bash
brew install panbanda/brews/higgs
```

A second runtime is the best thing that can happen to a format: independent
implementations are how encoding bugs get found. Issues and results are welcome in any of
these repositories.

## Acknowledgements

- [**@dusterbloom**](https://github.com/dusterbloom) — for help working through the NAX
  (M5 Neural Accelerator) behaviour affecting bit-exact custom Metal kernels, and for the
  NAX neutralization work on
  [higgs `nightly`](https://github.com/dusterbloom/higgs/commits/nightly/). M5 is the
  first chip where a custom Metal kernel's bit-exactness depends on whether the matrix
  coprocessor is in play, and having someone mapping that terrain in parallel made it a
  great deal less lonely.

## License

[Apache-2.0](LICENSE).
