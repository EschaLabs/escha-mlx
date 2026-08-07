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
  goldens — not a tolerance, and CI-enforced on every PR
- **Honest benchmarks**: measured on hardware, drift-controlled, negative results included

## Supported models

| model | quant | resident | status |
|---|---|---|---|
| [`EschaLabs/Qwen3.6-35B-A3B-Escha-W2`](https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2) | 2-bit experts + int8 dense | 12.3 GB | ✅ supported |
| Qwen3.5 VLM (dense) | — | — | 🚧 planned |
| Kimi K3 / GLM / DeepSeek MoE | — | — | 🔭 exploring |

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

Measurements on 24 GB Macs using macOS 26.5.2, MLX 0.32.0 and mlx-lm 0.31.3.
The M4 is the entry-level 10-core-GPU model; the M5 Pro has a 16-core GPU. Every
M5 Pro result in this summary table used runtime revision
`bf86c10d4d91e5d4aaa7d4046983723e139f47cc`, model revision
`1b7237f0886a10b4bd92cd7653090cd7381ae199`, AC power and High Power mode.

| workload | M4 base, escha W2 | M4 base, stock MLX 4-bit | M5 Pro, escha W2 |
|---|---:|---:|---:|
| resident memory | **12.25 GB** | 19.51 GB | **11.41 GB** |
| prefill, ISL 512 | 264 tok/s | — | **756.5 tok/s** |
| decode, single stream | 27.3 tok/s | **42.9 tok/s** | **59.68 tok/s** |
| aggregate @ batch 8 | 59.6 tok/s | **101.5 tok/s** | **193.03 tok/s** |
| aggregate @ batch 16 | **104.0 tok/s** | out of memory | **239.96 tok/s** |
| aggregate @ batch 128 | **185.6 tok/s** (18.1 GB peak) | out of memory | **539.25 tok/s** (18.06 GB peak) |
| served peak output / total | 99.5 / 244.5 tok/s | — | **206.00 / 589.08 tok/s** |

Prefill runs ~264 tok/s; single-stream decode reaches 69% of this chip's 39.3 tok/s
bandwidth ceiling. Read the comparison honestly in both directions: below batch 16 the
4-bit build is faster; from batch 16 up **only escha runs at all** on 24 GB — that is the
regime the 1.59× footprint buys, and where a Mac serving more than one user lives.

The M5 Pro results are a separate machine characterization, not a paired cross-chip
A/B. Its B=1/8/16/128 entries are five-run medians from the 16-token,
step-synchronized repeatability harness; B=128 alone uses a 19 GB wired limit.
The current P0 gates pass and the complete suite reports **180 passed, 1 skipped**.
A same-process native-vs-output-fused A/B/A through B=32 is reported in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md#current-output-hadamard-fusion-aba).

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
  this gate; the complete suite now reports **180 passed, 1 skipped** with
  default TF32 and with `MLX_ENABLE_TF32=0`. Historical counts and the diagnostic
  TF32 A/B are retained in
  [docs/PERFORMANCE.md](docs/PERFORMANCE.md#m5-pro-resolved-issue-dense-matmul-test-oracle-and-tf32).

## Repository layout

| path | what |
|---|---|
| `escha_mlx/ref.py` | NumPy bit-exact codec reference — **the semantic contract** |
| `escha_mlx/msl.py` | the Metal kernels (`mx.fast.metal_kernel`): decode, GEMV ×2, row-blocked GEMM, fused transform |
| `escha_mlx/quant.py` / `moe.py` / `loader.py` | int8→Q8 repack · expert toolkit · streaming loader — all architecture-agnostic |
| `escha_mlx/models/` | one plugin per architecture (`qwen3_5_moe` today): skeleton, tensor map, router, quirks |
| `escha_mlx/gdn_cache.py` | recurrent-state cache (fp16 state) |
| `escha_mlx/{generate,server}.py` | CLI / OpenAI-compatible server |
| `tests/` + `tests/data/` | golden-gated suite + the committed reference vectors |
| `bench/` | gates, roofline, serving grid, head-to-head; results per machine under `bench/results/` |
| `docs/` | install & tuning · performance · full campaign record |

Everything outside the Metal kernels is pure MLX; GDN, attention and the caches come from
mlx-lm untouched.

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
