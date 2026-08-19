# Installing the escha MLX runtime

## Requirements

| | |
|---|---|
| Hardware | Apple Silicon (M1–M5). Intel Macs are not supported — MLX requires Metal 3 + unified memory. |
| macOS | 14 (Sonoma) or newer. 15+ recommended: `mx.set_wired_limit` needs it, and that matters at high concurrency (see below). |
| RAM | **24 GB minimum.** 32 GB+ if you want long context *and* concurrency. |
| Python | 3.10–3.13 |
| Disk | ~13 GB for the checkpoint |

Verify you are on Apple Silicon:

```bash
sysctl -n machdep.cpu.brand_string     # should say "Apple M..."
```

## 1. Install

From source (recommended while the project is young — you get the tests and benches):

```bash
git clone https://github.com/EschaLabs/escha-mlx && cd escha-mlx
python3 -m venv ~/.venv-escha
source ~/.venv-escha/bin/activate
pip install -e '.[test]' 'mlx==0.32.*' 'mlx-lm==0.31.3'
```

Or install the wheel attached to the latest
[GitHub release](https://github.com/EschaLabs/escha-mlx/releases):

```bash
pip install escha_mlx-<version>-py3-none-any.whl
```

Either pulls `mlx`, `mlx-lm`, `numpy` and `safetensors`. Confirm Metal is live:

```bash
python -c "import mlx.core as mx; print(mx.metal.is_available(), mx.__version__)"
# True 0.32.0
```

## 2. Get the checkpoint

```bash
pip install huggingface_hub
huggingface-cli download EschaLabs/Qwen3.6-35B-A3B-Escha-W2 --local-dir ./escha-w2
```

No conversion step — the runtime reads the published export directly.

## 3. Run

```bash
# one-shot
escha-mlx-generate --model ./escha-w2 --prompt "What is 17 multiplied by 23?"

# OpenAI-compatible server
escha-mlx-server --model ./escha-w2 --port 8080 --prefill-step-size 256
```

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "./escha-w2",
  "messages": [{"role": "user", "content": "Say hello."}],
  "max_tokens": 64
}'
```

Continuous batching and prefix caching come from mlx-lm's scheduler and are on
by default.

---

## Memory settings — read this before running at concurrency

macOS caps how much unified memory the GPU may hold. On a 24 GB Mac that cap is
about **19 GB**. Check yours:

```bash
python -c "import mlx.core as mx; print(mx.device_info()['max_recommended_working_set_size']/1e9, 'GB')"
```

### The 23× cliff

MLX **wires nothing by default** (`mx.set_wired_limit` starts at 0). While the
working set is comfortably under the cap this costs nothing. Once you approach
the cap, macOS starts evicting pages and throughput collapses — silently, with no
error, just a very slow model. Measured on an M4 24 GB at batch 80 (19.3 GB
working set):

| | tok/s | run-to-run spread |
|---|---|---|
| nothing wired | **5.9** | 335% |
| `ESCHA_MLX_WIRED_GB=20` | **136.6** | 0.7% |

So if your working set will come within ~2 GB of the cap:

```bash
sudo sysctl iogpu.wired_limit_mb=21000     # raises the CAP (resets on reboot)
export ESCHA_MLX_WIRED_GB=20               # actually WIRES it
```

**Both are required.** The sysctl only raises the ceiling; `ESCHA_MLX_WIRED_GB`
is what makes MLX use it. Neither alone is enough.

Below ~18 GB of working set, leave both alone — wiring is a system-wide
commitment of memory and buys nothing there.

### Serving flags that matter on 24 GB

```bash
# --prefill-step-size 256: mlx-lm's default of 2048 peaks at 20.7 GB
escha-mlx-server --model ./escha-w2 --port 8080 \
    --prefill-step-size 256 \
    --decode-concurrency 16
```

`--prefill-step-size 256` is not optional on a 24 GB machine: the default
materialises a much larger activation transient and pushes peak memory past the
cap on long prompts.

---

## Tuning reference

All optional. The defaults are what we measured as best on an M4; every one of
these is gated bit-identical or documented where it is not.

| variable | effect |
|---|---|
| `ESCHA_MLX_WIRED_GB=N` | wire N GB. **Required above a ~18 GB working set** (see the cliff above). Must be ≤ the cap. |
| `ESCHA_MLX_FUSED_HAD=0` | use native MLX op chains for the expert input and output transforms. The default fused Metal kernels combine the scale gathers, radix-2 Hadamard, scaling and f16 cast without changing the final f16 bits; set this to 0 for performance comparison or debugging. |
| `ESCHA_MLX_GDN_STATE=fp32` | store the recurrent state in f32 instead of fp16. Costs ~10% throughput at batch ≥32 and 31.5 MB/sequence; use it if you need the pre-fp16 numerics exactly. |
| `ESCHA_MLX_LAST_LOGIT=0` | compute logits for **all** prompt positions, not just the last. Needed for per-position scoring (loglikelihood eval); costs ~7% prefill. |
| `ESCHA_MLX_Q8_GROUP=64` | 64-wide Q8 groups instead of 128. Identical numerics, +140 MB. |
| `ESCHA_MLX_BLOCK_R=N` | pin rows-per-expert-group. Default is size-dependent. |
| `ESCHA_MLX_KT_BLOCK=N` | code tiles staged per barrier pair. Default 4. |
| `ESCHA_MLX_GEMV=staged` | revert to the barrier/threadgroup-staged per-row GEMV (default is the barrier-free direct kernel). Bit-identical. |
| `ESCHA_MLX_DENSE=fp16` | fp16 dense weights instead of the Q8 repack. +~1.9 GB resident; bit-identical weight values. |
| `ESCHA_MLX_LUT=1` | table-based codec decode instead of the multiply-hash. Bit-exact by construction; use if a future Metal compiler ever breaks fp16 round-to-nearest-even in the hash path. |
| `ESCHA_MLX_MOE=ops` | NumPy expert path. Very slow; a correctness oracle, not for serving. |
| `ESCHA_MLX_FOLD_GEN=0` | disable the folded-first-token generate optimisation and restore the stock `mlx_lm.generate.generate_step`. The fold saves ~16 ms TTFT per request by sampling the first token from the final prefill chunk's logits instead of an extra single-token forward; token output is identical to stock, but logprob vectors may differ numerically (different kernel shapes). Default on. |
| `ESCHA_MLX_ASYNC_EVAL=0` | disable per-forward `mx.async_eval` on model output. Default on; overlaps each GPU forward with the next step's Python graph-build (~3.2 ms/step), keeping the GPU busy between decode steps. Output is bit-identical on/off. |
| `ESCHA_MLX_KERNEL_WARM=0` | skip kernel warm-up at load time. Default on; runs throwaway prefill and decode forwards at load so that Metal kernel compilation happens before any measured dispatch. Disable for cold-start fidelity testing. |

Four further flags (`ESCHA_MLX_SPLITK`, `ESCHA_MLX_FETCH`, `ESCHA_MLX_SORTX`,
`ESCHA_MLX_PREFETCH`) select alternate kernel strategies that measured neutral or
worse on a 10-core M4. They are kept because the trade-offs are
hardware-dependent and may favour wider GPUs (M-series Max/Ultra). All are gated
bit-identical.

## Troubleshooting

**`Insufficient Memory` / `kIOGPUCommandBufferCallbackErrorOutOfMemory`** — the
working set exceeded the cap. Lower `--decode-concurrency`, lower
`--prefill-step-size`, or raise the cap (see above).

**Decode suddenly runs at a few tok/s** — you are past the wired limit and
paging. This is the cliff; it does not raise an error. Check peak memory against
your cap.

**`ValueError: ... is not an eschamoe checkpoint`** — the path is not an escha
export. This runtime only loads `eschamoe`; use stock `mlx-lm` for other formats.

**Very slow first call after each new prompt length** — expected. Metal
specialises kernels per shape; the first call at a new shape includes
compilation. It is cached afterwards.
