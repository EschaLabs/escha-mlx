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

## The per-linear correction (`ESCHA_MLX_BIAS`)

A dense escha export stores a `bias` tensor beside every coded linear — the
additive correction its end-to-end stage learned. On Qwen3.8-27B all 400 are
non-zero, and applying one moves that linear's output by **6.7–8.3%** (measured
on the goldens in `tests/data/qwen3_5/`), compounding over 64 layers. That is
four orders of magnitude above the fp16-rounding differences every other gate
in this runtime is held to.

**The path those published numbers come from does not apply them.** That
quantization method registers exactly the six `escha_*` tensors and no bias;
every coded linear is constructed `bias=False`; and unmatched `.bias` names are
dropped through a GPTQ-era guard, silently and without a warning. So every published
number for this checkpoint was produced *without* the correction.

This runtime therefore leaves it **off by default**, so that what you serve is
the model those numbers describe, and logs at load that the tensors are present
and unused. `ESCHA_MLX_BIAS=1` applies them, which is what the export format's
own written contract asks for. Which is correct is a question about the
checkpoint, not about this runtime: run it as a paired A/B on a real task
before trusting either.

## Memory on the dense 27B

The numbers above are the 35B MoE's. The dense Qwen3.8-27B has a different
split, and the KV term is the one that bites:

| term | size |
|---|---|
| weights | 10.15 GB (2.469 bpw over 24.3 G coded params + int8 embed/head) |
| KV cache | **64 KiB per token** — 16 full-attention layers × 4 KV heads × 256 head dim × 2 (K,V) × 2 B. That is 2.15 GB at ctx 32k, 4.29 GB at 64k, 8.59 GB at 128k. |
| GDN recurrent state | 75.5 MB per sequence at fp16 (48 GDN layers × 48 v-heads × 128 × 128) |

Per token the dense model's KV is ~3.2× the MoE's (which has 10 attention
layers × 2 KV heads), so context, not batch, is the first thing to run out of
room. Against the ~19 GB working set a 24 GB Mac gives you, 10.15 GB of weights
leaves roughly 8.8 GB — a single-stream ceiling somewhere near 120k tokens,
against the `max_position_embeddings` of 262144 the config advertises. Budget
context first, then concurrency.

One caveat with teeth: **`--max-kv-size` and the rotating KV cache are silently
inert on this architecture.** `mlx_lm.models.cache.make_prompt_cache` only
builds a `RotatingKVCache` "if the model does not have a `make_cache` method",
and every hybrid GDN model defines one — it has to, because the linear-attention
layers need a recurrent-state cache rather than a KV cache. That is true of
stock mlx-lm as well; this runtime's fp16 GDN state does not cause it and
`ESCHA_MLX_GDN_STATE=fp32` does not undo it. Plan for a KV that grows with the
sequence, and bound context at the request level.

## Tuning reference

All optional. The defaults are what we measured as best on an M4; every one of
these is gated bit-identical or documented where it is not.

| variable | effect |
|---|---|
| `ESCHA_MLX_WIRED_GB=N` | wire N GB. **Required above a ~18 GB working set** (see the cliff above). Must be ≤ the cap. |
| `ESCHA_MLX_FUSED_HAD=0` | use native MLX op chains for the expert input and output transforms. The default fused Metal kernels combine the scale gathers, radix-2 Hadamard, scaling and f16 cast without changing the final f16 bits; set this to 0 for performance comparison or debugging. |
| `ESCHA_MLX_GDN_STATE=fp32` | store the recurrent state in f32 instead of fp16. Costs ~10% throughput at batch ≥32; the per-sequence cost is architecture-dependent — 31.5 MB on the 35B MoE (30 GDN layers × 32 v-heads), 75.5 MB on the 27B dense (48 × 48). The loader logs the actual figure for your model. Use it if you need the pre-fp16 numerics exactly. |
| `ESCHA_MLX_LAST_LOGIT=0` | compute logits for **all** prompt positions, not just the last. Needed for per-position scoring (loglikelihood eval); costs ~7% prefill. |
| `ESCHA_MLX_Q8_GROUP=64` | 64-wide Q8 groups instead of 128. Identical numerics, +140 MB. |
| `ESCHA_MLX_BLOCK_R=N` | pin rows-per-expert-group. Default is size-dependent. |
| `ESCHA_MLX_KT_BLOCK=N` | code tiles staged per barrier pair. Default 4. |
| `ESCHA_MLX_GEMV=staged` | revert to the barrier/threadgroup-staged per-row GEMV (default is the barrier-free direct kernel). Bit-identical. |
| `ESCHA_MLX_DENSE=fp16` | fp16 dense weights instead of the Q8 repack. +~1.9 GB resident; bit-identical weight values. |
| `ESCHA_MLX_LUT=1` | table-based codec decode instead of the multiply-hash. Bit-exact by construction; use if a future Metal compiler ever breaks fp16 round-to-nearest-even in the hash path. |
| `ESCHA_MLX_MOE=ops` | NumPy expert path. Very slow; a correctness oracle, not for serving. |
| `ESCHA_MLX_BIAS=1` | apply the per-linear correction a dense export ships. **Off by default, and this is a real fork in the model, not a tuning knob** — see below. |
| `ESCHA_MLX_LINEAR=ops` | NumPy path for the coded linears of a **dense** model (the MoE flag's counterpart). Very slow, and it materializes each fp16 weight; a correctness oracle, not for serving. |
| `ESCHA_MLX_DENSE_BLOCK_R=N` | pin rows-per-group for the dense row-blocked GEMM (1 = always the per-row kernel). Default is size-dependent and, unlike the rest of this table, **has not been measured on Metal** — see below. Bit-identical at every R. |

Four further flags (`ESCHA_MLX_SPLITK`, `ESCHA_MLX_FETCH`, `ESCHA_MLX_SORTX`,
`ESCHA_MLX_PREFETCH`) select alternate kernel strategies that measured neutral or
worse on a 10-core M4. They are kept because the trade-offs are
hardware-dependent and may favour wider GPUs (M-series Max/Ultra). All are gated
bit-identical.

**`ESCHA_MLX_DENSE_BLOCK_R` is the one unmeasured default in this table.** The
dense path was developed on Linux, where the Metal kernels do not run, so the
policy comes from the structure of the kernel rather than from a sweep. Unlike
the MoE case, every row of a dense linear shares the one coded stream, so a
group is never mostly padding and the decoded-stream reads fall by very nearly
R — the default is therefore `R = min(rows, 8)` snapped down to a power of two:
never more rows-per-group than there are rows (so it never pads), and greater
than 1 from two rows upward, because at concurrency 2 or 3 an `R = 1` would
read the entire coded stream once per row. The ceiling of 8 is where the
argument runs out (registers and staging), not where a measurement put it.
Sweeping this is the first thing worth doing on a dense model on real hardware.
Correctness does not depend on it: every R is gated bit-identical to the
per-row kernel.

## Troubleshooting

**`Insufficient Memory` / `kIOGPUCommandBufferCallbackErrorOutOfMemory`** — the
working set exceeded the cap. Lower `--decode-concurrency`, lower
`--prefill-step-size`, or raise the cap (see above).

**Decode suddenly runs at a few tok/s** — you are past the wired limit and
paging. This is the cliff; it does not raise an error. Check peak memory against
your cap.

**`ValueError: ... is not an escha checkpoint`** — the path is not an escha
export. This runtime loads `escha` (dense) and `eschamoe` (mixture-of-experts)
exports; use stock `mlx-lm` for other formats.

**`ValueError: unsupported model_type ...`** — the export *is* escha, but no
plugin is registered for its architecture. Supported types are listed in the
error; open a [model request](https://github.com/EschaLabs/escha-mlx/issues/new?template=new_model.yml).

**Very slow first call after each new prompt length** — expected. Metal
specialises kernels per shape; the first call at a new shape includes
compilation. It is cached afterwards.
