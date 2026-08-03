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
3. **Fused expert input transform.** The Hadamard is computed by an in-register
   butterfly rather than a dense 128×128 matmul — the same f32 arithmetic in a
   different summation order, which avoids materialising ~150 MB of f32
   intermediates per layer. Within 2e-3 relative of both the op chain and the
   NumPy reference; 0.12% of f16 outputs differ by one ulp.
   `ESCHA_MLX_FUSED_HAD=0` restores the op chain exactly.

## Reproducing

From the source tree:

```bash
python bench/baseline.py       --model ./escha-w2 --phases ABCD
python bench/isl_osl_grid.py   --model ./escha-w2 --grid nvidia --concurrency 1,8,16
python bench/head_to_head.py   --a ./escha-w2 --b ./qwen36-4bit --isls 512,2048 --batches 1,2,4,8,16,32
```

Measurement notes, learned the hard way and worth repeating if you benchmark
this yourself: warm up per prompt shape (Metal specialises kernels per shape),
call `mx.clear_cache()` after prefill (otherwise decode reads ~20× slow), and pair
A/B runs within one session with the baseline measured first *and* last — on this
box thermal drift alone moves results by 1–3%, and once by 7%.
