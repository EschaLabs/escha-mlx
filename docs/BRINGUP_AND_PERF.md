# escha-mlx on Apple Silicon — first hardware bring-up + perf campaign

**Date:** 2026-07-30 · **Box:** MacBook Pro, **Apple M4 base** (10-core GPU, 4P+6E), 24 GB, macOS 26.5.2
**Stack:** python 3.12 (uv venv `~/.venv-escha-mlx`), `mlx==0.32.0`, `mlx-lm==0.31.3`, `escha_mlx` editable
**Model:** `EschaLabs/Qwen3.6-35B-A3B-Escha-W2` (12.30 GB, 1362 tensors, vocab 248,320)

This is the record of the first contact between the blind-written escha-mlx runtime and
real Apple hardware, plus the perf campaign that followed. It is kept verbatim as a lab
notebook — including every negative result — rather than edited into a retrospective.

---

## 0. Headline

| | result |
|---|---|
| Metal kernels, first hardware contact | **11/11 gates pass, bit-exact vs the reference goldens** |
| Correctness battery | **all pass** (Paris / 391 / coherent thinking-mode haiku) |
| Batched decode B=1..48 | **replication-invariant at every B** (R6 risk cleared) |
| bs1 decode | **27.7 tok/s** (72% of the corrected roofline ceiling — §9.1) |
| In-process aggregate | **93.7 tok/s @ B=16** within the wired limit (105.7 @ B=24 exceeds it) |
| **Determinism** | **bit-reproducible** — identical tokens *and* identical logit hash across runs and processes (§10.5) |
| **In-process aggregate, after §10** | **135–140 tok/s @ B=64**, 17.9 GB (§10.3) |
| **In-process aggregate, after §12** | **167.5 tok/s @ B=128**, 18.98 GB — fp16 GDN state raised the ceiling (§12) |
| **Served, peak output throughput** | **99.5 tok/s** (128/1024, C=16) — was 81.9 pre-fusion (§18) |
| **Served, peak total throughput** | **244.5 tok/s** (2048/128, C=16) — was 191.4 (§18) |
| Prefill (after the kernel work) | **94–127 → ~207 tok/s**; **243 after §13**; **~261 after §17** |
| Peak memory @ 4k ctx | **20.74 → 12.68 GB** |
| vLLM / vllm-metal | **rejected on evidence today** — §6 |
| **vs stock mlx-lm 4-bit, same box** | **1.59× smaller**; 1.6× slower at bs1 (§14), and at batch we read up to **1.63× fewer bytes** but ran at 39–53% of roofline vs their 80% (§15 — whole-step figures, pre-§17; see the §16 correction: the trellis kernel itself beats `gather_qmm` at matched shapes — the gap is the transform pipeline + int8 dense vs their 4-bit dense, partly recovered by the §17 fusion) |

On the "100 tok/s" goal: met on **total system throughput** (149.5 at 1000/1000
C=16, 244.5 at 2048/128 C=16) and on **in-process decode** from B=16 up
(185.6 at B=128). Served **output-only** throughput now peaks at **99.5 tok/s**
(128/1024, C=16) — essentially at the target, up from 81.9 before the fused
transform kernel (§17, §18).

**Important hardware correction to the original plan.** The plan was sized for
M4/M5 *Max* (546/614 GB/s) and treated a 24 GB machine as the validation SKU. This
box is a **base M4 — 120 GB/s**, i.e. 4.5× less bandwidth than the dev-box tier the
perf targets were written against. Every number below should be read against that.

Two doc assumptions were also measured wrong, both in our favour:
* wired limit is **19.07 GB**, not the assumed ~17.2 GB (this M4 gets 3/4 of RAM, not 2/3);
* `max_buffer_length` is **14.3 GB** — a hard per-buffer cap worth remembering.

---

## 1. The theoretical ceiling on this instance

Decode is purely memory-bandwidth bound, so `tok/s = achieved_BW / bytes_per_token`.
Both terms were measured, not assumed (`bench/roofline.py`).

**Denominator — exact byte ledger**, walked from the loaded model:

| stream | MB/token |
|---|---|
| GDN dense int8 (30 layers) | 1142.3 |
| lm_head (248320×2048 Q8) | 572.1 |
| attn dense (10 layers) | 306.7 |
| routed experts (top_k=8 of 256, K2 gate_up + K3 down) | 300.8 |
| shared expert | 141.6 |
| router gate | 42.1 |
| **total** | **2.506 GB/token** |

Note the shape of this: **88% of the bs1 byte budget is dense weights that amortize
perfectly across a batch**; only 300 MB scales with tokens. That is what makes the
batching story good despite the low bandwidth.

**Numerator — measured roofline:** streaming fp16 read = **101.0 GB/s = 84% of the
120 GB/s advertised**. (An independent agent measured 103.7 GB/s on the same box.)

| ceiling | tok/s |
|---|---|
| at 101 GB/s achievable roofline | **40.3** |
| at 120 GB/s advertised peak | 47.9 |
| **measured bs1** | **27.8 → 69% of roofline** |

So the honest statement is: **bs1 100 tok/s is physically impossible on this
instance** — it would need 251 GB/s, more than twice the chip's bandwidth. 100+ tok/s
is reachable only as *aggregate* throughput under concurrency, which is what §4 shows.

### 1.1 Where the remaining gap is (in-graph ablation, not microbenchmarks)

Isolated kernel timings cannot answer this: a standalone call pays an
eval+synchronize round trip (~168 µs here) that a fused 40-layer graph never
pays, overstating small components by >2×. So components were ablated **inside
the real model** — each stubbed with a *data-dependent* cheap op (a constant
stub lets MLX dead-code-eliminate the whole upstream graph and reports the entire
model as that component's cost — this bit once and is worth knowing).

bs1, 41.0 ms/step, vs the same bytes at the 101 GB/s roofline:

| component | bytes | measured | at roofline | efficiency |
|---|---|---|---|---|
| `lm_head` | 572 MB | 5.89 ms | 5.7 ms | **97%** |
| attn + GDN + norms + sampling | 1449 MB | 20.34 ms | 14.3 ms | 70% |
| **routed experts (trellis)** | **301 MB** | **11.00 ms** | **3.0 ms** | **27%** |
| whole step | 2506 MB | 41.00 ms | 24.8 ms | 60% |

At B=16 (171.7 ms/step) the trellis experts are **89.7 ms — 52% of the step**.

**Interpretation.** `lm_head` — the single largest pure-GEMV stream — runs at 97%
of roofline, which is the evidence that the bandwidth-bound parts of this runtime
are already near-optimal. The shortfall is concentrated in (a) the trellis codec,
which is ALU/latency-bound rather than bandwidth-bound at low row counts, and (b)
mlx-lm's GDN glue. A pure-bandwidth roofline is an upper bound no real decoder
reaches; 60–70% of it is the normal band for a well-tuned LLM decode loop, and
80–95% is not a realistic target for a hybrid model carrying a 2-bit trellis
codec. The reachable next step is the trellis kernel: closing it to ~2× off
roofline would put bs1 near 31–32 tok/s (~80% of the 40.3 ceiling).

---

## 2. Correctness first (all gates green)

```
tests/test_ref_decode.py     6 passed   NumPy reference decode vs the goldens
tests/test_mlx_cpu.py        4 passed   Q8 repack bit-exact, w8a16 golden, MoE block
tests/test_metal.py         11 passed   K2/K3 tile decode + expert-stride + hash==LUT
tests/test_blocked.py       11 passed   row-blocked GEMM bit-identical (new, §3)
bench/p0_gates.py           ALL PASS
```

The blind-written MSL kernels were **bit-exact on first execution** — no compiler
divergence on the fp16-RNE hash path, so `ESCHA_MLX_LUT` was never needed.

Model-level battery: `" Paris, a city renowned for its iconic landmarks…"`, `17*23 → 391`,
coherent thinking-mode haiku. Load: **23 s, 11.55 GB resident**.

**Batched decode replication invariance** (bring-up gate R6 — batch≥3 hybrid-cache
corruption is a known ecosystem failure class): every batch size B ∈ {1,2,4,8,12,16,20,24,32,48}
produced rows identical to each other *and* identical to the B=1 sequence. Clean.

---

## 3. The one real bottleneck: the trellis kernel had zero batch amortization

`moe_gemv` reads **one full expert stream per row**, so cost is exactly linear in rows
with no reuse. At a prefill chunk of S tokens that is S×8 rows × 917 KB × 40 layers —
a 256-token chunk streams **75 GB**, which is why prefill sat at ~115 tok/s.

The originally planned fix ("dequantize to fp16 + `mx.gather_mm`") was measured
and **loses on every axis**: 295 ms/layer vs 250 for the existing kernel, plus a
1.61 GB/layer transient that does not fit next to 11.55 GB resident.

**What shipped instead — row-blocked trellis GEMM** (`msl.moe_gemm_rows` +
`moe.build_groups`): rows are sorted by expert and packed into fixed R-row groups, so
one expert stream read is shared by R rows. The kt/j accumulation order for a fixed
(row, out-channel) is untouched, so output is **bit-identical**, not merely close —
gated by `tests/test_blocked.py` with `np.array_equal` on raw f32, including
all-one-expert and one-per-expert skew.

Kernel A/B (same session, paired):

| M (rows) | R | gate_up K2 | down K3 |
|---|---|---|---|
| 64 | 4 | 0.93× | 1.37× |
| 256 | 4 | 1.28× | 1.17× |
| 2,048 | 16 | 2.29× | 2.17× |
| 8,192 | 16 | 3.83× | 3.59× |
| 16,384 | 16 | **4.83×** | **4.03×** |

At M=64 it is a slight *loss* — at bs1 the 8 draws land on ~8 distinct experts out of
256, so there is nothing to amortize and the E·(R−1) padding tax dominates. Hence the
size-dependent dispatch in `_blocked_R()`: **R=1 below 256 rows, R=4 to 2048, R=16 above**.

Two further fixes landed with it, both memory:
* **chunked prefill returning only the last position's logits** — the `[1, S, 248320]`
  tensor is 2.03 GB at S=4096 and prefill never needs row ≠ −1;
* **`mx.clear_cache()` after prefill** — see §5, this one is worth 12–24×.

Result: **prefill 94–127 → ~200 tok/s**, **peak @ ISL 4096: 20.74 → 12.68 GB**.
Chunk 256 is optimal; larger chunks *regress* because `_expert_path` intermediates
scale with M and outweigh the kernel gain.

### 3.1 Barrier-free per-row GEMV (the bs1 path)

Row-blocking does nothing at bs1 (R=1 there), so the decode path got a separate
fix. The staged kernel ran TK iterations (128 for gate_up) with **two threadgroup
barriers each — 256 synchronization points** on a chain already serial in kt,
while achieving only 11 GB/s of a 101 GB/s roofline: it is latency-bound, not
bandwidth-bound. `_moe_gemv_direct_source` drops the shared-memory staging of
both code and x; each simdgroup reads its own code tile and each lane its x slice
straight from device memory (redundant across lanes, same cache line), leaving
**zero synchronization in the inner loop**.

Bit-identical (21/21 gates, `tests/test_gemv_direct.py`, m ∈ {1,8,16,64,257} ×
K ∈ {2,3} × hash/LUT). Measured **1.15–1.78× on the kernel, 1.31× at bs1 row
counts**, **1.086× end-to-end** on the real generation path (25.5 → 27.7 tok/s).
Default ON; `ESCHA_MLX_GEMV=staged` reverts.

---

## 4. Throughput vs concurrency (direct, in-process)

Same prompt replicated across the batch, ISL=128, warmed, `mx.clear_cache()` before timing.

| B | 1 | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|
| aggregate tok/s (staged kernel) | 22.8 | 54.8 | 62.6 | 91.0 | 105.3 |
| **aggregate tok/s (final)** | **24.1** | **57.7** | **63.7** | **93.7** | **105.7** |
| per-seq tok/s | 24.1 | 14.4 | 8.0 | 5.9 | 4.4 |
| peak GB | 12.2 | 13.6 | 14.8 | **17.2** | 19.9 |

**The 100 tok/s target is met at B=24 (105 tok/s) — but read the memory column.**
B=24 peaks at **19.9 GB, past the 19.07 GB wired limit** (the B=32/48 points of the
earlier staged-kernel sweep were far past it and included paging). The **honest
stable operating point today is B=16: 93.7 tok/s at 17.2 GB**, with points near the
cap noisy for exactly this reason.

Per-sequence memory is dominated by the **GDN recurrent state: 64.4 MB/seq, constant
in sequence length** (30 `ArraysCache` layers, fp32 state hardcoded upstream) — equal
to the KV of a ~3,100-token sequence. KV itself is only 20.5 KB/token across the 10
SDPA layers. So concurrency on this box is GDN-state-bound, not KV-bound.

---

### 4.1 Served: NVIDIA-style ISL/OSL grid

`escha_mlx.server` (`--decode-concurrency 16 --prompt-concurrency 2
--prefill-step-size 256`), unique prompts per request (no prefix-cache
contamination), thinking off, `ignore_eos` so OSL is exact — **`osl_hit` = 1.00
on every row**. Raw JSON: `bench/results/m4-base-24gb/baseline.json`.

| ISL | OSL | C | TTFT p50 | TTFT p99 | TPOT ms | per-user | **out tok/s** | **total tok/s** |
|---|---|---|---|---|---|---|---|---|
| 128 | 128 | 1 | 0.93 | 1.15 | 40.7 | 20.93 | 20.92 | 43.8 |
| 128 | 128 | 8 | 5.23 | 7.36 | 154.9 | 5.38 | 42.27 | 88.5 |
| 128 | 128 | 16 | 11.82 | 16.47 | 222.4 | 3.35 | 52.00 | 109.0 |
| 128 | 1024 | 1 | 0.96 | 1.16 | 40.1 | 24.30 | 24.30 | 27.6 |
| 128 | 1024 | 8 | 5.30 | 7.43 | 130.2 | 7.43 | 59.37 | 67.5 |
| 128 | 1024 | 16 | 11.89 | 16.32 | 184.5 | 5.15 | **81.94** | 93.2 |   <!-- superseded by §18 -->
| 1000 | 1000 | 1 | 4.97 | 5.17 | 40.5 | 22.04 | 22.04 | 44.4 |
| 1000 | 1000 | 8 | 12.23 | 37.32 | 153.3 | 6.10 | 48.24 | 97.1 |
| 1000 | 1000 | 16 | 50.46 | 77.03 | 224.9 | 3.76 | 59.71 | **120.1** |
| 2048 | 128 | 1 | 10.17 | 10.48 | 40.6 | 8.37 | 8.37 | 143.0 |
| 2048 | 128 | 8 | 23.01 | 79.16 | 555.0 | 1.65 | 10.85 | 185.5 |
| 2048 | 128 | 16 | 104.45 | 163.51 | 747.2 | 0.72 | 11.20 | **191.4** |

**Peak output throughput 81.9 tok/s** (128/1024 C=16) — 87% of the in-process
B=16 figure; the remainder is endpoint SSE/detokenization.
**Peak total throughput 191.4 tok/s** (2048/128 C=16), because prefill runs at
~200 tok/s vs ~24 for decode, so prefill-heavy shapes move the most tokens.

Two things this table says plainly:

1. **At short OSL the limiter is the scheduler, not the model.** 128/128 C=8
   reaches only 66% of the in-process B=8 number; 128/1024 C=8 reaches 93%. The
   difference is prefill/ramp that cannot amortize over 128 output tokens.
2. **`--prompt-concurrency 2` is wrong for long inputs.** TTFT p99 blows out to
   77 s at 1000/1000 C=16 and 163 s at 2048/128 C=16 — 16–32 k prefill tokens
   queued two at a time. Throughput is fine; latency is not. Raising it (memory
   permitting) is the obvious next tuning step and was not swept here.

ISL 5000 / 20000 were not run: at ~200 tok/s prefill a single 20 k prompt is
~100 s of TTFT, and 16 of them exceed the session budget. They are feasible on
memory (KV is only 20.5 KB/token) — this is a time bound, not a capability bound.

## 5. Two traps worth recording

**(a) MLX's buffer cache silently costs 12–24× decode throughput.** After a long
prefill, `mx.get_cache_memory()` holds 3–5 GB of transients; active + cache pushes past
the wired limit and the machine thrashes. It looks exactly like context-length scaling:

| ISL | decode as-is | after `mx.clear_cache()` |
|---|---|---|
| 512 | 20.3 | 22.6 |
| 2048 | **1.84** | **22.4** |
| 4096 | **1.21** | **21.5** |

Decode is in fact **flat at ~21–23 tok/s from ISL 128 → 4096**. `mx.clear_cache()`
after prefill is mandatory, not hygiene.

**(b) MLX laziness invalidates naive microbenchmarks.** `for _ in range(n): r = fn()`
followed by one `mx.eval(r)` builds n graphs and computes **one**. This reported
**2055 GB/s on a 120 GB/s box** — a 17× overstatement — before it was caught. An
independent agent hit the identical trap the same session. Every iteration must be
forced (`bench/roofline.py::timeit`).

---

## 6. vLLM / vllm-metal — rejected today, on evidence

vllm-metal is alive (474 commits, v0.2.0 April 2026, arm64 py3.12) and its
supported-models table does list our architecture. It is still the wrong move now:

1. **Prefix caching is explicitly ❌ for Qwen3.5/3.6 hybrids** in their own
   supported-models doc — "Upstream vLLM keeps it off for hybrid/Mamba models."
   Mounting there would **lose** the prefix cache that already works for us on mlx-lm.
2. **Quant-format routing.** Supported custom formats are AWQ / GGUF / MLX-native;
   `quant_method: "eschamoe"` is none of them, and this checkpoint additionally carries
   a `vision_config`, which routes it to the VLM loader path.
3. Nothing in it addresses the prefill bottleneck, which was our actual problem.

**Recommendation: ship the standalone mlx-lm path; revisit vllm-metal only if/when
hybrid prefix caching lands upstream.** The original drop criterion applies.

---

## 7. What the runtime gets for free from mlx-lm 0.31.3

Verified present, not assumed: `BatchGenerator` (real continuous batching with
mid-flight admission, per-sequence stop machines, eviction), `LRUPromptCache` +
`PromptTrie` (prefix caching — confirmed live, `cached_tokens: 20` on a repeat
request), reasoning-content parsing, tool calls. `json_schema` / structured output is
genuinely **absent** — this belongs in the runtime README's Known Limitations, never
the model card.

Added in our wrapper: **`ignore_eos`**, required for honest benchmarking. Without it
random-token prompts hit EOS early (`osl_hit_rate` 0.04), the batch drains, and the
harness measures a decaying batch rather than steady state.

---

## 8. Reproduce

```bash
source ~/.venv-escha-mlx/bin/activate
pytest tests/ -q                     # 32 passed, 1 skipped at bring-up; the suite has since grown (current count: docs/PERFORMANCE.md)
python bench/p0_gates.py
python bench/roofline.py --model ~/models/escha-w2    # ceiling analysis
python bench/baseline.py --model ~/models/escha-w2 --phases ABCD

python -m escha_mlx.server --model ~/models/escha-w2 --port 8080 \
    --decode-concurrency 16 --prompt-concurrency 2 --prefill-step-size 256
python bench/isl_osl_grid.py --model ~/models/escha-w2 \
    --grid nvidia --concurrency 1,8,16 --out results.json
```

`--prefill-step-size 256` matters: mlx-lm's default of 2048 is what produced the
20.7 GB peak.

---

## 9. Next levers, ranked by the §1.1 ablation

1. **Trellis GEMV occupancy at low row counts** — the largest single gap: 27% of
   roofline at bs1, and **52% of the whole step at B=16**. At M=8 the kernel
   launches only 64 threadgroups for gate_up, each running a 128-iteration serial
   kt chain, on a 10-core GPU. The lever is **split-K**: partition TK across
   several threadgroups (4× the parallelism, 4× shorter chains) with a two-pass
   f32 reduction — the partials are tiny (4×8×1024 f32 = 128 KB). Expected to take
   bs1 to ~31–32 tok/s (≈80% of the 40.3 ceiling) and to help B≥8 more.
2. **GDN state fp16** (64.4 → 32 MB/seq) roughly doubles safe concurrency, which is
   the binding constraint on aggregate throughput (§4). Needs a numerics gate first —
   GDN silent-wrongness is a demonstrated failure mode in this project's history (R4).
3. **`mx.compile` on the MoE glue** — 1.20× measured on a chained 40-layer graph;
   attn+GDN+glue is 70% of roofline and is where this would land.
4. **Dense Q8 dead zone at M≈7–16** — `quantized_matmul` costs 2.7× more at T=8 than
   T=1 for identical bytes, landing exactly on the concurrency we want. Unverified on
   the real per-layer shapes; worth one experiment.
5. Raising `iogpu.wired_limit_mb` (~21000) to make B=24 a legitimate operating point
   rather than a paging measurement. *(Levers 5 and the Q8/`R` items below are now
   done or resolved — see §10; lever 5 is confirmed worthwhile because aggregate
   throughput does not plateau, §10.3.)*

**Not worth pursuing:** `lm_head` is already at 97% of roofline, and the dequant +
`gather_mm` prefill route is a measured loss (§3).

### 9.1 Ledger correction: cache traffic (2026-07-30, later)

The §1 ledger counts **weights only**. Decode also reads *and writes* the GDN
recurrent state every step — 30 GDN layers × 32 v-heads × 128 × 128 × 4 B =
62.9 MB, plus conv state ≈ **64.4 MB/seq in f32**, touched twice per step — and
20.5 KB per context token of KV across the 10 full-attention layers.

| | bytes/token | ceiling @ 101 GB/s |
|---|---|---|
| weights only (§1) | 2.506 GB | 40.3 |
| **+ GDN state traffic (short ctx)** | **2.635 GB** | **38.3** |
| + KV @ 4k ctx | 2.719 GB | 37.1 |

So measured bs1 is **72% of the honest short-ctx ceiling**, not 69%, and the
§1.1 "attn + GDN" bucket is at **77%** of roofline rather than 70% — its bytes
were undercounted. Two consequences: GDN-state fp16 (lever 2) is a *bandwidth*
lever and not merely a capacity one (at B=16 the state is 2.06 GB of a 9.1 GB
step, larger than the entire dense weight budget), and every ceiling quoted
against 40.3 should be read against 38.3.

---

## 10. Cheap-tier levers landed (2026-07-30)

Three levers from §9 that were XS-effort, plus a determinism fix the measurement
work forced out (§10.5). All numbers are paired same-session A/B, medians over
repeats, with the position-bias control described in §10.4.

**Reading the absolute numbers.** This session's bs1 figures (24.1 for the old
default, 25.1–25.7 for the new) sit below §0's 27.7, which was measured in an
earlier session on mlx 0.31 rather than 0.32. Absolute decode rates on this box
move with session/thermal state by several percent, so **only the paired
same-session deltas in this section are claims**; the 25.12 in §10.1 is not a
regression from 27.7 and the two should not be differenced.

### 10.1 Q8 group size 64 → 128 (default changed)

`pack_q8` writes the escha **per-output-channel** scale into every group of a
row, so the group size cannot change a represented value — only how many copies
of that constant are stored. scales+biases are f32 = 8 bytes/group: at group 64
that is **12.5% metadata** on top of the 1 byte/weight payload, at 128 it is
6.25%. Across the 2.16 GB of Q8 streams read per token that is **−120 MB/token**
for provably zero numerical change. All 212 int8 tensors in the export have
K ∈ {512, 2048, 4096}, so 128 (MLX's maximum) always applies; `quant.fit_group`
degrades to 64/32 for exports with odd K.

3 paired repeats, bs1 ISL=128, **no overlap between arms**:

| | group 64 | group 128 |
|---|---|---|
| decode tok/s | 24.15 / 24.33 / 23.83 | **25.01 / 25.21 / 25.14** |
| mean | 24.10 | **25.12 (+4.2%)** |
| resident | 11.55 GiB | **11.41 GiB** |
| peak (ISL 128 / 512) | 12.17 / 12.58 GiB | **12.03 / 12.44 GiB** |

*(Units corrected 2026-08-09: these values came from `baseline.py`, whose `_gb`
divided by 1024³ while printing "GB". 11.41 GiB = 12.25 decimal GB — the same
residency `head_to_head.py` reports. `_gb` now divides by 10⁹ like every other
harness.)*

Predicted +4.6% from the byte ledger, measured **+4.2%** — the closest
prediction-to-measurement agreement in this campaign. Correctness battery
(Paris / 391 / coherent thinking-mode haiku) green. Gate:
`test_q8_group_size_is_numerically_free` asserts group 32/64/128 outputs are
bit-identical, so if MLX ever makes the dequant group-dependent this fails loudly
instead of silently shifting numerics.

### 10.2 Rows-per-group (`R`) at decode row counts

`_blocked_R` was tuned on prefill row counts. Decode is a different
distribution: m = 8B rows drawn from E=256 gives only
E·(1−(1−1/E)^m) distinct experts, so rows/expert is 1.13 at m=64 and 1.93 at
m=384. Grouping cuts stream reads to (distinct/m) but pads partial groups up to
R, inflating row work — and the kernel is ~53% of roofline at B=16, neither
purely bandwidth- nor ALU-bound, so which side wins is a measurement.

In-model whole-step decode, median of 3:

| B | m | rows/expert | R=1 | R=2 | R=3 | R=4 | old policy | new |
|---|---|---|---|---|---|---|---|---|
| 8 | 64 | 1.13 | **62.4** | −3.5% | −9.4% | −13.2% | 1 ✓ | 1 |
| 16 | 128 | 1.27 | 90.9 | **+5.9%** | −3.8% | — | 1 ✗ | **2** |
| 24 | 192 | 1.42 | 106.9 | **+3.7%** | +1.9% | — | 1 ✗ | **2** |
| 32 | 256 | 1.58 | 108.4 | **+13.1%** | +13.1% | **−5.8%** | 4 ✗✗ | **2** |
| 48 | 384 | 1.93 | 108.7 | +14.7% | **+17.7%** | — | 4 ✗ | **3** |

Note m=256: the old policy picked **R=4, the worst of the four** — 16.7% below
R=2 (+13.1% vs −5.8%). That threshold came from an isolated-kernel table where R=4 at M=256
measured 1.28×; in-model the `build_groups` + padding cost eats it. **Kernel
microbenchmarks do not settle this question; the whole-step number does.**

Shipped-policy confirmation vs R=1, position-bias corrected: **+3.8% at B=16,
+14.7% at B=32, ~+16% at B=48**, R=1 correctly retained at B=8. The m ≥ 1024
bands are unchanged and explicitly *not* re-measured (see §10.4).

### 10.3 Wired limit — two knobs, and BOTH are needed above ~18 GB

There are two separate controls and they were conflated on the first pass here:

1. `sudo sysctl iogpu.wired_limit_mb=N` — raises the **ceiling**
   (`max_recommended_working_set_size`: 19.07 → 22.02 GB at N=21000).
2. `mx.set_wired_limit(bytes)` — how much MLX actually **wires**. Defaults to
   **0**: MLX wires nothing. Capped by (1).

**Below the ceiling, knob 2 is a wash** — at a 16.5 GB peak nothing was being
evicted, so wiring 18 GB changed nothing (B=16 92.07 → 91.96, B=48
124.53 → 126.49, both inside a 3–4% spread).

**Above it, knob 2 is the whole story.** With the sysctl raised to 21000 but
nothing wired, B=80 at a 19.28 GB peak collapses:

| B=80, peak 19.28 GB | tok/s | run-to-run spread |
|---|---|---|
| sysctl raised, nothing wired | **5.92** | **335%** (thrashing) |
| sysctl raised + `set_wired_limit(20 GB)` | **136.61** | 0.7% |

**A 23× cliff, with no error message** — just a very slow model. Raising the
sysctl alone does *not* fix it; it only permits the call that does. This is why
`loader.apply_wired_limit()` / `ESCHA_MLX_WIRED_GB` now exists, and why running
unwired logs the cap at load time.

**But aggregate throughput plateaus at B=64, so none of this buys throughput:**

| B | m | rows/expert | policy R | tok/s | peak GB |
|---|---|---|---|---|---|
| 16 | 128 | 1.27 | 2 | 92.6 | 13.67 |
| 32 | 256 | 1.58 | 2 | 125.6 | 15.09 |
| 48 | 384 | 1.93 | 3 | 122.4–124.6 | 16.49 |
| **64** | 512 | 2.31 | 3 | **135.4–140.2** | **17.89** |
| 80 | 640 | 2.72 | 3 | 136.6 *(wired)* | 19.28 |

B=64 → B=80 is **flat** (140.2 → 136.6, within spread) for +1.4 GB. So **B=64 at
17.89 GB is the operating point**, it fits under the *original* 19.07 GB cap, and
the sysctl is worth having for headroom and long-context room at a given B —
**not** for more tokens per second. The earlier reading of this table
("aggregate has not plateaued, so the sysctl is justified") was drawn from a
B=32→64 trend that does not continue; B=96 was not retested with wiring because
there is no throughput upside left to find.

### 10.4 Two measurement traps this tier surfaced

**Greedy decode here is run-to-run nondeterministic — do not A/B by diffing
tokens.** The first `R` sweep reported every R>1 as divergent. The control
(R=1 vs R=1, same seed, same prompt) diverged too, which invalidated the gate
rather than the kernel. Cause: `_expert_path` ends in
`y.at[row_token].add(contrib)`, an **atomic f32 scatter-add with top_k=8
duplicate indices per token**, so per-token summation order varies between runs;
over 40 layers that is enough to flip argmax on near-ties. This is a known trap
class with atomic scatter-adds, and it reappeared here. The gate is now an in-process bit-identity check on the GEMM output
itself (`check_gemm_equiv`), which is the actual kernel contract and holds
exactly — 16/16 configs bit-identical.

This is now **fixed** — see §10.5.

**Sequential configs in one process carry a ~1–2% position bias**, and one run
showed 7.7% for two configs that were computationally identical. Any A/B
here needs an A/B/A control; the measured drift over three configs was −2.2%
(B=8), −0.9% (B=16), −1.5% (B=32). The prefill `R` sweep was additionally
swamped by cold-kernel time (`--decode-steps 4` gave 15.77 vs 6.40 tok/s for
identical configs) and was **discarded rather than used** — hence the untouched
m ≥ 1024 bands above.

### 10.5 The runtime is now bit-reproducible (determinism fix)

The routed-expert tail is a fixed-order segmented reduction instead of an atomic
scatter:

```python
# was: y = mx.zeros((t, H), mx.float32).at[row_token].add(contrib)
y = contrib.reshape(t, self.top_k, -1).sum(axis=1)
```

`_rows` lays rows out token-major, so token *i*'s top_k contributions are
rows [i·top_k, (i+1)·top_k) — contiguous — and the reduction adds **exactly the
same addends**, just in a fixed order.

Isolated control at decode magnitudes (T=512, top_k=8, H=2048, 40 evaluations):

| primitive | runs differing from run 0 | max abs delta |
|---|---|---|
| `.at[].add()` f32, duplicate indices | **39 / 39** | 1.47e-03 *per block* |
| segmented `reshape + sum` | **0 / 39** | 0.0 |

1.47e-03 per block, compounded over 40 layers, is what was flipping argmax.

End-to-end on the real model — the exact scenario that diverged before (same
prompt, same seed, greedy, repeated):

- 5/5 repetitions produced **identical token sequences**
- 5/5 produced an **identical raw-logit tensor hash** (`b88aa376…`), which is
  strictly stronger than argmax agreement
- the same hash reproduced in a **separate process**

Cost: none. bs1 25.66 tok/s (vs 25.12 pre-fix), B=16 92.55 / B=32 125.58 /
B=64 136.20 — all inside run-to-run spread. Correctness battery green.

It is **not** bit-identical to pre-fix output (f32 addition is not associative,
so reordering the sum changes the last bits); it is within f32 rounding and
gated both ways. Five gates in `tests/test_moe_determinism.py`, including
`test_rows_layout_is_token_major` (the reshape is only valid while rows are
token-major — a switch to slot-major would silently sum across tokens with no
error) and `test_no_float_scatter_add_remains`, which fails on any new
non-int32 `.at[].add()` anywhere in the package. The three surviving scatters in
`build_groups` are int32, where atomic ordering cannot change the result.

Why this was worth doing beyond reproducibility: with nondeterministic output no
kernel A/B can be settled by comparing outputs (it is what invalidated the first
`R` sweep), and every eval number silently carries a run-to-run term.

---

## 11. Three ranked levers, all measured NEGATIVE (2026-07-30)

§9's top-ranked lever (split-K), plus `mx.compile` glue fusion and
shuffle-broadcast fetch, were implemented, gated correct, and measured
in-model. **All three are rejected on evidence.** The kernels are kept behind
flags (default off) because two of the three arguments are hardware-dependent;
the compile path was removed outright.

Method: `bench/sweep_kernel_variants.py` toggles each lever inside ONE loaded
model (env is read per call), medians of 5 × 48 steps, with `base` measured
first and last as a drift control. The first pass at 3 repeats produced 5–8%
spreads and inconsistent signs — worthless; the numbers below are from the
tight pass (spreads 0.1–1.5%, drift control −0.4%).

### 11.1 Split-K on the trellis GEMV — rejected (−1.3 to −1.8%)

| B=1 | tok/s | | B=4 | tok/s |
|---|---|---|---|---|
| S=1 (sequential) | **26.76** | | S=1 | **53.29** |
| S=4 | 26.40 (−1.3%) | | S=4 | 52.82 (−0.9%) |
| S=policy (8/4) | 26.28 (−1.8%) | | S=policy | 52.53 (−1.4%) |

The premise — bs1 launches only (OC/128)·M = 64 threadgroups for gate_up, each
on a 128-iteration serial `kt` chain, so the GPU starves — **does not hold on a
10-core M4**: 64 threadgroups is 512 simdgroups, already enough to saturate it.
The extra kernel launch plus the S×M×OC partial write+read costs more than the
shortened chains save.

Two things I got wrong in the ranking, worth recording:

1. **The applicable window is far narrower than claimed.** `split_k_for` returns
   1 for gate_up once m ≥ 64 (B ≥ 8), and from m ≥ 128 (B ≥ 16) the MoE takes the
   row-blocked GEMM path so `moe_gemv` is not called at all. Split-K could only
   ever have helped **B ≤ 4**. My first measurement pass "tested" it at B=8/16 and
   got −0.2% — because it was comparing *identical code*.
2. The predicted "bs1 → 31–32 tok/s (~80% of ceiling)" was never plausible from
   the ablation: even a 2× faster trellis is +15%, and the trellis did not get
   faster.

**Kept, default off** (`ESCHA_MLX_SPLITK=auto` re-enables): the starvation
argument is real on wider GPUs — 64 threadgroups is 6.4/core here but 1.6/core on
a 40-core M4 Max and 0.8/core on an 80-core M3 Ultra. This is the first thing to
re-measure there.

### 11.2 Shuffle-broadcast the code tile — rejected (−3.2 to −6.6%)

| | B=1 | B=4 |
|---|---|---|
| load (2 redundant loads/lane) | **26.76** | **53.29** |
| shuffle (1 load + 2 `simd_shuffle`) | 25.94 (−3.2%) | 49.99 (−6.6%) |

Consistent in sign at every batch size and every split factor. The premise was
that 64 load-issue slots for 16 distinct words (K2) is 4× redundant waste. It
isn't: those loads all hit one cache line and are nearly free, while
`simd_shuffle` puts ALU latency directly on the `kt` dependency chain and idles
lanes ≥ wpt during the cooperative load. Bit-identical (gated, 41 cases), so
this is purely a speed decision. `ESCHA_MLX_FETCH=shuffle` re-enables.

### 11.3 `mx.compile` on the MoE glue — REMOVED, not just disabled

Throughput: a wash (+0.3% at B=1, −1.0% at B=16, inside a ±0.6% drift control).
That alone would justify dropping it. What justifies *removing* it:

**Enabling compile changed 979,981 of 993,280 model logits, max |Δ| 0.66 on
logits of magnitude ≤11.84 — 5.6% relative.** Not a rounding effect.
Reproduced across separate processes (`compile=off` gives logit hash
`b88aa376…`, matching the pre-compile baseline exactly; `compile=safe` gives
`e95b5943…`). Greedy argmax happened to be unchanged on the probe, which is
exactly how something like this ships unnoticed.

The diagnosis is incomplete and that is the point: the compiled chain
(`_c_swiglu`) is **bit-identical in isolation** — 15 shape/seed combinations,
contiguous *and* strided-slice inputs, and fusing the following op into the same
graph still matches. So the divergence is not in the chain; it is in how the
compile boundary changes MLX's treatment of the surrounding 40-layer graph. A
path that is provably exact in isolation and 5.6% off in situ is not something
to ship behind a flag, so the four `mx.compile` wrappers were deleted.

Also measured, for the record: fusing the *inexact* chains too
(`ESCHA_MLX_COMPILE=all`) was +1.5–2.5% — the only positive number in this
section, and not worth 5.6% logit drift.

**Gate lesson.** `test_compile_glue.py` passed on every isolated chain while the
assembled model diverged. Unit-gating a fused kernel against its unfused twin is
necessary but *not sufficient*: the end-to-end logit hash is what caught this,
and it is now the check that matters (`det_check`-style comparison against a
known-good hash).

### 11.4 Where this leaves the ranking

The §9 ranking was ordered by expected lever size and generality. Measured, the
top-ranked item is worth −1.8% and the two "free" micro-optimizations are −3 to
−7% and actively harmful respectively. The bandwidth-bound analysis in §1.1 was
right about *where* the time goes (trellis at 27% of roofline) and wrong about
*why* — it is not threadgroup starvation, so adding parallelism does not help.
What remained untested from the ranking: **GDN-state fp16** — now done and
POSITIVE (§12) — and **speculative decoding**, the only remaining item that
changes bytes/token rather than chasing efficiency inside a roofline we are
already at 72% of.

---

## 12. GDN recurrent state in fp16 — the one that worked (2026-07-30)

**Result: +10.6% at B=64, −2.2 GB peak, and the concurrency ceiling moves from
B=64 to B=128 → aggregate 140 → 167.5 tok/s (+20%).** Default changed to fp16;
`ESCHA_MLX_GDN_STATE=fp32` restores the previous numerics exactly.

### 12.1 Why it is a bandwidth lever

§9.1's ledger correction: the GDN state is read *and written* every decode step —
30 layers × 32 v-heads × 128 × 128 × 4 B = **62.9 MB/seq in f32**. At B=16 that
is 2.06 GB of a 9.1 GB step, larger than the entire dense weight budget.

Where the precision actually goes matters, and it is good news. mlx-lm's
`gated_delta_step` kernel loads the state into `float` registers, runs the whole
T-token recurrence in f32, and casts only on the final store. So the storage
dtype costs **one rounding per kernel call, not per timestep** — prefill is
nearly free (one rounding per 256-token chunk) and decode is the worst case
(T=1 → one rounding per token) on a geometrically-decaying accumulator.

### 12.2 Numerics — and the measurement trap

Teacher-forced (both configurations fed an identical token sequence, so
trajectory divergence cannot masquerade as drift), 384 decode steps after a
512-token prefill, forced continuation = the f32 model's own greedy output:

| | top-1 agreement | KL mean | rel mean | drift first→last quarter |
|---|---|---|---|---|
| fp16 | **99.74%** | 2.78e-4 | 1.09e-2 | 2.5e-4 → 1.9e-4 (**flat**) |
| bf16 | 99.48% | 4.78e-4 | 1.30e-2 | flat |

fp16 beats bf16 on every metric — 10-bit mantissa vs 8, and the state's range
never needed bf16's exponent. The drift is **flat, not accumulating**: the delta
rule's self-correction (`delta = (v − kv_mem)·beta` drives the readout back
toward v) bounds the error instead of letting it compound.

**Trap: measure on real text.** The first pass teacher-forced on *random* token
ids and reported 94.0% top-1 agreement for the identical configuration — random
context puts the model off-distribution where logits are flat and near-ties
abound. On real text with an on-distribution continuation the same setup gives
99.74%. A 5.7-point swing purely from the probe distribution; the random-context
number would have killed a good lever.

### 12.3 Is that drift acceptable? Calibrate against what already varies

Same prompt, same fixed continuation, 192 decode steps:

| perturbation | max abs | rel mean | KL mean | top-1 |
|---|---|---|---|---|
| batch shape B=1 → B=16 (f32 state) | 7.076 | 1.07e-2 | 2.59e-4 | 100% |
| **fp16 state (B=1)** | **2.793** | **1.04e-2** | **2.52e-4** | **100%** |

**fp16 state perturbs logits slightly *less* than changing the batch size from 1
to 16 already does.** Batch-shape variation is unavoidable here — the
row-blocking factor R changes with row count (§10.2) — so fp16 state introduces
no new class of variation, only more of one the runtime already has.

Determinism is unaffected: identical shape + identical input → identical bits,
verified end-to-end (3/3 identical token sequences and logit hashes). Note this
*does* change bs1 output versus the f32 build, by the amount quantified above.

Replication invariance was re-verified while calibrating: within-batch row
spread is **exactly 0.0** at B ∈ {1,2,8,16,32} for identical rows.

### 12.4 Throughput and the new ceiling

| B | f32 tok/s | fp16 tok/s | Δ | f32 peak | fp16 peak |
|---|---|---|---|---|---|
| 1 | 26.99 | 27.22 | +0.9% | 12.34 | 12.31 |
| 16 | 93.61 | 96.09 | +2.7% | 13.67 | 13.11 |
| 32 | 122.52 | 134.80 | +10.0% | 15.09 | 13.98 |
| 64 | 137.31 | 151.83 | **+10.6%** | 17.89 | **15.68** |
| 96 | — | 162.45 | — | — | 17.37 |
| **128** | — | **167.54** | — | — | **18.98** |

The 2.2 GB freed at B=64 is what unlocks the rest: **B=128 at 18.98 GB fits
under even the ORIGINAL 19.07 GB cap**, so this ceiling does not depend on the
sysctl (though wiring is still required to run there — §10.3).

Aggregate throughput on this box across the campaign:
**93.7 (§4, B=16) → 140 (§10, B=64) → 167.5 (B=128)**, at bs1 27.2 tok/s.

### 12.5 Method note (see also §13 for prefill)

`bench/sweep_gdn_state.py`. The implementation is one subclass —
`GDNStateCache.__setitem__` casts slot 1 — rather than a patch of
`GatedDeltaNet.__call__`, because `gated_delta_kernel` already templates on
`state.dtype`. Slot 0 (the ~50 KB conv state) is deliberately left f32: it is
not the accumulated quantity and casting it risks the conv window for ~nothing.
`extract` is overridden because `ArraysCache.extract` hardcodes its own class,
which would silently revert to f32 on any server path that splits a batch.

---

## 13. Prefill: 207 → 243 tok/s (2026-07-30)

TTFT, not decode, is this runtime's user-visible weak spot — the 163 s p99 at
2048/128 C=16 (§4.1) is prefill queueing, not slow tokens. Prefill is also a
different regime from decode: a 256-token chunk reads the same ~2.2 GB of dense
weights as one decode step but does 256× the work, so its **bandwidth** roofline
is ~11,600 tok/s while we measured ~207. It is an efficiency problem, not a
bandwidth one.

**Attribution first** (`bench/prefill_profile.py`, ablated in-model with
data-dependent stubs, warmed per shape):

| chunk | full | routed experts | whole MoE | lm_head | wasted head |
|---|---|---|---|---|---|
| 128 | 201 tok/s | 66.6% | 69.6% | 7.4% | 7.0% |
| 256 | 206 tok/s | 65.7% | 68.3% | 6.9% | 6.8% |
| 512 | 220 tok/s | 61.3% | 64.6% | 5.8% | 7.2% |

The trellis GEMM is ~2/3 of prefill. Three changes landed, in order of certainty:

### 13.1 What landed

**1. `lm_head` on the last position only (+7%).** mlx-lm's `TextModel.__call__`
applies the head to the *whole* sequence, so a 256-token chunk computes a
[256, 2048] @ [2048, 248320] GEMM — 260 GFLOP, materialising 127 MB of logits —
and generation reads one row. `loader.LastPositionHead` wraps the head rather
than patching the model class (Python resolves `__call__` on the type, so an
instance override would not fire). Returns [B,1,V]; every generation path
indexes `[:, -1, :]` so it is transparent, but **per-position scoring needs
`ESCHA_MLX_LAST_LOGIT=0`** — gated and asserted in `tests/test_last_logit.py`.

**2. `R=12` at prefill row counts (+2–4%).** The m ≥ 2048 band was inherited
from a kernel microbenchmark and never measured in-model (§10.2). Measured:

| | R=16 | R=12 | R=8 | R=4 |
|---|---|---|---|---|
| S=256 (m=2048, 8 rows/expert) | — | **+3.8%** | +2.6% | −0.8% |
| S=512 (m=4096, 16 rows/expert) | — | **+1.8%** | −0.6% | −5.3% |

**3. `KT_BLOCK=4` — amortize the GEMM's barriers (+4.5% at chunk 256).** The
row-blocked GEMM ran *two* threadgroup barriers per `kt` iteration — 256 sync
points per threadgroup at TK=128, the same pattern that cost the per-row GEMV
1.15–1.78× before it went barrier-free. The GEMM cannot simply drop staging
(its `s_x` tile is genuinely shared across all 8 simdgroups, and the `rows_idx`
indirection makes a direct read a gather), so it stages KB tiles per barrier
pair instead. Bit-identical by construction and gated across KB ∈ {1,2,4,8,16}.

| | KB=1 | KB=2 | KB=4 | KB=8 | KB=16 |
|---|---|---|---|---|---|
| S=256 | 233.1 | +1.8% | **+4.5%** | +4.3% | +2.4% |
| S=512 | 243.0 | −0.3% | +0.3% | +0.3% | −1.7% |

**End to end: 243.2 tok/s @ ISL 512, 236.9 @ 2048, 231.9 @ 4096** (from ~207),
decode and correctness unchanged.

### 13.2 What the negative results tell us about the remaining 60%

I had ranked `simdgroup_matrix` tiling as the big prefill lever. **The R sweep
rules it out.** R=16 vs R=8 at m=2048 changes the MAC count by exactly 2× (same
group count, half the padded rows) and costs ~3%. So MACs are a few percent of
this kernel — replacing scalar FMAs with matrix intrinsics targets something
that is not the bottleneck.

What the expert GEMM is *not*, all measured:

* not MAC-bound (2× MACs ≈ 3%),
* not barrier-bound beyond ~4.5% (KB sweep),
* not bandwidth-bound — 9.4 GB of stream reads per chunk is 93 ms at the 101 GB/s
  roofline against ~800 ms measured, **8.6× off**,
* not decode-ALU-dominated at prefill row counts — row-blocking already
  amortizes each expert stream across R rows.

Which means the next step is **not another hypothesis**. Five have now been
tested this campaign (split-K, shuffle-fetch, mx.compile, R, KT_BLOCK); three
were negative and two worth ~4% each, and in every case the FLOP/byte arithmetic
predicted a much larger effect than materialized. The honest next move is a
Metal System Trace / Instruments capture to find where the cycles actually go,
rather than more reasoning from counters. Absent that, prefill stays ~243 tok/s.

---

## 14. Head-to-head vs stock mlx-lm 4-bit — we are 1.6× SLOWER at bs1 (2026-07-30)

Every published Apple-Silicon comparison is cross-report (different chips,
prompt lengths, thermal states, Air vs Pro, engine versions), so this ran both
builds through identical measurement code on this box, back to back:

  A  `EschaLabs/Qwen3.6-35B-A3B-Escha-W2`   (2-bit trellis experts + int8 dense)
  B  `mlx-community/Qwen3.6-35B-A3B-4bit`    (mlx-lm affine 4-bit, everything)

**Same base model.** Same mlx/mlx-lm, same GPU, same prefill chunk (256), same
warmup and `clear_cache` discipline, same 21 GB wired limit. A measured first
AND last as a drift control. `bench/head_to_head.py`.

| | escha W2 | stock 4-bit | winner |
|---|---|---|---|
| resident | **12.25 GB** | 19.51 GB | **A, 1.59× smaller** |
| peak @ ISL 2048 | **13.14 GB** | 20.05 GB | **A** |
| decode @ ISL 512 | 26.40 / 27.22 | **42.74** | **B, 1.59×** |
| decode @ ISL 2048 | 25.90 / 24.98 | **40.50** | **B, 1.59×** |
| prefill @ ISL 2048 (warm) | 236.2 / 238.3 | **338.9** | **B, 1.43×** |

Drift control: A's two runs differ by +1.8%/+0.9% (prefill) and +3.1%/−3.6%
(decode). Both B advantages are far outside that.

### 14.1 Why the "more quantized" model is slower — read the ledger

This is not a kernel deficiency. It is a **bit-allocation** result, and §1's
ledger predicts it exactly:

| stream | ours | bits |
|---|---|---|
| GDN dense | 1142.3 MB | **int8** |
| lm_head | 572.1 MB | **int8** |
| attn dense | 306.7 MB | **int8** |
| shared expert | 141.6 MB | **int8** |
| routed experts | 300.8 MB | 2/3-bit |
| router gate | 42.1 MB | fp16 |

**Only 12% of our per-token bytes are the 2-bit experts.** The other 86% is
int8. The stock build puts *everything* at 4-bit, so per token it reads
~1,730 MB against our 2,506 MB — **we read ~1.45× MORE bytes than the model with
"worse" quantization.** Measured decode ratio 1.56–1.59×, tracking the byte
ratio: both builds are bandwidth-bound, as §1 says they must be.

The root cause is that escha's allocation is tuned for **capacity**, not for
bs1 **bandwidth**. Across the whole 35B model the experts are ~93% of
parameters, so 2-bit experts are exactly right for fitting the model. But
decode reads only 8 of 256 experts per token, which collapses the experts to
12% of *traffic* while every dense byte is read every token. Optimising total
size and optimising per-token bytes are different problems on a MoE, and we
optimised the first.

### 14.2 What this says to do next

Quantizing the dense side (GDN / attn / lm_head / shared) from int8 to 4-bit
would take the ledger from 2,506 → ~1,489 MB/token, plus 129 MB of GDN state:

    ceiling  101 GB/s / 1.618 GB = 62 tok/s
    at our measured 72% of roofline  ~45 tok/s

i.e. **faster than the stock 4-bit build (40.5–42.7) while still ~5 GB smaller**,
because the experts stay at 2 bits. That single change is worth more than every
kernel lever in §11–§13 combined, yet it never made the §9 list —
badly underrated, because I ranked by kernel effort rather than by ledger share.

It is a model-side change needing an accuracy gate, not a runtime change. Note
the comparison is favourable on quality grounds too: the stock build already
runs *all* weights at 4-bit, so 4-bit dense + 2-bit experts should be no worse
than what it does, at far smaller footprint.

### 14.3 Where we still win

Memory, and everything downstream of it. At 12.25 GB resident against 19.51 GB
we have ~7 GB of headroom on this box, which is what makes **B=128 at 18.98 GB
and 167.5 tok/s aggregate** reachable (§12.4). The 4-bit build sits at 19.51 GB
resident before any KV or GDN state — it cannot run meaningful concurrency here
at all, and it only loads because `iogpu.wired_limit_mb` was raised (its 20.43 GB
on disk exceeds the stock 19.07 GB cap outright).

So the honest summary: **escha W2 is the only way to run this model with
headroom on a 24 GB Mac, and it serves far more concurrent load — but at bs1,
stock 4-bit is 1.6× faster, and the fix is on the quantization side, not in the
kernels.**

### 14.4 Harness caveat

`measure()` warms decode but NOT prefill, so the first ISL of each arm carries
Metal kernel specialisation. That is visible in B (216.9 cold → 338.9 warm)
and small in A (238.5 → 236.2). Only the **warm** prefill figures are
comparable; the cold ones are listed above for completeness but should not be
differenced.

---

## 15. Batched head-to-head — the 2-bit advantage is real, the KERNEL is losing it

§14 measured bs1 only and concluded "the fix is on the quantization side, not in
the kernels." **That conclusion was wrong for the case that matters.** Batched
decode, same harness, A/B/A (`bench/head_to_head.py --batches`):

| B | escha W2 | stock 4-bit | | ours MB/step | 4-bit MB/step | **byte adv** |
|---|---|---|---|---|---|---|
| 1 | 26.4 / 27.3 | **42.1** | 4-bit 1.57× | 2570 | 1860 | 0.72× |
| 2 | 45.3 / 45.4 | **65.2** | 4-bit 1.44× | 2935 | 2536 | 0.86× |
| 4 | 58.3 / 58.2 | **85.5** | 4-bit 1.47× | 3666 | 3837 | **1.05×** |
| 8 | 58.5 / 64.1 | **102.0** | 4-bit 1.66× | 5126 | 6253 | **1.22×** |
| 16 | **105.9 / 104.7** | **OOM** | — | 7028 | 10441 | **1.49×** |
| 32 | **151.0 / 146.6** | **OOM** | — | 10357 | 16872 | **1.63×** |

4-bit OOMs at B=16 (`Insufficient Memory`, peak 20.66 GB at B=8 against the
22.02 GB cap). Ours peaks at 13.74 GB at B=32.

### 15.1 We already read fewer bytes — and are still slower

From B=4 upward our per-step byte count is **below** the 4-bit build's, reaching
**1.63× fewer at B=32**. Yet at B=8 we read 1.22× fewer bytes and run 1.66×
slower. Converting both to achieved bandwidth against the 101 GB/s roofline:

| B | ours | 4-bit |
|---|---|---|
| 1 | 68% | 78% |
| 2 | 66% | 82% |
| 4 | 53% | 81% |
| 8 | **39%** | **80%** |
| 16 | 46% | — |
| 32 | 49% | — |

**`mx.gather_qmm` sustains ~80% of roofline at every batch size. Our trellis
kernel runs at 39–53%.** That is the entire deficit — a ~1.9× execution gap, not
a bit-allocation one. §14's bs1-only conclusion missed it because at bs1 the
dense int8 streams dominate and mask the kernel.

### 15.2 What closing the gap is worth

At parity efficiency (80% of roofline), with the CURRENT weights untouched:

| B | ours projected | 4-bit (if it fit) | advantage |
|---|---|---|---|
| 8 | 126 tok/s | 103 | 1.22× |
| 16 | 184 | 124 | 1.49× |
| 32 | **250** | 153 | **1.63×** |

So the throughput advantage the 2.42-bit codec is supposed to deliver is
**entirely reachable without touching the weights** — it needs the expert kernel
to convert bytes at the rate Apple's does. Bit allocation (§14.2) remains the
only bs1 lever, but bs1 is not where MoE serving lives.

### 15.3 The most promising borrow

`mx.fast` itself has nothing for us (`rms_norm`/`rope`/SDPA are already used by
mlx-lm's untouched layers). The relevant op is **`mx.gather_qmm`** in core, which
we cannot use: it is `mode="affine"`/mxfp4, and escha is a trellis code.

But mlx-lm's `SwitchGLU` reveals one structural difference worth testing. It
**physically permutes x** (`_gather_sort`) so each expert's rows are contiguous,
calls `gather_qmm(..., sorted_indices=True)`, then unsorts. We leave x in place
and gather *inside* the kernel:

    s_x[i] = xh[rows_idx[grp*R+rr] * IC + (kb+kk)*16 + cc];   // indirect

Pre-sorting would turn that into a contiguous, coalescable load, for the price
of one permute + one unpermute per leg. Given that staging is on the hot path at
every kt block, this is the leading candidate for the unexplained gap — and the
first one grounded in a working reference implementation rather than in FLOP
arithmetic, which has been wrong five times running (§11, §13.2).

Also noted: `metal_kernel(ensure_row_contiguous=True)` is the default, so MLX
inserts a contiguity copy for any non-contiguous input on every call — worth
auditing. And `compile_options={"math_mode": "fast"}` must NOT be used: it is
exactly what breaks the fp16 round-to-nearest-even decode contract.

### 15.4 The sorted-x borrow — implemented, gated, measured a WASH

§15.3 proposed copying mlx-lm's `_gather_sort`: physically permute x so a
group's rows are consecutive, letting the kernel compute `src_row0 + rr` instead
of loading `rows_idx[grp*R+rr]` and chasing an arbitrary row on every staged
element, TK times per group. The output write keeps the indirection (it happens
once per group, not TK times), so no un-permute is needed.

Implemented, gated **bit-identical** across m ∈ {64, 300, 2048} and both K
shapes, plus an invariant test that `src_row0`/`n_valid` describe exactly the
rows `rows_idx` names. Then measured, A/B/A:

| | sorted-x vs gather | drift control |
|---|---|---|
| prefill S=256 | −0.8% | −1.1% |
| prefill S=512 | −0.0% | −2.8% |
| decode B=16 | −1.7% | +0.2% |
| decode B=32 | −0.5% | +0.3% |

**No signal.** Every delta sits inside the A-vs-A drift band, and the signs are
inconsistent. Default OFF (`ESCHA_MLX_SORTX=1` enables); `build_groups` only
computes the extra index arrays when asked, so the default path is byte-for-byte
the code that shipped before.

That is **six kernel hypotheses, six non-results** (split-K, shuffle-fetch,
mx.compile, R at prefill ≈2-4%, KT_BLOCK ≈4%, sorted-x). The premise was wrong
every time, and this one had the strongest prior of the lot — a working
reference implementation that demonstrably sustains 80% of roofline.

**The conclusion to draw is about method, not about this change.** Reasoning
from counters (FLOPs, bytes, barrier counts, addressing modes) has now failed
six times on this kernel while correctly predicting every *bandwidth* result
elsewhere in this document. The gap between our 39-53% and gather_qmm's 80% is
real and worth ~1.63× at B=32, but it will not be found by inspection. It needs
a Metal System Trace / Instruments GPU capture — occupancy, register pressure,
memory-stall attribution — which is the one tool this campaign never used.

Measurement-hygiene note: after flipping the default, the A/B table still
labelled the unset arm "sorted-x (default)", silently making it an A/A
comparison that reported a 6.7% "win" at B=16 (spread 7.0%). Both arms now set
the flag explicitly. Relying on "unset == the variant I mean" breaks the moment
a default moves — and it also usefully pinned the B=16 noise floor at ~7%.

### 15.5 Profiling tooling: Instruments is NOT available on this box

`xcode-select -p` -> `/Library/Developer/CommandLineTools`. Only Command Line
Tools are installed, so `xctrace` refuses to run ("requires Xcode"), and PyPI is
unreachable from this venv, so `pyobjc-framework-Metal` cannot be installed to
query pipeline reflection (register-limited occupancy) either.

What DOES work: `mx.metal.start_capture` under `MTL_CAPTURE_ENABLED=1`. A capture
of the trellis GEMM at prefill shape (m=2048, R=12, K=2, three dispatches) is at

    /Users/<user>/escha_profiling/trellis.gputrace     (148 MB)

Reading it needs Xcode (`open` it, GPU debugger). The three things that would
settle §15.4: occupancy / max-threads-per-threadgroup for
`escha_moe_gemm_k2_r12_kb4` (register-limited?), shader-timeline stall reasons
(memory vs ALU vs sync), and limiter counters on the compute encoder.

`bench/gpu_busy.py` is the CLI substitute — GPU active residency via
powermetrics, which answers the first fork: saturated-but-slow (look inside the
kernel) vs partly-idle (look at dispatch/op overhead). Root-only, and it must be
run with the VENV interpreter, since sudo resets PATH to a system python that is
Python 2/3.6 on this box:

    sudo ~/.venv-escha-mlx/bin/python bench/gpu_busy.py --arm trellis
    sudo ~/.venv-escha-mlx/bin/python bench/gpu_busy.py --arm dense

`--arm dense` is the control: `mx.quantized_matmul`, measured at ~97% of roofline
(§1.1), so it calibrates what "healthy" reads as on this box rather than against
an assumed 100%.

### 15.6 GPU residency + power: the fork, answered

`bench/gpu_busy.py`, 16 samples each (run under sudo with the venv interpreter):

| | trellis GEMM | `mx.quantized_matmul` control |
|---|---|---|
| GPU active residency | **100.0%** | **100.0%** |
| GPU frequency | 1578 MHz | 1578 MHz |
| GPU power | **17.4 W** | **20.9 W** |
| achieved bandwidth | 48% of roofline | ~97% |

**Dispatch/launch overhead is ruled out.** The GPU is fully resident during our
kernel: nothing is being lost between kernels, to MLX op count, or to
command-buffer breaks. That eliminates a whole class of fixes -- and explains
retroactively why `mx.compile` (§11.3), which targets exactly that class, was a
wash.

Residency did NOT discriminate (both arms pin at 100%); the useful signal was
**power**. At identical clock and identical residency our kernel draws 17% less
power while delivering half the bandwidth. A kernel saturating the ALU on an
integer-heavy decode hash would draw MORE. Resident, clocked, not switching =
stalled. Caveat: part of that gap is simply lower DRAM activity, so power does
not prove stalling by itself -- but it does rule out ALU saturation.

Cumulative eliminations for the trellis GEMM: not MAC-bound (2x MACs = 3%), not
barrier-bound beyond ~4.5%, not bandwidth-bound (8.6x off roofline), not
dispatch-bound (100% residency), not ALU-saturated (low power at max clock).

### 15.7 Code-stream prefetch — hypothesis #7, also a wash

The surviving explanation was memory-level parallelism: `KT_BLOCK` batched the
**x** staging loads, but the **code** stream was still fetched one tile per kt
and consumed immediately -- one outstanding load per lane on a 128-iteration
chain. The two word offsets a lane needs depend on `lane` alone, never on kt, so
a whole KB block can be fetched up front as independent loads.

Implemented (`ESCHA_MLX_PREFETCH=1`), gated **bit-identical** across
KB ∈ {2,4,8} × both K shapes. Measured A/B/A:

| | prefetch vs per-kt | drift control |
|---|---|---|
| prefill S=256 | −0.0% | −1.1% |
| prefill S=512 | −0.4% | −2.3% |
| decode B=32 | −4.7% | −2.5% |

**No signal**, and the exact 0.0% at S=256 is the tell: the `k2` loop already
carries `#pragma clang loop unroll(full)`, so within a KB block the compiler
already sees KB independent `wp[i0]`/`wp[i1]` pairs and schedules them early.
The change made explicit something the Metal compiler was already doing. That
also means MLP *within* a KB block was never the constraint -- consistent with
`KT_BLOCK` itself buying only 4.5% at S=256 and 0% at S=512.

Default OFF; kept behind the flag with its gate.

### 15.8 Seven hypotheses, seven non-results — stop guessing

split-K, shuffle-fetch, mx.compile, R-at-prefill (~3%), KT_BLOCK (~4%),
sorted-x, code-prefetch. Two bought ~4%; five bought nothing. In every case the
reasoning was sound and the prediction was wrong.

Meanwhile the **bandwidth** model in this document has been right every single
time: it predicted the Q8-group win (+4.6% predicted, +4.2% measured), the fp16
GDN-state win, the entire 4-bit head-to-head (byte ratio 1.45x vs measured
1.56-1.59x), and the batch crossover. The ledger works; kernel micro-reasoning
does not.

So the remaining ~1.63x at B=32 is real, quantified, and **not reachable by
inspection**. It needs the capture at
`/Users/<user>/escha_profiling/trellis.gputrace` opened in Xcode's GPU
debugger -- specifically occupancy/register limits for
`escha_moe_gemm_k2_r12_kb4`, and per-instruction stall attribution. Everything
short of that has now been tried.

---

## 16. CORRECTION: the kernel was never the problem (2026-07-31, with Xcode)

§15 concluded "our trellis kernel runs at 39–53% of roofline where gather_qmm
sustains ~80%" and called closing that the main open work item. **That was
wrong**, and it invalidated the premise behind seven kernel hypotheses.

Those percentages were **whole-step** efficiency — modeled bytes divided by
whole-step wall time. They charged every surrounding operation (four Hadamard
transforms, four rin/rout gathers, the f32 intermediates) to the kernel.

### 16.1 Measured head-to-head, isolated, matched shapes

Same IC/OC/E, ours on the shipped `R` policy, `gather_qmm` with
`sorted_indices=True` (mlx-lm's own fast path):

| B | rows | trellis ms | gather_qmm ms | |
|---|---|---|---|---|
| 1 | 8 | **0.305** | 0.365 | ours 1.20× |
| 8 | 64 | **0.920** | 0.997 | ours 1.08× |
| 32 | 256 | **2.136** | 2.490 | ours 1.17× |
| 128 | 1024 | **5.000** | 6.402 | ours 1.28× |
| 256 | 2048 | 8.077 | **7.506** | theirs 1.08× |

**Our kernel is faster than Apple's fused quantized GEMM at every decode row
count**, and within 8% at prefill — while moving roughly half the bytes. There
was no 1.9× kernel deficit to find. Seven hypotheses failed because nothing was
there.

Hardware confirms it independently. A Swift probe compiling the kernel through
Metal at R ∈ {4,8,12,16} × KB ∈ {1,4}:

```
kernel           maxTPTG   simdW   tgMem(B)
gemm_r12_kb4        1024      32       1536
gemm_r16_kb4        1024      32       2048
```

`maxTotalThreadsPerThreadgroup` is 1024 — the hardware maximum — at every
variant. **Not register-limited.** Threadgroup memory is 1.5 KB of 32 KB
available. **Not shared-memory-limited.** Both were on the shortlist.

### 16.2 Where the time actually goes

In-model ablation, data-dependent stubs:

| component | prefill (S=256) | decode (B=32) |
|---|---|---|
| whole expert path | 68.1% | 61.4% |
| — trellis GEMM | 41.7% | 47.8% |
| — rin gather + scale | **15.8%** | **11.3%** |
| — Hadamard (RHT) | 3.3% | 8.9% |
| — rout gather + scale | 0.9% | 4.0% |

The transform pipeline is **~20% of both prefill and decode** — and it is work
the 4-bit build simply does not do. Affine quantization has no random Hadamard
transform and no per-expert input/output scale vectors. That, plus int8 dense
versus their 4-bit dense, fully explains the gap measured in §14/§15 without any
kernel being slow.

Why rin costs 15.8% while rout costs 0.9%: `rin` is f32 `[E, IC]`, so
`rin[row_expert]` materialises `[2048, 2048]` f32 = **16.8 MB per layer per leg**,
and the surrounding chain (`astype(f32)` → multiply → matmul → `*RS` → `astype(f16)`)
materialises several more. The arithmetic is trivial; the traffic is not.

### 16.3 Things tried against it, all measured

| change | result |
|---|---|
| `mx.hadamard_transform` (fast WHT) instead of matmul-by-H | **−0.5%** — wash. FLOPs say 37× less work; it is memory-bound, so the isolated op is only 1.06–1.37× faster. |
| `rin`/`rout` stored f16 | **−1.4%** (slower). Numerically **free** — logits bit-identical, the checkpoint values round-trip exactly through f16. |
| whole input transform in f16 | **+5.5%** prefill, but logits move 1.1% relative — it degrades the RHT, which is part of the codec contract. Not shipped. |
| folding `RS` into H | rejected on analysis: RS = 1/√128 is not a power of two, so it trades one rounding for 128. |

### 16.4 What this changes

The remaining opportunity is **~20% in the transform pipeline**, not in the
kernel, and the way to get it is fusion — one Metal kernel doing
gather → scale → Hadamard → scale → cast without materialising four `[m, IC]`
f32 intermediates. That is a real, bounded piece of work with a measured target.

`mx.compile` cannot do it (§11.3 measured it harmful), and MLX has no fused
primitive for a scaled Hadamard.

**Nine hypotheses, and the two that mattered were the ones the byte ledger
predicted.** The ledger has now been right on every prediction it made; kernel
micro-reasoning was wrong seven times and, it turns out, was aimed at a kernel
that was already ahead of Apple's.

---

## 17. Fused transform kernel — the win the ledger pointed at (+7–11%)

§16 located the remaining cost in the transform pipeline, not the kernel. This
acts on it: `msl.scaled_had` replaces the ~5-op chain

    rows.astype(f32) → * rin[row_expert] → H128 matmul → * RS → astype(f16)

with **one Metal kernel**. Each of those arrows materialised an `[m, IC]` f32
tensor — at m=2048, IC=2048 that is 16.8 MB apiece, ~150 MB of DRAM traffic per
layer per leg to perform 128 additions per output. The kernel loads 128 values
into threadgroup memory, scales, runs the transform there, and writes f16 once:
only the input and output reach DRAM.

The transform is the in-place radix-2 butterfly (7 stages), which computes the
same Sylvester-ordered unnormalised WHT as matmul-by-H in a different summation
order. **Arithmetic stays f32** — this is a reassociation, not a precision
reduction, which is what separates it from the all-f16 variant rejected in §16.3.

### 17.1 Measured

| | before | after | |
|---|---|---|---|
| prefill ISL 512 | 243.2 | **261.1** tok/s | +7.4% |
| prefill ISL 2048 | 236.9 | **258.7** tok/s | +9.2% |
| decode B=32 | 150.96 | 151.58 | +0.4% |
| decode B=64 | 151.83 | **162.47** | +7.0% |
| decode B=128 | 167.54 @ 18.98 GB | **185.63 @ 18.05 GB** | **+10.8%, and 0.9 GB smaller** |

Full batch curve after the change: B=1 26.96 · B=8 60.29 · B=16 108.45 ·
B=32 151.58 · B=64 162.47 · B=96 181.16 · B=128 **185.63**.

### 17.2 Correctness

Gated in `tests/test_fused_had.py` (4 shapes incl. the m=2048 prefill case):
within 2e-3 relative of the op chain, independently within 2e-3 of the NumPy
`ref.h128` reference, bit-reproducible across 16 evaluations, and <1% of f16
outputs differ (measured 0.12%).

It is **not** bit-identical to the op chain — no reduction reordering is. Whole
-model effect: logits move 1.27e-2 relative, the same magnitude that changing
batch size from 1 to 16 already causes (1.07e-2, §12.3), and one token in a
32-step × 8-row greedy probe changed. Decode remains **bit-reproducible**
(3/3 identical logit hashes). `ESCHA_MLX_FUSED_HAD=0` restores the op chain
exactly.

Correctness battery green (Paris / 391 / coherent thinking-mode haiku).

> **Current status:** this subsection records the original dense-matmul op chain.
> Production `moe.had_blocks` now uses `mx.hadamard_transform`, whose radix-2
> butterfly order matches the fused kernel. The current gate requires their final
> FP16 outputs to be bit exact under both default TF32 and `MLX_ENABLE_TF32=0`;
> the dense-matmul/TF32 mismatch is retained only as historical diagnosis in
> [PERFORMANCE.md](PERFORMANCE.md#m5-pro-resolved-issue-dense-matmul-test-oracle-and-tf32).

### 17.3 Why it was worth only 7–11% and not the 20% projected

The ablation said the transform pipeline was ~20% of both prefill and decode, and
fusing recovered roughly half of it. The residue is the parts fusion cannot
remove: the `xf[row_token]` gather feeding it, the output-side `rout` scale
(0.9%/4.0%), the SwiGLU, and the f16 casts between legs. Predicting 20% and
getting 7–11% is the same overprediction pattern as every other estimate in this
campaign — but this time in the right direction, because the prediction came from
a **measured ablation** rather than from counting FLOPs.

---

## 18. Served ISL/OSL grid, re-measured after the fused kernel

§4.1's grid predated §17. Re-run through the OpenAI server, all 12 points,
0 errors, OSL hit rate 1.00 everywhere (`bench/results/m4-base-24gb/grid_fused.json`):

| ISL | OSL | C | TTFT p50 | TTFT p99 | TPOT ms | output tok/s | total tok/s | vs §4.1 |
|---|---|---|---|---|---|---|---|---|
| 128 | 128 | 1 | 0.81 | 0.9 | 36.4 | 23.5 | 49.4 | +12% |
| 128 | 128 | 8 | 4.43 | 6.3 | 136.0 | 47.3 | 99.1 | +12% |
| 128 | 128 | 16 | 3.87 | 13.4 | 204.2 | 65.3 | 136.8 | +26% |
| 128 | 1024 | 1 | 0.85 | 0.9 | 35.6 | 27.4 | 31.2 | +13% |
| 128 | 1024 | 8 | 4.48 | 6.4 | 130.9 | 59.7 | 67.9 | +1% |
| 128 | 1024 | 16 | 3.99 | 13.5 | 155.3 | **99.5** | 113.2 | +21% |
| 1000 | 1000 | 1 | 4.08 | 4.2 | 36.0 | 24.8 | 49.9 | +13% |
| 1000 | 1000 | 8 | 10.37 | 30.9 | 147.7 | 50.4 | 101.5 | +5% |
| 1000 | 1000 | 16 | 11.10 | 64.0 | 201.7 | 74.3 | 149.5 | +24% |
| 2048 | 128 | 1 | 8.11 | 8.4 | 36.2 | 10.0 | 171.5 | +20% |
| 2048 | 128 | 8 | 18.13 | 61.6 | 449.9 | 13.5 | 230.3 | +24% |
| 2048 | 128 | 16 | 19.29 | 127.3 | 933.9 | 14.3 | **244.5** | +28% |

**Peak served output 81.9 → 99.5 tok/s. Peak total 191.4 → 244.5.**

The gain tracks how many expert rows are in flight, exactly as §17 predicts:
~12–13% where decode dominates at low concurrency, +21–28% wherever the batch is
wide, and one flat row (128/1024 C=8, 64 rows) where there is little transform
work to fuse.

**TTFT improved more than throughput** on prompt-heavy rows, because prefill is
what requests queue behind: 1000/1000 C=16 went 50.5 s → 11.1 s p50, and
2048/128 C=16 went 104.5 s → 19.3 s. That was the runtime's worst user-visible
number and it is now 5× better.

### 18.1 Two harness traps, both of which produced plausible wrong numbers

**Lazy model load inside the first measurement.** `/v1/models` answers ~2 s after
launch, but mlx-lm loads the weights on the first *generation*. The first attempt
therefore charged ~21 s of model load to the first grid point's TTFT p99 and
reported 12.2 tok/s where the truth is 23.5 — a plausible-looking halving. §4.1
never hit this because that server was already warm. `grid.sh` now issues an
explicit warmup request after the readiness probe.

**A kill pattern that matched its own command line.** The teardown used
`awk '/[g]rid\.sh/'` to find stale processes, but the same command line went on to
*relaunch* `grid.sh` — so the pattern matched the shell's own cmdline and killed
it (exit 144), silently skipping both the edit and the relaunch. Same class as
the classic `pkill -f` self-match trap, and the fix is the same: never put
the teardown and the launch in one invocation.
