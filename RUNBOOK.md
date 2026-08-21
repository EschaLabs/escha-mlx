# escha-mlx — bring-up runbook (first run on a new machine)

Written during the first hardware bring-up on an M4 MacBook (24 GB); the
sequence applies to any Apple Silicon Mac. Target: Qwen3.6-35B-A3B-Escha-W2
(12.32 GB) serving on a stock 24 GB machine. A 24 GB machine has a **~19 GB
default GPU working-set limit** (about 3/4 of physical RAM; measured 19.07 GB
via `max_recommended_working_set_size` on the bring-up M4 — the ~17.2 GB /
"2/3 of RAM" figure this runbook originally assumed was falsified during
bring-up, doc §1); the model needs ~14–15.5 GB at ctx ≤ 32 k and the loader
peaks ~13.5 GB — it fits, but close memory-hungry apps (browsers) for the
first session.

## 0. Setup (~15 min + model download)

```bash
# native arm64 python >= 3.10 (check: python3 -c "import platform; print(platform.machine())" -> arm64)
git clone https://github.com/EschaLabs/escha-mlx && cd escha-mlx
python3 -m venv ~/.venv-escha-mlx && source ~/.venv-escha-mlx/bin/activate
# EXACT pins for the bring-up session — this is the hardware-validated stack;
# do not stack an untested mlx/mlx-lm on top of untested kernels. (Quotes
# required: zsh globs [test].)
pip install -e '.[test]' 'mlx==0.32.*' 'mlx-lm==0.31.3'

# model (12.32 GB) — needs a recent huggingface_hub for `hf`; else use
# `huggingface-cli download` or copy from another box
hf download EschaLabs/Qwen3.6-35B-A3B-Escha-W2 --local-dir ~/models/escha-w2
export ESCHA_MODEL=~/models/escha-w2
```

## 1. P0 gates — run FIRST, in this order (~10 min)

```bash
# 1a. reference decode vs the committed goldens (no Metal needed — must pass anywhere)
pytest tests/test_ref_decode.py -v

# 1b. MLX tests that need no checkpoint (repack, w8a16 golden, synthetic MoE)
pytest tests/test_mlx_cpu.py -v -k "not ckpt_golden"

# 1c. THE METAL GATES — kernel bit-exactness, expert-stride, hash==LUT, bench
pytest tests/test_metal.py -v
python bench/p0_gates.py     # exception-isolated; capture FULL output

# 1d. layer-0 MoE block vs the committed golden (slow, ~minutes; needs $ESCHA_MODEL)
pytest tests/test_mlx_cpu.py -v -s -k ckpt_golden

# 1e. DENSE architectures (qwen3_5 — Qwen3.8-27B and siblings). Skip if you are
#     only bringing up the MoE model.
pytest tests/test_dense_linear.py -v          # incl. the Metal parity gates
ESCHA_MLX_SLOW_TESTS=1 ESCHA_DENSE_MODEL=<dir> pytest tests/test_dense_checkpoint.py -v
```

> **The dense Metal kernels have not been executed on hardware yet.** The dense
> path was developed on Linux with `mlx[cpu]`, where the Metal kernels do not
> run — everything else (the NumPy reference, the module against real shipped
> tensors, the loader against the real checkpoint, mixed-rate routing) is gated
> and green, but **step 1e is the first time the dense kernel sources meet the
> Metal compiler.** That is the same position the MoE runtime was in at its own
> bring-up, and it came out bit-exact on first contact; still, treat 1e as a
> genuine gate rather than a formality, and report a compile diagnostic or a
> value mismatch as a bug. `ESCHA_MLX_LINEAR=ops` runs the whole dense model
> through the NumPy oracle if you need to separate a kernel fault from a
> model fault.

**If a hash-decode gate fails but the LUT variant passes:** the Metal compiler
broke fp16 RNE on the hash path — `export ESCHA_MLX_LUT=1` for everything
below and report mlx/macOS versions.

**If kernels fail to COMPILE (Metal compiler error, not a value mismatch):**
capture the full compiler diagnostic; `ESCHA_MLX_MOE=ops` (MoE) or
`ESCHA_MLX_LINEAR=ops` (dense) still lets step 2 run as a correctness oracle
(slow, ~20 s/token).

**If the Q8 repack gate fails:** `export ESCHA_MLX_DENSE=fp16`
(+~1.9 GB resident; still fits at short ctx) and report.

## 2. First generation (greedy smoke)

**Load-failure branch first:** load should print `[escha-mlx] model loaded`
within ~2–3 min (disk-bound) at a ~13.5 GB peak. If the machine swap-storms
or sits >5 min, the LOADER is the bug (not Metal) — free RAM, watch Activity
Monitor memory pressure, retry; report where it hung. Steps 1a–1c never touch
the loader, so green gates don't clear it.

```bash
python -m escha_mlx.generate --model $ESCHA_MODEL --raw \
    --prompt "The capital of France is" --max-tokens 16
# want: " Paris" + coherent continuation (Linux CPU reference produced
# " Paris, a city"). Garbage => STOP, re-run step 1 gates.

python -m escha_mlx.generate --model $ESCHA_MODEL \
    --prompt "What is 17*23? Answer directly." --max-tokens 64
# want: 391

python -m escha_mlx.generate --model $ESCHA_MODEL \
    --prompt "Write a haiku about Tokyo." --max-tokens 400 --thinking
```

Correctness escalation if generation is garbage but step-1 gates pass:
`ESCHA_MLX_MOE=ops python -m escha_mlx.generate ...` — if THIS is coherent,
the bug is in the fused path; re-run `pytest tests/test_metal.py -k gemv`.

## 3. Serving

```bash
python -m escha_mlx.server --model $ESCHA_MODEL --port 8080
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is 17*23?"}],
  "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": false}}'
```

Known gap: no json_schema/structured output (an mlx-lm server limitation). Reasoning content + tool calls ARE parsed.

## 4. What to record (worth keeping — and worth sharing in an issue)

- `p0_gates.py` full output (gate pass/fail + DRAM-side GB/s).
- decode tok/s + prompt tok/s + peak mem from `escha_mlx.generate` at
  max-tokens 128 / prompt ~50 tok, and once at a ~2000-token prompt
  (a 2000-token prefill takes ~8 s on a base M4 — measured ~264 tok/s).
- chip + RAM + macOS + mlx versions (`sw_vers; python -c "import mlx.core as c; print(c.__version__)"`).
- Any test failure verbatim.

Expected decode floors at ≤4 k ctx: base M4 ~27 tok/s (measured), M4 Pro ~39–41
(measured, `bench/results/m4-pro-48gb/`), M4 Max ≥ 50 (bandwidth-scaled — the
M4 Pro measurement shows scaling is far from linear at bs1, so read scaled
floors loosely). First-session numbers below that are perf
work, not correctness failures — but garbage output or gate failures are BUGS:
capture and report.

## 5. Escape hatches (all composable)

| lever | effect |
|---|---|
| `ESCHA_MLX_LUT=1` | LUT decode instead of the hash (bit-exact by construction) |
| `ESCHA_MLX_MOE=ops` | numpy expert path (slow, correctness oracle) |
| `ESCHA_MLX_LINEAR=ops` | numpy path for a dense model's coded linears (slow, correctness oracle) |
| `ESCHA_MLX_DENSE_BLOCK_R=N` | pin rows-per-group for the dense row-blocked GEMM (1 = per-row kernel); default unmeasured on Metal |
| `ESCHA_MLX_DENSE=fp16` | fp16 dense instead of Q8 repack (+1.9 GB) |
| `ESCHA_MLX_Q8_GROUP=64` | revert to 64-wide Q8 groups (128 is the default) |
| `sudo sysctl iogpu.wired_limit_mb=N` | raise the GPU working-set cap if the fit is tight (e.g. 21000 on 24 GB raises 19.07 → 22.02 GB; also set `ESCHA_MLX_WIRED_GB` — the sysctl alone wires nothing, see INSTALL's 23× cliff) |
