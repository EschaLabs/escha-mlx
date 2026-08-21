# Contributing to escha-mlx

Thanks for looking under the hood. This project has one unusual property that shapes every
rule below: **the kernels are provably correct**, gated bit-identical against committed
reference vectors, and we intend to keep it that way. A contribution that keeps the
gates green is easy to merge no matter how deep it reaches; a contribution that quietly
changes numerics is unmergeable no matter how fast it is.

## Development setup

```bash
git clone https://github.com/EschaLabs/escha-mlx && cd escha-mlx
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[test]' 'mlx==0.32.*' 'mlx-lm==0.31.3'    # the validated stack
```

No Mac? You can still do real work: the codec reference, the repack, the loader and the
MoE ops path all run on Linux with `pip install 'mlx[cpu]==0.32.*'`. The entire runtime
was originally written blind on Linux this way and was bit-exact on first Metal contact —
the gates are what made that possible.

Run the gates before and after any change:

```bash
pytest tests/test_ref_decode.py -v          # NumPy codec vs the committed goldens — passes anywhere
pytest tests/test_mlx_cpu.py -v -k "not ckpt_golden"
pytest tests/test_metal.py -v               # Metal kernel bit-exactness (needs a Mac)
pytest tests/ -v                            # full suite; non-applicable tests self-skip
python bench/p0_gates.py                    # gates + DRAM-side GB/s
```

Tests self-skip by capability (`needs_mlx` / `needs_metal` / `needs_ckpt` markers), so
`pytest tests/` is always safe to run: on Linux the Metal tests skip, without a downloaded
checkpoint the end-to-end tests skip. Everything else runs from the small goldens
committed in `tests/data/`.

## The correctness bar

1. **Kernel and numerics changes must stay bit-identical** to the goldens and to the
   other kernel paths — `np.array_equal` on raw floats, not `allclose`. If your change is
   bit-identical, say so in the PR and point at the gate run that proves it.
2. **A deliberate numerics change** (e.g. reassociating a reduction for speed) must be
   (a) behind an env flag with the previous behavior recoverable, (b) documented in the
   tuning table in docs/INSTALL.md ("Tuning reference") with the measured effect, and (c) validated end-to-end, not just
   at the kernel boundary. `ESCHA_MLX_FUSED_HAD` is the template: default-on because it
   was measured and characterized, with `=0` restoring the exact op chain.
3. **Never loosen a gate to make a test pass.** If a gate fails on your machine with an
   unmodified tree, that is a bug report we badly want — file it with your macOS/mlx
   versions before doing anything else.

## The measurement bar

Perf numbers on Apple Silicon are easy to get wrong. Claims in a PR should follow the
practices that produced the numbers in `docs/`:

- **Paired, same-session A/B**: run baseline and candidate back to back, same machine,
  same power state; re-run the baseline at the end as a drift control.
- **Report the machine**: chip (base/Pro/Max), RAM, macOS, `mlx` version — a result from
  a 10-core M4 does not predict a 40-core M3 Ultra, and several committed defaults are
  known to be M4-shaped (`ESCHA_MLX_SPLITK`, `ESCHA_MLX_FETCH` are default-off *because*
  they lost on M4; they may win on wider GPUs, and showing that is a welcome PR).
- **Use the committed harnesses** (`bench/`) where one fits, and diff against the
  committed results under `bench/results/<machine>/` rather than re-deriving a baseline.
- **Negative results are contributions.** `docs/BRINGUP_AND_PERF.md` records seven
  consecutive failed optimization hypotheses on purpose: they stop the next person from
  spending a weekend on them. A PR that only adds a dated "measured X, it was a wash, here
  is the data" section to the docs is a good PR.

## Supporting a new chip (M1–M5, Pro/Max/Ultra)

One tree, one `main` — **never** a per-chip branch or a per-chip code folder. Chips
differ in what's *fastest*, not in what's *correct*, so chip support is data, not forks:

1. **Kernel strategies are flags, not copies.** Every strategy variant lives behind an
   existing env knob (`ESCHA_MLX_GEMV`, `ESCHA_MLX_SPLITK`, `ESCHA_MLX_FETCH`,
   `ESCHA_MLX_BLOCK_R`, `ESCHA_MLX_KT_BLOCK`, …) and is gated bit-identical like every
   other path. If your chip wants a genuinely new kernel shape, add it as one more gated
   strategy — selectable, never replacing.
2. **Results are folders.** Commit your paired runs as
   `bench/results/<chip>-<ram>/*.json` (pattern: `m4-base-24gb`) and add a section to
   `docs/PERFORMANCE.md` with the machine table filled in.
3. **Defaults move only on evidence.** Today's defaults are M4-base-tuned. When a second
   machine's numbers land, defaults graduate from per-knob constants to a small
   device-keyed table (selected via `mx.device_info()`, env always overriding) — that
   table is the only chip-conditional code we ever want.

## Adding a new model architecture

This repo is meant to grow one architecture at a time (MoE architectures like Kimi K3 /
GLM / DeepSeek next). The runtime is a codec engine plus **one plugin module per
architecture** — `escha_mlx/models/<model_type>.py`, resolved from the checkpoint's
`config.json` (contract: `escha_mlx/models/__init__.py`). Copy the closest existing
plugin as the template — `models/qwen3_5_moe.py` for a mixture of experts,
`models/qwen3_5.py` for a dense model:

1. **Skeleton from mlx-lm**: the model class, attention/SSM, caches and chat template come
   from `mlx_lm.models.<model_type>` untouched. If mlx-lm can't load the fp16 version of
   the architecture, fix that upstream first.
2. **Write the plugin**: a `CheckpointLoader` that builds the skeleton in `__init__`,
   maps the export's tensor names in `consume` (coded trios / Q8 pairs / fp16
   remainder), and installs modules + post-load quirks in `finalize`. Routing
   conventions live in the plugin's MoE block over the shared
   `escha_mlx.moe.EschaExperts` toolkit; a dense architecture installs
   `escha_mlx.dense.EschaLinear` per projection and writes no block at all — the
   kernels never fork per architecture. Loading stays streaming (per-tensor
   `safe_open`); never materialize the checkpoint twice. Note that a dense export
   groups its tensors by *leaf name*, not by module, so a linear cannot be installed
   the moment its group completes: convert each code stream on arrival
   (`dense.pack_code`) and assemble in `finalize`, or peak memory doubles.
3. **Goldens before Metal**: obtain per-layer golden inputs/outputs for the new export
   (open a model-request issue — we generate and supply goldens for published escha
   exports), commit them under `tests/data/<model_type>/` (they are small), and make the
   NumPy reference reproduce them bit-exactly. Only then wire anything new.
4. **Synthetic-checkpoint test**: extend `tests/test_models.py` with a tiny
   format-faithful export for your plugin, so CI runs your entire load-and-forward path
   on every PR without the real model.
5. **Register and gate**: add the module to `REGISTRY` in `escha_mlx/models/__init__.py`,
   and gate every kernel-touching path the way `tests/test_metal.py` does —
   bit-identical to the reference and to each other.

If you want an architecture but can't build all of this, open a
[model request issue](https://github.com/EschaLabs/escha-mlx/issues/new?template=new_model.yml) —
we can generate and supply the goldens for any published escha export.

## Code style

- `from __future__ import annotations`; lowercase generics (`list[str]`); `X | None`.
- No `print()` in library code — `logging.getLogger(__name__)`. CLIs may print.
- Comments state constraints the code can't show (memory contracts, numerics invariants,
  "this order is load-bearing"), not narration of what the next line does.
- Match the density and idiom of the file you're editing.

## Pull requests

- One change per PR. A perf change and a refactor are two PRs.
- Fill in the PR template — it is short and every box is load-bearing (gates, pairing,
  hardware).
- CI runs the Linux CPU gates and the macOS suite on every PR; both must pass.
- By contributing you agree your contributions are licensed under
  [Apache-2.0](LICENSE), the license of the project (inbound = outbound). Please
  sign off your commits (`git commit -s`, [DCO](https://developercertificate.org/)).

## Reporting bugs

Use the bug template. The single most useful thing you can attach is the output of the
gates (`pytest tests/ -v` and `python bench/p0_gates.py`) plus chip / RAM / macOS /
`mlx` versions — "which gate is the first to fail" localizes most bugs immediately.
