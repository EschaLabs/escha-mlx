"""Does an fp16/bf16 GDN recurrent state actually hold up over a long decode?

This is the R4 gate. The GDN state is a geometrically-decaying accumulator and
mlx-lm's kernel rounds it to the storage dtype once per call -- which at decode
(T=1) means once per TOKEN. An error injected at step t decays as prod(g) but is
re-injected every step, so the steady-state relative error is roughly
eps/(1-g), not eps. Whether that matters is an empirical question.

METHOD -- teacher forcing. Free-running greedy decode is useless here: the
moment two configurations pick different tokens their inputs differ, and every
later difference is trajectory divergence rather than numerical drift. So both
configurations are fed the IDENTICAL token sequence and their logits are
compared step by step. Reported per step:

    rel      max|logit_dt - logit_f32| / max|logit_f32|
    top1     fraction of positions where argmax still agrees
    kl       KL(softmax(f32) || softmax(dt)) in nats, the decision-relevant metric

Drift that grows with step index is the failure mode to look for; a flat curve
means the delta rule's self-correction is holding.

Also reports the actual byte saving and decode throughput, since the point of
the exercise is bandwidth.

    python bench/sweep_gdn_state.py --model <dir> \
        [--steps 512] [--prefill 512] [--batch 1] [--dtypes fp32,fp16,bf16]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

WARMUP = 4


def build_cache(model, dt):
    """make_prompt_cache, but with the GDN slot forced to `dt`."""
    from mlx_lm.models.cache import KVCache
    from escha_mlx.gdn_cache import GDNStateCache

    layers = model.language_model.model.layers
    return [GDNStateCache(size=2, dtype=dt) if l.is_linear else KVCache()
            for l in layers]


def teacher_forced_logits(model, dt, prompt_ids, forced_ids):
    """Prefill `prompt_ids`, then feed `forced_ids` one at a time.

    Returns [S, V] float32 logits, one row per forced token, plus the GDN state
    bytes actually resident.
    """
    cache = build_cache(model, dt)
    lg = model(prompt_ids, cache=cache)
    mx.eval(lg)
    out = []
    for i in range(forced_ids.shape[1]):
        lg = model(forced_ids[:, i:i + 1], cache=cache)
        mx.eval(lg)
        out.append(np.array(lg[:, -1, :].astype(mx.float32))[0])
    gdn_bytes = sum(c.nbytes for c in cache if hasattr(c, "gdn_dtype"))
    del cache
    mx.clear_cache()
    return np.stack(out), gdn_bytes


def timed_decode(model, dt, B, steps):
    cache = build_cache(model, dt)
    ids = mx.random.randint(1000, 60000, shape=(B, 16)).astype(mx.int32)
    lg = model(ids, cache=cache)
    mx.eval(lg)
    tok = mx.argmax(lg[:, -1, :], axis=-1)[:, None].astype(mx.int32)
    mx.eval(tok)
    mx.clear_cache()

    def step(t):
        l = model(t, cache=cache)
        return mx.argmax(l[:, -1, :], axis=-1)[:, None].astype(mx.int32)

    for _ in range(WARMUP):
        tok = step(tok)
        mx.eval(tok)
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(steps):
        tok = step(tok)
        mx.eval(tok)
    mx.synchronize()
    dt_s = (time.perf_counter() - t0) / steps
    peak = mx.get_peak_memory() / 1e9
    del cache
    mx.clear_cache()
    return B / dt_s, peak


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def compare(ref, got):
    scale = np.maximum(np.abs(ref).max(-1), 1e-6)
    rel = np.abs(ref - got).max(-1) / scale
    top1 = (ref.argmax(-1) == got.argmax(-1))
    p, q = softmax(ref.astype(np.float64)), softmax(got.astype(np.float64))
    kl = (p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum(-1)
    return rel, top1, kl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--prefill", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dtypes", default="fp32,fp16,bf16")
    ap.add_argument("--perf-batches", default="1,16")
    ap.add_argument("--random-context", action="store_true",
                    help="use random token ids instead of real text (see below)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from escha_mlx.gdn_cache import _DTYPES
    from escha_mlx.loader import load

    print(f"loading {args.model} ...")
    model, tokenizer = load(args.model)

    mx.random.seed(7)
    if args.random_context:
        # Kept as an option, but NOT the default: random ids put the model far
        # off-distribution, where logits are flat and near-ties are common, so
        # argmax agreement understates badly. Measured on this model: 94.0%
        # top-1 agreement on random context vs the real-text number below.
        prompt = mx.random.randint(1000, 60000, shape=(1, args.prefill)).astype(mx.int32)
        forced = mx.random.randint(1000, 60000, shape=(1, args.steps)).astype(mx.int32)
        mx.eval(prompt, forced)
    else:
        # Real text, and the forced continuation is the f32 model's OWN greedy
        # output, so every scored position is on-distribution.
        seed_text = (
            "The history of computing hardware spans several centuries. "
            "Explain, in detail and step by step, how memory bandwidth limits "
            "the decoding speed of a large language model, and why batching "
            "helps. Cover the arithmetic intensity of matrix-vector products, "
            "the role of cache hierarchies, and the difference between "
            "prefill and decode phases.\n\n")
        ids = tokenizer.encode(seed_text)
        while len(ids) < args.prefill:            # repeat to reach the length
            ids = ids + ids
        prompt = mx.array(np.array(ids[:args.prefill], dtype=np.int32))[None]
        mx.eval(prompt)
        print(f"  generating {args.steps} on-distribution tokens with f32 ...")
        cache = build_cache(model, mx.float32)
        lg = model(prompt, cache=cache)
        mx.eval(lg)
        tok = mx.argmax(lg[:, -1, :], axis=-1)[:, None].astype(mx.int32)
        seq = []
        for _ in range(args.steps):
            seq.append(tok)
            lg = model(tok, cache=cache)
            tok = mx.argmax(lg[:, -1, :], axis=-1)[:, None].astype(mx.int32)
            mx.eval(tok)
        forced = mx.concatenate(seq, axis=1)
        mx.eval(forced)
        del cache
        mx.clear_cache()

    names = [d.strip() for d in args.dtypes.split(",")]
    ref = None
    report = {"numerics": [], "perf": []}

    print(f"\nteacher-forced numerics: prefill {args.prefill}, {args.steps} steps")
    for name in names:
        dt = _DTYPES[name]
        lg, gdn_bytes = teacher_forced_logits(model, dt, prompt, forced)
        if ref is None:
            ref, ref_bytes = lg, gdn_bytes
            print(f"  {name:5s}  reference        GDN state {gdn_bytes/1e6:6.1f} MB")
            report["numerics"].append({"dtype": name, "gdn_mb": gdn_bytes / 1e6})
            continue
        rel, top1, kl = compare(ref, lg)
        # drift over time: first vs last quarter
        q = max(1, len(rel) // 4)
        print(f"  {name:5s}  GDN state {gdn_bytes/1e6:6.1f} MB "
              f"({100*(1-gdn_bytes/ref_bytes):.0f}% smaller)")
        print(f"         rel  mean {rel.mean():.2e}  max {rel.max():.2e}   "
              f"first{q} {rel[:q].mean():.2e} -> last{q} {rel[-q:].mean():.2e}")
        print(f"         KL   mean {kl.mean():.2e}  max {kl.max():.2e}   "
              f"first{q} {kl[:q].mean():.2e} -> last{q} {kl[-q:].mean():.2e}")
        print(f"         top-1 agreement {100*top1.mean():.2f}%  "
              f"(first divergence at step "
              f"{int(np.argmin(top1)) if not top1.all() else -1})")
        report["numerics"].append({
            "dtype": name, "gdn_mb": gdn_bytes / 1e6,
            "rel_mean": float(rel.mean()), "rel_max": float(rel.max()),
            "rel_first_q": float(rel[:q].mean()), "rel_last_q": float(rel[-q:].mean()),
            "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
            "kl_first_q": float(kl[:q].mean()), "kl_last_q": float(kl[-q:].mean()),
            "top1_agree": float(top1.mean()),
            "first_divergence": int(np.argmin(top1)) if not top1.all() else -1,
        })

    print("\nthroughput")
    for B in [int(b) for b in args.perf_batches.split(",")]:
        base = None
        for name in names:
            tps, peak = timed_decode(model, _DTYPES[name], B, 24)
            if base is None:
                base = tps
                d = ""
            else:
                d = f"{100*(tps-base)/base:+6.1f}%"
            print(f"  B={B:3d} {name:5s} {tps:7.2f} tok/s {d:>8s}  peak {peak:5.2f} GB")
            report["perf"].append({"batch": B, "dtype": name,
                                   "tok_s": round(tps, 2), "peak_gb": round(peak, 2)})

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
