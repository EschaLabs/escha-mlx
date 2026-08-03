"""Where does prefill time actually go?

Prefill is NOT bandwidth-bound the way decode is: a 256-token chunk reads the
same ~2.2 GB of dense weights as one decode step but does 256x the work, so the
bandwidth roofline sits around 11,600 tok/s while we measure ~200. That makes it
a compute/efficiency problem, and the first job is attribution rather than
optimisation -- three ranked levers in a row have now been measured negative
because their premise was wrong (doc §11).

Ablations, all measured INSIDE the real model on a real chunk:

  full            the chunk forward as it runs today
  last-logit      lm_head applied to the final position only, not all S
  no-experts      routed-expert path stubbed (data-dependent, so MLX cannot
                  dead-code-eliminate the upstream graph -- doc §5)
  no-moe          the whole MoE block stubbed
  no-lm_head      lm_head stubbed entirely (upper bound on the head's cost)

Every configuration is warmed per shape before timing: Metal specialises kernels
per shape and the first call otherwise lands in the measurement (an earlier
prefill sweep was ruined exactly this way, doc §10.4).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

WARMUP = 3
ITERS = 6


def time_chunk(fn, ids, iters=ITERS, warm=WARMUP):
    """Time a single-chunk forward. Fresh cache each call so every iteration
    does the same work (a persisting cache would grow the attention span)."""
    for _ in range(warm):
        mx.eval(fn(ids))
    mx.synchronize()
    mx.clear_cache()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(ids))
    mx.synchronize()
    dt = (time.perf_counter() - t0) / iters
    mx.clear_cache()
    return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunks", default="128,256,512,1024")
    ap.add_argument("--sweep-prefetch", action="store_true",
                    help="A/B/A the code-stream prefetch path")
    ap.add_argument("--sweep-sortx", action="store_true",
                    help="A/B/A the pre-sorted-x staging path")
    ap.add_argument("--sweep-kb", default=None,
                    help="comma list of ESCHA_MLX_KT_BLOCK values to sweep")
    ap.add_argument("--sweep-r", default=None,
                    help="comma list of R values to sweep instead of ablating")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from escha_mlx.loader import load
    from mlx_lm.models.cache import make_prompt_cache

    print(f"loading {args.model} ...")
    model, _ = load(args.model)
    lm = model.language_model
    layers = lm.model.layers
    V = lm.args.vocab_size

    rows = []
    for S in [int(c) for c in args.chunks.split(",")]:
        ids = mx.random.randint(1000, 60000, shape=(1, S)).astype(mx.int32)
        mx.eval(ids)
        print(f"\nchunk S={S}")

        def full(x):
            return model(x, cache=make_prompt_cache(model))

        if args.sweep_prefetch:
            import os
            base = None
            for tag in ("0", "1", "0"):     # A/B/A
                os.environ["ESCHA_MLX_PREFETCH"] = tag
                t = time_chunk(full, ids)
                lbl = "code prefetch" if tag == "1" else "per-kt fetch"
                if base is None:
                    base, d = t, ""
                else:
                    d = f"{100*(base-t)/base:+6.1f}%"
                print(f"  {lbl:16s} {t*1000:8.1f} ms  {S/t:8.1f} tok/s {d:>8s}")
                rows.append({"S": S, "prefetch": tag, "ms": t*1000, "tok_s": S/t})
            os.environ.pop("ESCHA_MLX_PREFETCH", None)
            continue

        if args.sweep_sortx:
            import os
            base = None
            for tag in ("1", "0", "1"):     # A/B/A: drift shows as A-vs-A gap
                os.environ["ESCHA_MLX_SORTX"] = tag
                t = time_chunk(full, ids)
                lbl = "sorted-x" if tag == "1" else "rows_idx gather"
                if base is None:
                    base, d = t, ""
                else:
                    d = f"{100*(base-t)/base:+6.1f}%"
                print(f"  {lbl:16s} {t*1000:8.1f} ms  {S/t:8.1f} tok/s {d:>8s}")
                rows.append({"S": S, "sortx": tag, "ms": t*1000, "tok_s": S/t})
            os.environ.pop("ESCHA_MLX_SORTX", None)
            continue

        if args.sweep_kb:
            import os
            base_kb = None
            for kb in [int(x) for x in args.sweep_kb.split(",")]:
                os.environ["ESCHA_MLX_KT_BLOCK"] = str(kb)
                t = time_chunk(full, ids)
                if base_kb is None:
                    base_kb, d = t, ""
                else:
                    d = f"{100*(base_kb-t)/base_kb:+6.1f}%"
                print(f"  KT_BLOCK={kb:3d}  {t*1000:8.1f} ms  {S/t:8.1f} tok/s {d:>8s}")
                rows.append({"S": S, "kt_block": kb, "ms": t*1000, "tok_s": S/t})
            os.environ.pop("ESCHA_MLX_KT_BLOCK", None)
            continue

        if args.sweep_r:
            # Rows/expert at prefill is m/E = 8S/256 = S/32, so a group of R
            # rows is only ~S/32 full: R above that is pure padding, and the
            # padded rows cost real MACs.
            m = 8 * S
            print(f"  m={m} rows over 256 experts ~= {m/256:.1f} rows/expert")
            base_r = None
            for r in [int(x) for x in args.sweep_r.split(",")]:
                for l in layers:
                    l.mlp._block_env = r
                t = time_chunk(full, ids)
                if base_r is None:
                    base_r, d = t, ""
                else:
                    d = f"{100*(base_r-t)/base_r:+6.1f}%"
                print(f"  R={r:3d}  {t*1000:8.1f} ms  {S/t:8.1f} tok/s {d:>8s}")
                rows.append({"S": S, "R": r, "ms": t * 1000, "tok_s": S / t})
            for l in layers:
                l.mlp._block_env = None
            continue

        def last_logit(x):
            cache = make_prompt_cache(model)
            h = lm.model(x, cache=cache)
            return lm.lm_head(h[:, -1:, :])

        base = time_chunk(full, ids)
        t_last = time_chunk(last_logit, ids)

        # --- stub routed experts -------------------------------------------
        saved = {}
        for i, l in enumerate(layers):
            saved[i] = l.mlp._expert_path
            l.mlp._expert_path = types.MethodType(
                lambda self, xf, i_, s_: xf * 0.5, l.mlp)
        t_noexp = time_chunk(full, ids)
        for i, l in enumerate(layers):
            l.mlp._expert_path = saved[i]

        # --- stub the whole MoE block ---------------------------------------
        orig = [l.mlp for l in layers]

        class _MoEStub:
            def __call__(self, x):
                return x * 0.5           # data-dependent: no DCE
        for l in layers:
            l.mlp = _MoEStub()
        t_nomoe = time_chunk(full, ids)
        for l, m_ in zip(layers, orig):
            l.mlp = m_

        # --- stub lm_head ----------------------------------------------------
        head = lm.lm_head

        class _HeadStub:
            def __call__(self, x):
                return mx.broadcast_to(mx.sum(x, axis=-1, keepdims=True),
                                       x.shape[:-1] + (V,)).astype(mx.float16)
        lm.lm_head = _HeadStub()
        t_nohead = time_chunk(full, ids)
        lm.lm_head = head

        def tps(t):
            return S / t

        print(f"  full              {base*1000:8.1f} ms   {tps(base):8.1f} tok/s")
        print(f"  last-logit only   {t_last*1000:8.1f} ms   {tps(t_last):8.1f} tok/s"
              f"   -> wasted head {100*(base-t_last)/base:5.1f}%")
        print(f"  minus experts     {t_noexp*1000:8.1f} ms   experts cost "
              f"{(base-t_noexp)*1000:7.1f} ms  ({100*(base-t_noexp)/base:4.1f}%)")
        print(f"  minus whole MoE   {t_nomoe*1000:8.1f} ms   MoE cost      "
              f"{(base-t_nomoe)*1000:7.1f} ms  ({100*(base-t_nomoe)/base:4.1f}%)")
        print(f"  minus lm_head     {t_nohead*1000:8.1f} ms   head cost     "
              f"{(base-t_nohead)*1000:7.1f} ms  ({100*(base-t_nohead)/base:4.1f}%)")
        rows.append({"S": S, "full_ms": base * 1000, "last_logit_ms": t_last * 1000,
                     "no_experts_ms": t_noexp * 1000, "no_moe_ms": t_nomoe * 1000,
                     "no_head_ms": t_nohead * 1000,
                     "full_tok_s": tps(base), "last_logit_tok_s": tps(t_last)})

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
