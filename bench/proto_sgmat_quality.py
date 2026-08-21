"""Does the reassociated GEMM change what the model actually says?

Runs a long prompt (so prefill uses the R=16 path the matrix kernel replaces)
and reports, for the scalar and matrix kernels: the last-position logits, the
greedy token, the top-1 margin, and a greedy continuation.  Run it once per
config and diff the JSON -- the flag is latched at module construction, so the
two paths cannot coexist in one process.

    ESCHA_MLX_DENSE_MAT=0 python bench/proto_sgmat_quality.py --model DIR --out a.json
    ESCHA_MLX_DENSE_MAT=1 python bench/proto_sgmat_quality.py --model DIR --out b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from escha_mlx import loader, msl  # noqa: E402

PROMPT = (
    "The following is a technical discussion about computer architecture.\n\n"
    "Memory bandwidth and arithmetic throughput are the two walls that bound "
    "the performance of any matrix-multiplication kernel. On a unified-memory "
    "system the weights of a neural network must be streamed from DRAM once "
    "per forward pass, so a model whose weights exceed the last-level cache is "
    "bandwidth-bound at batch size one. As the batch grows, each weight that "
    "has been fetched is reused across more rows of the activation matrix, and "
    "the kernel eventually becomes limited by arithmetic instead. The crossover "
    "point is called the ridge point of the roofline model, and it is the ratio "
    "of peak arithmetic throughput to peak memory bandwidth. Quantized formats "
    "shift this balance: they reduce the bytes that must be streamed, but if "
    "decoding the compressed representation costs arithmetic instructions, they "
    "spend some of the saved bandwidth on additional compute. A format that "
    "requires several instructions per decoded weight can therefore turn a "
    "bandwidth-bound problem into a compute-bound one, which changes entirely "
    "which optimizations are worth pursuing.\n\n"
    "Question: In one paragraph, explain when a quantized format stops helping "
    "and what the engineer should measure to find out.\n\nAnswer:"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", type=int, default=96)
    args = ap.parse_args()

    model, tok = loader.load(args.model)
    ids = tok.encode(PROMPT)
    print(f"matrix path active: {msl.use_dense_mat()} | prompt tokens: {len(ids)}")

    x = mx.array([ids])
    logits = model(x)[0, -1].astype(mx.float32)
    mx.eval(logits)
    order = mx.argsort(-logits)
    top = [int(i) for i in order[:5].tolist()]
    lv = [float(logits[i]) for i in top]

    # Greedy continuation, feeding the whole prefix so prefill runs the R=16 path.
    cur = list(ids)
    out = []
    for _ in range(args.gen):
        lg = model(mx.array([cur]))[0, -1]
        mx.eval(lg)
        nxt = int(mx.argmax(lg).item())
        out.append(nxt)
        cur.append(nxt)
        if nxt == tok.eos_token_id:
            break

    rec = {
        "dense_mat": msl.use_dense_mat(),
        "n_prompt": len(ids),
        "top5_ids": top,
        "top5_logits": lv,
        "top1_margin": lv[0] - lv[1],
        "greedy_ids": out,
        "greedy_text": tok.decode(out),
    }
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(f"top1={top[0]} margin={rec['top1_margin']:.4f}")
    print("continuation:", rec["greedy_text"][:200].replace("\n", " "))


if __name__ == "__main__":
    main()
