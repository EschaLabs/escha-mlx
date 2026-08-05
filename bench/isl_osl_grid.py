"""ISL/OSL serving benchmark for escha-mlx (NVIDIA-style grid).

Drives an OpenAI-compatible endpoint (escha_mlx.server) with the standard
input/output-sequence-length grid used for LLM inference benchmarking, sweeping
concurrency at each point, and reports the metrics NVIDIA's GenAI-Perf /
TensorRT-LLM perf-overview tables use:

  TTFT      time to first token (streaming, per request)
  TPOT/ITL  (e2e - TTFT) / (out_tokens - 1)   -- inter-token latency
  per-user  out_tokens / e2e                  -- what one client perceives
  aggregate sum(out_tokens) / wall            -- system output throughput
  total     sum(in+out) / wall                -- incl. prefill work

Prompts are built to an EXACT token length with the model's own tokenizer and
are UNIQUE per request by default, so the server's prefix cache cannot inflate
results.  `--shared-prefix` flips that to measure the radix/prefix cache.

    python -m escha_mlx.bench.isl_osl_grid --model ~/models/escha-w2 \
        --grid nvidia --concurrency 1,4,8,16 --out results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

import httpx

from escha_mlx.benchmark_metadata import annotate_report, benchmark_metadata

# NVIDIA's published ISL/OSL pairs for LLM inference perf tables.
NVIDIA_GRID: list[tuple[int, int]] = [
    (128, 128),
    (128, 2048),
    (128, 4096),
    (500, 2000),
    (1000, 1000),
    (1000, 2000),
    (2048, 128),
    (2048, 2048),
    (5000, 500),
    (20000, 2000),
]

# A short grid for iteration / smoke.
SHORT_GRID: list[tuple[int, int]] = [(128, 128), (1000, 1000), (2048, 128)]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


class PromptFactory:
    """Exact-token-length prompts.

    Random ids from a mid-vocab band (avoids specials/bytes that could merge or
    render as control chars), decoded to text.  We then VERIFY the re-encoded
    length and correct it, because detok->retok is not always identity.
    """

    def __init__(self, tokenizer, seed: int = 0) -> None:
        self.tok = tokenizer
        self.rng = random.Random(seed)
        v = getattr(tokenizer, "vocab_size", None) or 100000
        self.lo, self.hi = 1000, min(v - 1, 60000)

    def make(self, n_tokens: int, shared_prefix_tokens: int = 0,
             shared_ids: list[int] | None = None) -> str:
        ids: list[int] = []
        if shared_prefix_tokens and shared_ids:
            ids.extend(shared_ids[:shared_prefix_tokens])
        while len(ids) < n_tokens:
            ids.append(self.rng.randint(self.lo, self.hi))
        text = self.tok.decode(ids[:n_tokens])
        # correct for detok/retok drift
        for _ in range(6):
            got = len(self.tok.encode(text, add_special_tokens=False))
            if got == n_tokens:
                break
            if got > n_tokens:
                ids = ids[: max(1, len(ids) - (got - n_tokens))]
            else:
                ids = ids + [self.rng.randint(self.lo, self.hi)
                             for _ in range(n_tokens - got)]
            text = self.tok.decode(ids)
        return text


async def one_request(client: httpx.AsyncClient, url: str, prompt: str,
                      osl: int, sem: asyncio.Semaphore, timeout: float) -> dict:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": osl,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        # exact OSL: without this the batch drains as sequences hit EOS and the
        # numbers describe a decaying batch, not steady-state throughput
        "ignore_eos": True,
    }
    async with sem:
        t0 = time.perf_counter()
        ttft = None
        n_chunks = 0
        usage = None
        text_len = 0
        try:
            async with client.stream("POST", url, json=body, timeout=timeout) as r:
                if r.status_code != 200:
                    body_txt = (await r.aread()).decode()[:200]
                    return {"error": f"HTTP {r.status_code}: {body_txt}"}
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        d = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if d.get("usage"):
                        usage = d["usage"]
                    ch = d.get("choices") or []
                    if ch:
                        delta = ch[0].get("delta") or {}
                        piece = delta.get("content") or delta.get("reasoning_content") or ""
                        if piece:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            n_chunks += 1
                            text_len += len(piece)
        except Exception as e:  # noqa: BLE001 - report, don't abort the sweep
            return {"error": f"{type(e).__name__}: {str(e)[:160]}"}
        e2e = time.perf_counter() - t0

    out_tok = (usage or {}).get("completion_tokens") or n_chunks
    in_tok = (usage or {}).get("prompt_tokens")
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    tpot = ((e2e - ttft) / (out_tok - 1)) if (ttft is not None and out_tok > 1) else float("nan")
    return {
        "ttft": ttft, "e2e": e2e, "out_tok": out_tok, "in_tok": in_tok,
        "cached_tok": cached, "tpot": tpot,
        "per_user_tps": (out_tok / e2e) if e2e > 0 else 0.0,
    }


async def run_point(url: str, prompts: list[str], osl: int, conc: int,
                    timeout: float) -> dict:
    sem = asyncio.Semaphore(conc)
    limits = httpx.Limits(max_connections=conc + 4, max_keepalive_connections=conc + 4)
    async with httpx.AsyncClient(limits=limits) as client:
        t0 = time.perf_counter()
        rs = await asyncio.gather(*[
            one_request(client, url, p, osl, sem, timeout) for p in prompts
        ])
        wall = time.perf_counter() - t0

    ok = [r for r in rs if "error" not in r and r.get("ttft") is not None]
    errs = [r["error"] for r in rs if "error" in r]
    if not ok:
        return {"error": errs[:2] or ["no successful requests"], "wall_s": round(wall, 2)}

    ttfts = [r["ttft"] for r in ok]
    tpots = [r["tpot"] for r in ok if r["tpot"] == r["tpot"]]
    out_total = sum(r["out_tok"] for r in ok)
    in_total = sum(r["in_tok"] or 0 for r in ok)
    return {
        "n_ok": len(ok), "n_err": len(errs), "errors": errs[:2],
        "wall_s": round(wall, 2),
        "ttft_p50": round(pct(ttfts, 50), 3),
        "ttft_p99": round(pct(ttfts, 99), 3),
        "tpot_ms_p50": round(1000 * pct(tpots, 50), 2) if tpots else None,
        "per_user_tps": round(statistics.mean(r["per_user_tps"] for r in ok), 2),
        "aggregate_out_tps": round(out_total / wall, 2),
        "total_tps": round((out_total + in_total) / wall, 2),
        "out_tok_mean": round(out_total / len(ok), 1),
        "osl_target": osl,
        "osl_hit_rate": round(sum(1 for r in ok if r["out_tok"] >= osl) / len(ok), 3),
        "cached_tok_mean": round(sum(r["cached_tok"] for r in ok) / len(ok), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir (for the tokenizer)")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--grid", default="short", help="nvidia | short | 'isl:osl,isl:osl'")
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--requests-per-point", type=int, default=0,
                    help="0 = 2x concurrency (min 4)")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--shared-prefix", type=int, default=0,
                    help="tokens of common prefix across requests (measures prefix cache)")
    ap.add_argument("--max-isl", type=int, default=10**9,
                    help="skip grid points above this ISL")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    metadata = benchmark_metadata(args.model)

    if args.grid == "nvidia":
        grid = NVIDIA_GRID
    elif args.grid == "short":
        grid = SHORT_GRID
    else:
        grid = [tuple(int(x) for x in p.split(":")) for p in args.grid.split(",")]
    concs = [int(c) for c in args.concurrency.split(",")]

    from mlx_lm.utils import load_tokenizer
    tok = load_tokenizer(__import__("pathlib").Path(args.model))
    pf = PromptFactory(tok, seed=1234)

    shared_ids = None
    if args.shared_prefix:
        shared_ids = [pf.rng.randint(pf.lo, pf.hi) for _ in range(args.shared_prefix)]

    print(f"ISL/OSL grid: {len(grid)} points x concurrency {concs}")
    print(f"{'ISL':>6} {'OSL':>6} {'C':>4} | {'TTFT p50':>9} {'TTFT p99':>9} "
          f"{'TPOT ms':>8} | {'per-user':>9} {'AGG tok/s':>10} {'total':>8} | "
          f"{'osl hit':>8} {'cached':>7}")
    print("-" * 108)

    results = []
    for isl, osl in grid:
        if isl > args.max_isl:
            print(f"{isl:>6} {osl:>6}    - | SKIPPED (--max-isl {args.max_isl})")
            results.append({"isl": isl, "osl": osl, "skipped": "max-isl"})
            continue
        for c in concs:
            n = args.requests_per_point or max(4, 2 * c)
            prompts = [pf.make(isl, args.shared_prefix, shared_ids) for _ in range(n)]
            r = asyncio.run(run_point(args.url, prompts, osl, c, args.timeout))
            r.update({"isl": isl, "osl": osl, "concurrency": c, "n_requests": n})
            results.append(r)
            if "error" in r:
                print(f"{isl:>6} {osl:>6} {c:>4} | ERROR {str(r['error'])[:80]}")
            else:
                print(f"{isl:>6} {osl:>6} {c:>4} | {r['ttft_p50']:>9.3f} {r['ttft_p99']:>9.3f} "
                      f"{(r['tpot_ms_p50'] or 0):>8.2f} | {r['per_user_tps']:>9.2f} "
                      f"{r['aggregate_out_tps']:>10.2f} {r['total_tps']:>8.1f} | "
                      f"{r['osl_hit_rate']:>8.2f} {r['cached_tok_mean']:>7.0f}")
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(annotate_report(results, metadata), f, indent=2)

    if args.out:
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
