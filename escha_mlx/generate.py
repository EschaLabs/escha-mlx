"""CLI generation smoke: python -m escha_mlx.generate --model <dir> --prompt "..."

Uses mlx-lm's generation loop (prompt cache, streaming) on the escha model.
"""
from __future__ import annotations

import argparse
import logging
import time


def main() -> None:
    ap = argparse.ArgumentParser(description="escha-mlx generation smoke")
    ap.add_argument("--model", required=True, help="path to the escha checkpoint dir")
    ap.add_argument("--prompt", default="What is 17 multiplied by 23?")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--raw", action="store_true", help="no chat template")
    ap.add_argument("--thinking", action="store_true", help="enable thinking mode")
    ap.add_argument("--reasoning-effort",
                    help="reasoning effort for models whose chat template takes one "
                         "(Qwen3.8: low | medium | xhigh, default xhigh). Passed "
                         "through to the template only when given, so each model "
                         "keeps its own default; a template that does not use it "
                         "ignores it.")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from .loader import load

    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"[escha-mlx] model loaded in {time.time() - t0:.1f}s")

    if args.raw:
        prompt = args.prompt
    else:
        extra = ({"reasoning_effort": args.reasoning_effort}
                 if args.reasoning_effort else {})
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=args.thinking, **extra)

    sampler = make_sampler(temp=args.temp)
    last = None
    for r in stream_generate(model, tokenizer, prompt,
                             max_tokens=args.max_tokens, sampler=sampler):
        print(r.text, end="", flush=True)
        last = r
    print()
    if last is not None:
        print(f"[escha-mlx] prompt: {last.prompt_tokens} tok @ "
              f"{last.prompt_tps:.1f} tok/s | generation: {last.generation_tokens} tok @ "
              f"{last.generation_tps:.1f} tok/s | peak mem {last.peak_memory:.1f} GB")


if __name__ == "__main__":
    main()
