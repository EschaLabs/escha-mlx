"""Compare real-model AR, FP16 MTP and Q4 MTP outputs.

This is an output comparison, not a throughput benchmark. All arms share one
unchanged target and use fresh caches; independent heads load the same MTP file.
By default, run 12 greedy cases (256 or 512 tokens), two sampled cases (128
tokens), then repeat an observed greedy difference, a sampled case, and an
unchanged greedy control when present. Every arm stops at EOS or its token limit.

Example::

    python bench/mtp_head_output.py --model /path/to/checkpoint \\
        --out /tmp/mtp-head-output

The optional ``--source`` selects a different escha-mlx checkout. The report
records that checkout's actual revision, dirty state and source hashes; a run
against a rebased checkout is a new measurement, not a relabeling of old data.
The runner uses a 19 GB MLX memory limit, matching the original 24 GB test host.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


CASES = [
    dict(id="code_python", thinking=False, limit=256, prompt=
         "Write a Python function merge_intervals(intervals) that merges overlapping "
         "closed intervals. Include three assertions covering empty input, touching "
         "intervals, and unsorted input. Output only code."),
    dict(id="code_sql", thinking=False, limit=256, prompt=
         "PostgreSQL has orders(id, customer_id, created_at, amount). Write a query "
         "returning each customer's latest order, breaking timestamp ties by largest "
         "id. Explain in two sentences why the result has one row per customer."),
    dict(id="math_calc", thinking=False, limit=256, prompt=
         "Calculate (137 * 289) - (48 * 76) + 2026. Show the intermediate products "
         "and put the exact integer answer on the last line."),
    dict(id="math_word", thinking=False, limit=256, prompt=
         "A tank is 1/3 full. Adding 24 liters makes it 3/5 full. What is its full "
         "capacity in liters? Show the equations, then verify the answer."),
    dict(id="cjk_explain", thinking=False, limit=256, prompt=
         "请用中文解释数据库事务的隔离级别，说明可重复读与读已提交的区别，"
         "并给出一个两笔事务交错执行的具体例子。控制在200字以内。"),
    dict(id="cjk_translation", thinking=False, limit=256, prompt=
         "将下面这句话译成自然的英文，并解释你如何处理其中的比喻："
         "与其临渊羡鱼，不如退而结网；但开始之前，先确认这片水域真的有鱼。"),
    dict(id="prose", thinking=False, limit=256, prompt=
         "Write a 150-word scene in which a lighthouse keeper discovers that the "
         "light is sending messages. Use concrete sensory details and end with "
         "a single line of dialogue."),
    dict(id="chat", thinking=False, limit=256, prompt=
         "I'm organizing a small reading group for six friends with different tastes. "
         "Suggest a fair way to pick books and run our first meeting. Keep it practical "
         "and give a four-week plan in fewer than 180 words."),
    dict(id="json", thinking=False, limit=256, prompt=
         "Return only valid JSON, no markdown. Extract the following into an object "
         "with keys name (string), date (YYYY-MM-DD), attendees (array of strings), "
         "and duration_minutes (integer): Project Atlas review, September 16, 2026, "
         "10:00-11:30; attendees Alice Chen, Bob Smith and 王小明."),
    dict(id="factual", thinking=False, limit=256, prompt=
         "Explain why the Moon has phases and why they are not normally caused by "
         "Earth's shadow. Mention the conditions for a lunar eclipse. Use fewer "
         "than 180 words."),
    dict(id="reasoning", thinking=True, limit=512, prompt=
         "There are three boxes labeled Apples, Oranges, and Mixed, and all three "
         "labels are wrong. You may draw one fruit from one box without looking "
         "inside. Which box do you choose, and how can you correctly relabel all "
         "three boxes? Explain the logic."),
    dict(id="long_cot", thinking=True, limit=512, prompt=
         "How many length-8 strings over the alphabet {A, B, C} contain exactly "
         "three A's and no two A's next to each other? Derive the count in two "
         "different ways and verify that the methods agree."),
]
SAMPLED_CASES = [
    dict(id="sample_prose", thinking=False, limit=128, temperature=0.7, top_p=0.9,
         seed=42, prompt="Write a short scene about two strangers who discover they "
         "have the same childhood memory. Begin with dialogue."),
    dict(id="sample_cjk", thinking=False, limit=128, temperature=0.7, top_p=0.9,
         seed=123, prompt="写一段中文科幻小说开头：一位维修工程师发现城市里的钟"
         "每天会同时停下一秒。用具体细节制造悬念。"),
]


def digest(tokens):
    return hashlib.sha256(b"".join(struct.pack("<I", t) for t in tokens)).hexdigest()


def source_dirty(source):
    """Record checkout state without publishing local paths or git filenames."""
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=normal"],
            check=False, capture_output=True, text=True,
        )
    except OSError:
        return None
    return bool(result.stdout) if result.returncode == 0 else None


def compare(left, right, tokenizer):
    a, b = left["tokens"], right["tokens"]
    common = min(len(a), len(b))
    first = next((i for i in range(common) if a[i] != b[i]), None)
    if first is None and len(a) != len(b):
        first = common
    matched = sum(x == y for x, y in zip(a, b))
    result = {
        "exact": a == b,
        "lengths": [len(a), len(b)],
        "position_matches": matched,
        "position_agreement": matched / max(len(a), len(b), 1),
        "first_difference_1based": first + 1 if first is not None else None,
    }
    if first is not None:
        start, end = max(0, first - 20), first + 32
        result["shared_prefix_tail"] = tokenizer.decode(a[max(0, first - 40):first])
        result["left_window"] = tokenizer.decode(a[start:end])
        result["right_window"] = tokenizer.decode(b[start:end])
        result["left_token"] = a[first] if first < len(a) else None
        result["right_token"] = b[first] if first < len(b) else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="escha-mlx source checkout; defaults to this repository.")
    parser.add_argument("--model", type=Path, required=True, help="Local checkpoint directory.")
    parser.add_argument("--model-id", help="Public model identifier to record instead of the directory name.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory; progress is saved as results.json.")
    parser.add_argument("--only", default="", help="Comma-separated case ids; empty runs all.")
    parser.add_argument("--max-tokens", type=int, help="Override output limit for a longer follow-up.")
    args = parser.parse_args()
    source, model_path, out = (path.expanduser().resolve() for path in (args.source, args.model, args.out))
    if not (source / "escha_mlx" / "mtp.py").is_file():
        parser.error("--source must contain the escha_mlx package with native MTP support")
    chosen = set(filter(None, args.only.split(",")))
    unknown = chosen - {case["id"] for case in CASES + SAMPLED_CASES}
    if unknown:
        parser.error(f"Unknown --only case ids: {', '.join(sorted(unknown))}")
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.model_id and (Path(args.model_id).is_absolute() or args.model_id.startswith("~")):
        parser.error("--model-id must be a public identifier, not a local path")
    sys.path.insert(0, str(source))
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.generate import generate_step, generation_stream
    from mlx_lm.sample_utils import make_sampler
    from escha_mlx import envs
    from escha_mlx.benchmark_metadata import (
        escha_mlx_git_revision, model_display_name, model_hf_revision,
    )
    from escha_mlx.loader import load
    from escha_mlx.mtp import load_mtp, mtp_generate_step

    mx.set_memory_limit(19_000_000_000)
    # Check checkout state before creating an output directory inside it. Keep
    # model-location environment variables out of reports intended for sharing.
    recorded_env_names = [name for name in envs.environment_variables
                          if name not in {"ESCHA_MODEL", "ESCHA_DENSE_MODEL"}]
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": escha_mlx_git_revision(source),
        "source_dirty": source_dirty(source),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((source / "escha_mlx").rglob("*.py"))},
        "machine": mx.device_info(),
        "macos": platform.mac_ver()[0],
        "versions": {name: importlib.metadata.version(name)
                     for name in ("mlx", "mlx-lm", "numpy", "safetensors")},
        "model": args.model_id or model_display_name(model_path),
        "model_hf_revision": model_hf_revision(model_path),
        "checkpoint_files": {
            name: {"bytes": (model_path / name).stat().st_size,
                   "sha256": hashlib.sha256((model_path / name).read_bytes()).hexdigest()}
            for name in ("config.json", "mtp/config.json", "mtp/model.safetensors")
        },
        "environment": {name: os.environ[name] for name in recorded_env_names if name in os.environ},
        "settings": {
            "batch": 1, "prefill_step_size": 256, "stop_at_eos": True,
            "target": "Same loaded target instance, same weights for every arm",
            "caches": "Fresh target and draft caches on every run",
            "rng": "Reseed immediately before each generation, after all model loading",
            "ar": "mlx_lm.generate.generate_step; upstream prefill boundary retained",
            "comparison": "Same-position token equality, including EOS; not a quality metric",
        },
        "cases": [],
        "repeats": [],
    }
    out.mkdir(parents=True, exist_ok=True)

    def save():
        (out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print("Loading target...", flush=True)
    started = time.perf_counter()
    model, tokenizer = load(model_path, load_mtp=False)
    mx.synchronize()
    print(f"Target loaded in {time.perf_counter() - started:.1f}s", flush=True)
    heads = {}
    for arm, bits in (("fp16", "fp16"), ("q4", "4")):
        os.environ["ESCHA_MLX_MTP_HEAD_BITS"] = bits
        heads[arm] = load_mtp(model_path, model)
        assert heads[arm]._embed_tokens is model.language_model.model.embed_tokens
        assert heads[arm]._lm_head is getattr(model.language_model.lm_head, "inner", model.language_model.lm_head)
    report["resolved_environment"] = {name: envs.environment_variables[name].get()
                                      for name in recorded_env_names}
    report["head_bytes"] = {
        arm: sum(v.nbytes for _, v in tree_flatten(head.parameters()))
        for arm, head in heads.items()
    }
    eos = set(tokenizer.eos_token_ids)
    report["eos_token_ids"] = sorted(eos)
    mx.clear_cache()
    save()

    def run(case, arm, prompt_ids, references=None):
        temperature = case.get("temperature", 0.0)
        sampler = make_sampler(temp=temperature, top_p=case.get("top_p", 0.0))
        mx.synchronize()
        mx.random.seed(case.get("seed", 42))
        tokens, accepted, differences = [], [], {}
        finished = "length"
        started = time.perf_counter()
        with mx.stream(generation_stream):
            prompt = mx.array(prompt_ids, dtype=mx.uint32)
            if arm == "ar":
                iterator = generate_step(prompt, model, max_tokens=case["limit"],
                                         sampler=sampler, prefill_step_size=256)
            else:
                iterator = mtp_generate_step(prompt, model, mtp_head=heads[arm],
                                             max_tokens=case["limit"], sampler=sampler,
                                             prefill_step_size=256)
            try:
                for item in iterator:
                    token, logprobs = int(item[0]), item[1]
                    position = len(tokens)
                    for name, reference in (references or {}).items():
                        ref_tokens = reference["tokens"]
                        if name not in differences and position < len(ref_tokens) and token != ref_tokens[position]:
                            ref_token = ref_tokens[position]
                            ids = mx.argpartition(-logprobs, kth=4)[:5]
                            values = mx.take(logprobs, ids)
                            mx.eval(ids, values)
                            top = sorted(zip(ids.tolist(), values.tolist()), key=lambda pair: -pair[1])
                            differences[name] = {
                                "position_1based": position + 1,
                                "token": token,
                                "reference_token": ref_token,
                                "logprob_token": float(logprobs[token].item()),
                                "logprob_reference_token": float(logprobs[ref_token].item()),
                                "top5": [{"id": int(t), "text": tokenizer.decode([int(t)]),
                                          "logprob": float(lp)} for t, lp in top],
                            }
                    tokens.append(token)
                    accepted.append(bool(item[2]) if len(item) == 3 else False)
                    if len(tokens) % 64 == 0:
                        print(f"  {case['id']} {arm}: {len(tokens)} tokens / {time.perf_counter() - started:.1f}s", flush=True)
                    if token in eos:
                        finished = "eos"
                        break
            finally:
                iterator.close()
                mx.synchronize()
        elapsed = time.perf_counter() - started
        result = {
            "tokens": tokens,
            "text": tokenizer.decode(tokens),
            "token_count": len(tokens),
            "digest": digest(tokens),
            "finish_reason": finished,
            "elapsed_s": elapsed,
            "from_draft_count": sum(accepted),
            "mean_emit": len(tokens) / max(1, len(tokens) - sum(accepted)),
            "first_difference_logprobs": differences,
        }
        print(f"DONE {case['id']} {arm}: {len(tokens)} tokens, {elapsed:.1f}s, {finished}, {result['digest'][:16]}", flush=True)
        gc.collect()
        mx.clear_cache()
        return result

    cases = [c for c in CASES + SAMPLED_CASES if not chosen or c["id"] in chosen]
    if args.max_tokens is not None:
        cases = [dict(case, limit=args.max_tokens) for case in cases]
    for case in cases:
        kwargs = {"enable_thinking": case["thinking"]}
        if case["thinking"]:
            kwargs["reasoning_effort"] = "low"
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": case["prompt"]}], tokenize=False,
            add_generation_prompt=True, **kwargs)
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        entry = {"case": case, "rendered_prompt": rendered, "prompt_ids": prompt_ids,
                 "prompt_tokens": len(prompt_ids), "runs": {}}
        report["cases"].append(entry)
        for arm in ("ar", "fp16", "q4"):
            entry["runs"][arm] = run(case, arm, prompt_ids, entry["runs"])
            save()
        entry["comparisons"] = {
            "fp16_vs_q4": compare(entry["runs"]["fp16"], entry["runs"]["q4"], tokenizer),
            "ar_vs_fp16": compare(entry["runs"]["ar"], entry["runs"]["fp16"], tokenizer),
            "ar_vs_q4": compare(entry["runs"]["ar"], entry["runs"]["q4"], tokenizer),
        }
        print("COMPARE", case["id"], json.dumps({k: v["first_difference_1based"]
              for k, v in entry["comparisons"].items()}), flush=True)
        save()

    # Recheck the earliest greedy difference and the first sampled case, as well
    # as one unchanged greedy control. Same-head reproducibility separates drift
    # due to the precision setting from nondeterministic reruns.
    repeat_entries = []
    for predicate in (
        lambda e: e["case"].get("temperature", 0) == 0 and not e["comparisons"]["fp16_vs_q4"]["exact"],
        lambda e: e["case"].get("temperature", 0) > 0,
        lambda e: e["case"].get("temperature", 0) == 0 and e["comparisons"]["fp16_vs_q4"]["exact"],
    ):
        match = next((e for e in report["cases"] if predicate(e)), None)
        if match is not None and match not in repeat_entries:
            repeat_entries.append(match)
    for entry in repeat_entries:
        result = {"case_id": entry["case"]["id"], "runs": {}, "same_head_exact": {}}
        for arm in ("q4", "fp16"):
            result["runs"][arm] = run(entry["case"], arm, entry["prompt_ids"], entry["runs"])
            result["same_head_exact"][arm] = result["runs"][arm]["tokens"] == entry["runs"][arm]["tokens"]
        report["repeats"].append(result)
        print("REPEAT", result["case_id"], result["same_head_exact"], flush=True)
        save()

    report["summary"] = {}
    for mode, sampled in (("greedy", False), ("sampled", True)):
        group = [e for e in report["cases"] if (e["case"].get("temperature", 0) > 0) == sampled]
        report["summary"][mode] = {
            "cases": len(group),
            "fp16_q4_exact": sum(e["comparisons"]["fp16_vs_q4"]["exact"] for e in group),
            "ar_fp16_exact": sum(e["comparisons"]["ar_vs_fp16"]["exact"] for e in group),
            "ar_q4_exact": sum(e["comparisons"]["ar_vs_q4"]["exact"] for e in group),
            "token_counts": {arm: sum(e["runs"][arm]["token_count"] for e in group)
                             for arm in ("ar", "fp16", "q4")},
        }
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    save()
    print("SUMMARY", json.dumps(report["summary"]), flush=True)
    print("Wrote", out / "results.json", flush=True)


if __name__ == "__main__":
    main()
