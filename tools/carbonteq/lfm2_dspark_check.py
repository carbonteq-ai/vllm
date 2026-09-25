# SPDX-License-Identifier: Apache-2.0
"""Greedy LFM2.5 generation with and without its DSpark drafter (CarbonTeq).

    python tools/carbonteq/lfm2_dspark_check.py --mode target --out target.json
    python tools/carbonteq/lfm2_dspark_check.py --mode dspark --out dspark.json
    python tools/carbonteq/lfm2_dspark_check.py --compare target.json dspark.json

Each mode runs in its own process so the engines never share GPU memory.
"""

import argparse
import json
import os
import time

TARGET = "LiquidAI/LFM2.5-2.6B"
TARGET_REVISION = "654f9463ce32b05d0429d76fe1f580b27d4c1ac0"
DRAFT = "LiquidAI/LFM2.5-2.6B-DSpark"
DRAFT_REVISION = "458cedab07d0f7b2b05700c77e1aa463d43d6f04"

PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a haiku about a lighthouse, then explain its imagery in two sentences.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Show the steps.",
    "List five ways to reduce the memory use of a Python web service.",
    "Summarize the causes of the French Revolution in one paragraph.",
    "Write a SQL query that returns the three most recent orders per customer.",
    "What is the difference between a process and a thread?",
    "Draft a polite email asking a colleague to review a pull request by Friday.",
]


def run(mode: str, out: str, concurrency: int, max_tokens: int) -> None:
    from vllm import LLM, SamplingParams

    kwargs = {}
    if mode == "dspark":
        kwargs["speculative_config"] = {
            "method": "dspark",
            "model": DRAFT,
            "revision": DRAFT_REVISION,
            "num_speculative_tokens": 9,
            **json.loads(os.environ.get("SPEC_EXTRA_JSON", "{}")),
        }
    llm = LLM(
        TARGET,
        revision=TARGET_REVISION,
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        seed=0,
        disable_log_stats=False,
        enable_prefix_caching=True,
        **kwargs,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        for text in PROMPTS
    ]
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    llm.generate(prompts[:1], params)  # warm up graphs and caches

    results = {"mode": mode, "runs": {}}
    for batch in (1, concurrency):
        batch_prompts = (prompts * ((batch + len(prompts) - 1) // len(prompts)))[:batch]
        start = time.perf_counter()
        outputs = llm.generate(batch_prompts, params)
        elapsed = time.perf_counter() - start
        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        results["runs"][str(batch)] = {
            "seconds": elapsed,
            "output_tokens": tokens,
            "tokens_per_second": tokens / elapsed,
            "token_ids": [list(o.outputs[0].token_ids) for o in outputs[: len(PROMPTS)]],
        }
    metrics = {}
    for metric in llm.get_metrics():
        if metric.name.startswith("vllm:spec_decode"):
            value = getattr(metric, "value", None)
            if value is None and hasattr(metric, "values"):
                value = list(metric.values)
            metrics[metric.name] = value
    results["spec_decode_metrics"] = metrics
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle)


def compare(target_path: str, dspark_path: str) -> None:
    target = json.load(open(target_path, encoding="utf-8"))
    dspark = json.load(open(dspark_path, encoding="utf-8"))
    for batch, run_target in target["runs"].items():
        run_dspark = dspark["runs"][batch]
        pairs = list(zip(run_target["token_ids"], run_dspark["token_ids"]))
        identical = sum(a == b for a, b in pairs)
        prefix = [
            next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            for a, b in pairs
        ]
        print(
            f"batch {batch}: identical {identical}/{len(pairs)}, first divergence {prefix}, "
            f"target {run_target['tokens_per_second']:.1f} tok/s, "
            f"dspark {run_dspark['tokens_per_second']:.1f} tok/s, "
            f"speedup {run_dspark['tokens_per_second'] / run_target['tokens_per_second']:.2f}x"
        )
    metrics = dspark["spec_decode_metrics"]
    drafts = metrics.get("vllm:spec_decode_num_drafts") or 0
    accepted = metrics.get("vllm:spec_decode_num_accepted_tokens") or 0
    if drafts:
        print(f"mean accepted tokens per draft: {accepted / drafts:.2f} (of 9)")
    print(json.dumps(metrics)[:600])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("target", "dspark"))
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args.mode, args.out, args.concurrency, args.max_tokens)


if __name__ == "__main__":
    main()
