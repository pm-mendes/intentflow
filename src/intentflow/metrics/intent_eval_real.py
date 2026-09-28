"""
intentflow/metrics/intent_eval_real.py

Real (non-simulated) intent extractor evaluation on the 200-message gold
standard from intent_eval_extended.py. That module's simulate_llm_predictions()
never calls an LLM -- it draws from hardcoded per-field "known" success rates.
This script actually calls the extractor (gpt-4o-mini via OpenAI, or any
OpenAI-compatible endpoint such as a local Ollama server) and computes real F1.

Usage:
  poetry run python -m intentflow.metrics.intent_eval_real --model gpt-4o-mini
  poetry run python -m intentflow.metrics.intent_eval_real --model llama3 \
      --base-url http://localhost:11434/v1 --api-key ollama --tag llama3
"""

import argparse
import csv
import random
import time
from pathlib import Path

from intentflow.intent.extractor import IntentExtractor
from intentflow.metrics.intent_eval_extended import build_gold_standard, compute_f1
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/intent_eval_real")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ["action", "recipient", "path", "endpoint", "date", "participants", "selector"]


def real_llm_predictions(
    gold_list: list[dict],
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    json_mode: bool = False,
) -> tuple[list[dict], int]:
    extractor = IntentExtractor(
        model=model, temperature=0.0, use_llm=True, base_url=base_url, api_key=api_key,
        json_mode=json_mode,
    )
    preds = []
    n_fallback = 0
    t0 = time.time()
    for i, g in enumerate(gold_list):
        intent = extractor.extract(g["msg"], mode="llm")
        if intent.extraction_mode != "llm":
            n_fallback += 1
        params = intent.parameters or {}
        pred = {
            "action": intent.action if intent.action and intent.action != "unknown" else None,
            "recipient": params.get("recipient"),
            "path": params.get("path"),
            "endpoint": params.get("endpoint"),
            "date": params.get("date"),
            "participants": params.get("participants"),
            "selector": params.get("query"),
        }
        preds.append(pred)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            logger.info(f"[{model}] {i + 1}/{len(gold_list)} done ({elapsed:.0f}s elapsed, "
                        f"{n_fallback} JSON/parse fallbacks so far)")
    return preds, n_fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--tag", default=None, help="Output file tag (default: sanitized model name)")
    parser.add_argument("--json-mode", action="store_true",
                         help="Request response_format=json_object (recommended for non-OpenAI models)")
    parser.add_argument("--n-sample", type=int, default=None,
                         help="Random subsample size (fixed seed=42) instead of the full 200")
    args = parser.parse_args()

    tag = args.tag or args.model.replace("/", "_").replace(":", "_")

    gold = build_gold_standard()
    if args.n_sample and args.n_sample < len(gold):
        gold = random.Random(42).sample(gold, args.n_sample)
    logger.info(f"Gold standard: {len(gold)} messages; running real extraction with model={args.model}")

    t0 = time.time()
    preds, n_fallback = real_llm_predictions(
        gold, args.model, base_url=args.base_url, api_key=args.api_key, json_mode=args.json_mode
    )
    elapsed = time.time() - t0
    print(f"\nJSON/parse fallbacks to heuristic: {n_fallback}/{len(gold)} "
          f"({n_fallback / len(gold) * 100:.0f}%) -- these do NOT reflect {args.model}'s own output")

    f1_results = compute_f1(gold, preds)

    print(f"\n{'=' * 65}")
    print(f"REAL INTENT EXTRACTOR EVAL -- model={args.model} (N={len(gold)})")
    print(f"{'=' * 65}")
    print(f"{'Field':<15} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    rows = []
    for field in FIELDS:
        r = f1_results.get(field)
        if not r:
            continue
        print(f"{field:<15} {r['f1']:>7.1f}% {r['precision']:>9.1f}% {r['recall']:>7.1f}%")
        rows.append({"field": field, **r})

    avg_f1 = sum(r["f1"] for r in rows) / len(rows) if rows else 0.0
    print(f"\nAverage F1: {avg_f1:.1f}%")
    print(f"Total wall time: {elapsed:.0f}s ({elapsed / len(gold) * 1000:.0f}ms/call avg)")

    out_path = RESULTS_DIR / f"intent_eval_real_{tag}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["field", "f1", "precision", "recall", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(rows)

    with open(RESULTS_DIR / f"intent_eval_real_{tag}_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "n", "avg_f1", "wall_time_s", "ms_per_call"])
        writer.writerow([args.model, len(gold), round(avg_f1, 1), round(elapsed, 1),
                          round(elapsed / len(gold) * 1000, 0)])

    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
