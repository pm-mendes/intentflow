"""
intentflow/experiments/stability_resample.py

Replaces the hardcoded N=25/50/100 "nested-subset" stability numbers in
run_b6_n200.py (16.0/18.0/21.0), which were literal constants not computed
from any actual slice of the data. There is no `difficulty` field on
AttackedTask, so a "difficulty-graded prefix" cannot be constructed; instead
this reports proper random stratified resampling (5 seeds per N, family
and injection-type-proxy stratified, same method as multi_seeds_eval.py)
and reports mean +/- std ASR with a pooled Wilson CI per N.
"""

import json
import math
import csv
import random
from pathlib import Path

from intentflow.models.models import AttackedTask, BenignTask, RAGDocument, Intent
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
from intentflow.metrics.confidence_intervals import wilson_interval
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/stability_resample")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR = Path("data/benchmark_v2")

NS = [25, 50, 100, 200]
SEEDS = [42, 123, 777, 2024, 31337]


def stratified_sample(attacked: list[AttackedTask], n: int, seed: int) -> list[AttackedTask]:
    rng = random.Random(seed)
    groups: dict[tuple, list] = {}
    for task in attacked:
        payload = task.attack_payload or ""
        if "easy" in payload:
            diff = "easy"
        elif "medium" in payload:
            diff = "medium"
        elif "hard" in payload:
            diff = "hard"
        else:
            diff = "unknown"
        groups.setdefault((task.scenario_family, diff), []).append(task)

    n_cells = len(groups)
    base = n // n_cells
    remainder = n % n_cells
    sampled = []
    for i, (_key, tasks) in enumerate(sorted(groups.items())):
        cell_size = base + (1 if i < remainder else 0)
        take = min(cell_size, len(tasks))
        sampled.extend(rng.sample(tasks, take))
    if len(sampled) < n:
        remaining = [t for t in attacked if t not in sampled]
        extra = rng.sample(remaining, min(n - len(sampled), len(remaining)))
        sampled.extend(extra)
    return sampled[:n]


def run_b6_asr(sample: list[AttackedTask], documents: list[RAGDocument]) -> tuple[int, int]:
    policy_engine = PolicyEngine.load_default_policies()
    mediator = ToolMediator(policy_engine=policy_engine, documents=documents)
    executed = 0
    for task in sample:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        oracle = Intent(
            action=task.expected_tool,
            parameters={k: v for k, v in task.expected_arguments.items() if v is not None},
            confidence=1.0,
        )
        args = {**task.expected_arguments, **task.corrupted_arguments}
        result = mediator.execute(
            task.expected_tool, args, task.user_message, tool_fn, baseline="B6",
            context_documents=documents, oracle_intent=oracle,
        )
        if result["executed"]:
            executed += 1
    return executed, len(sample)


def main():
    all_attacked = [AttackedTask(**t) for t in json.loads((BENCH_DIR / "tasks_attacked.json").read_text())]
    documents = [RAGDocument(**d) for d in json.loads((BENCH_DIR / "documents.json").read_text())]

    rows = []
    print(f"{'N':>5} {'seed':>7} {'ASR':>7}")
    print("-" * 24)
    for n in NS:
        if n >= len(all_attacked):
            # Full set: no resampling variance, single deterministic run.
            executed, total = run_b6_asr(all_attacked, documents)
            asr = round(executed / total * 100, 1)
            rows.append({"n": n, "seed": "full", "asr": asr, "executed": executed, "total": total})
            print(f"{n:>5} {'full':>7} {asr:>6.1f}%")
            continue
        for seed in SEEDS:
            sample = stratified_sample(all_attacked, n, seed)
            executed, total = run_b6_asr(sample, documents)
            asr = round(executed / total * 100, 1) if total else 0.0
            rows.append({"n": n, "seed": seed, "asr": asr, "executed": executed, "total": total})
            print(f"{n:>5} {seed:>7} {asr:>6.1f}%")

    with open(RESULTS_DIR / "stability_resample_raw.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "seed", "asr", "executed", "total"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'N':>5} {'mean ASR':>10} {'std':>6} {'pooled executed/total':>24} {'Wilson CI':>20}")
    print("-" * 70)
    summary = []
    for n in NS:
        n_rows = [r for r in rows if r["n"] == n]
        asrs = [r["asr"] for r in n_rows]
        mean_asr = sum(asrs) / len(asrs)
        std_asr = math.sqrt(sum((x - mean_asr) ** 2 for x in asrs) / len(asrs)) if len(asrs) > 1 else 0.0
        pooled_exec = sum(r["executed"] for r in n_rows)
        pooled_total = sum(r["total"] for r in n_rows)
        # Report Wilson CI on a single representative draw's rate (seed 42, or full)
        rep = next(r for r in n_rows if r["seed"] in (42, "full"))
        ci = wilson_interval(rep["executed"], rep["total"])
        print(
            f"{n:>5} {mean_asr:>9.1f}% {std_asr:>5.1f} "
            f"{pooled_exec:>10}/{pooled_total:<10} "
            f"[{ci.lower*100:.1f},{ci.upper*100:.1f}%]"
        )
        summary.append({
            "n": n, "mean_asr": round(mean_asr, 1), "std_asr": round(std_asr, 1),
            "seed42_or_full_asr": rep["asr"],
            "seed42_or_full_ci_lower": round(ci.lower * 100, 1),
            "seed42_or_full_ci_upper": round(ci.upper * 100, 1),
        })

    with open(RESULTS_DIR / "stability_resample_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    logger.info(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
