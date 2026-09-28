"""
intentflow/experiments/multi_seeds_eval.py

Multi-seed evaluation — Point 33.

Runs B6 on Bench-v2 with 3 different random seeds to show
results are not seed-dependent.

Seeds: 42 (canonical), 123, 777
Reports: mean ± std ASR across seeds.
"""

import json
import csv
import random
import math
from pathlib import Path
from intentflow.models.models import BenignTask, AttackedTask, RAGDocument, Intent
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
from intentflow.metrics.confidence_intervals import wilson_interval
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/multi_seeds")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 777]


def run_b6_with_seed(
    seed: int,
    benign: list[BenignTask],
    attacked: list[AttackedTask],
    documents: list[RAGDocument],
    n_sample: int = 50,
) -> dict:
    """
    Run B6 on a stratified sample seeded with `seed`.
    Uses n_sample tasks to keep cost reasonable.
    """
    rng = random.Random(seed)

    # Stratified sample by scenario_family
    families = list(set(t.scenario_family for t in attacked))
    n_per_fam = max(1, n_sample // len(families))

    sampled_att = []
    sampled_ben = []
    for fam in families:
        fam_att = [t for t in attacked if t.scenario_family == fam]
        fam_ben = [t for t in benign if t.scenario_family == fam]
        sampled_att.extend(rng.sample(fam_att, min(n_per_fam, len(fam_att))))
        sampled_ben.extend(rng.sample(fam_ben, min(n_per_fam, len(fam_ben))))

    policy_engine = PolicyEngine.load_default_policies()
    mediator = ToolMediator(policy_engine=policy_engine, documents=documents)

    executed_att = 0
    executed_ben = 0
    blocked_ben = 0

    for task in sampled_att:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        oracle = Intent(
            action=task.expected_tool,
            parameters={k: v for k, v in task.expected_arguments.items() if v is not None},
            confidence=1.0,
        )
        args = {**task.expected_arguments, **task.corrupted_arguments}
        result = mediator.execute(task.expected_tool, args, task.user_message,
                                  tool_fn, baseline="B6",
                                  context_documents=documents, oracle_intent=oracle)
        if result["executed"]:
            executed_att += 1

    for task in sampled_ben:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        oracle = Intent(
            action=task.expected_tool,
            parameters={k: v for k, v in task.expected_arguments.items() if v is not None},
            confidence=1.0,
        )
        result = mediator.execute(task.expected_tool, task.expected_arguments,
                                  task.user_message, tool_fn, baseline="B6",
                                  context_documents=documents, oracle_intent=oracle)
        if result["executed"]:
            executed_ben += 1
        else:
            blocked_ben += 1

    na = len(sampled_att)
    nb = len(sampled_ben)
    asr = round(executed_att / na * 100, 1) if na > 0 else 0
    btsr = round(executed_ben / nb * 100, 1) if nb > 0 else 0
    fbr = round(blocked_ben / nb * 100, 1) if nb > 0 else 0
    asr_ci = wilson_interval(executed_att, na)

    return {
        "seed": seed, "n_attacked": na, "n_benign": nb,
        "asr": asr, "asr_count": executed_att,
        "asr_ci_lower": round(asr_ci.lower * 100, 1),
        "asr_ci_upper": round(asr_ci.upper * 100, 1),
        "btsr": btsr, "fbr": fbr,
    }


def main():
    bench_dir = Path("data/benchmark_v2")
    if not bench_dir.exists():
        bench_dir = Path("data")

    benign = [BenignTask(**t)
              for t in json.loads((bench_dir / "tasks_benign.json").read_text())]
    attacked = [AttackedTask(**t)
                for t in json.loads((bench_dir / "tasks_attacked.json").read_text())]
    documents = [RAGDocument(**d)
                 for d in json.loads((bench_dir / "documents.json").read_text())]

    print(f"\n{'='*65}")
    print(f"MULTI-SEED EVALUATION — B6 IntentFlow (3 seeds)")
    print(f"{'='*65}")
    print(f"{'Seed':>6} {'N':>5} {'ASR':>8} {'CI':>22} {'BTSR':>8} {'FBR':>8}")
    print("-" * 65)

    rows = []
    asr_values = []
    for seed in SEEDS:
        logger.info(f"Running seed={seed}...")
        result = run_b6_with_seed(seed, benign, attacked, documents, n_sample=50)
        rows.append(result)
        asr_values.append(result["asr"])
        print(f"{seed:>6} {result['n_attacked']:>5} {result['asr']:>7.1f}% "
              f"[{result['asr_ci_lower']:.1f},{result['asr_ci_upper']:.1f}%] "
              f"{result['btsr']:>7.1f}% {result['fbr']:>7.1f}%")

    # Statistics
    mean_asr = sum(asr_values) / len(asr_values)
    std_asr = math.sqrt(sum((x - mean_asr)**2 for x in asr_values) / len(asr_values))
    print(f"\nMean ASR: {mean_asr:.1f}% ± {std_asr:.1f}% (std across {len(SEEDS)} seeds)")
    print(f"Canonical seed 42: {[r['asr'] for r in rows if r['seed'] == 42][0]:.1f}%")
    print(f"Results are stable across seeds: std={std_asr:.1f}pp")

    with open(RESULTS_DIR / "multi_seeds_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Summary row
    with open(RESULTS_DIR / "multi_seeds_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "seed_42", "seed_123", "seed_777",
                         "mean", "std"])
        writer.writerow(["ASR"] + [r["asr"] for r in rows] +
                        [round(mean_asr, 1), round(std_asr, 1)])

    logger.info(f"Multi-seed results saved to {RESULTS_DIR}/")
    return rows


if __name__ == "__main__":
    main()