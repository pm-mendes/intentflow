"""
intentflow/experiments/run_b6_n100.py

B6 évalué sur N=100 stratifié (recommandation Priorité 2).

Double le N=50 précédent tout en conservant la stratification.
Répond directement à : "The main defense is not evaluated on the full
large-scale benchmark."

`use_oracle=True` (voir `run_b6_n200.py`) : ce benchmark contrôlé est censé
isoler la qualité de médiation de l'extraction d'intention ; vérifié sur
N=200 que les deux méthodologies donnent des résultats identiques, avec un
gain de latence massif (~1000ms/appel -> ~35ms, plus d'appel API).
"""

import json
import csv
from pathlib import Path

from intentflow.models.models import BenignTask, AttackedTask, RAGDocument
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.baselines.runner import BaselineRunner
from intentflow.metrics.evaluator import MetricsComputer
from intentflow.metrics.confidence_intervals import compute_all_intervals
from intentflow.experiments.run_benchmark_v2_n50 import (
    stratified_sample_attacked,
    stratified_sample_benign,
    compute_residual_asr_by_attack_type,
)
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/b6_n100")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR = Path("data/benchmark_v2")


def _load_jsonl(path) -> list:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    # Charger Bench-v2
    all_benign = [
        BenignTask(**t) for t in json.loads((BENCH_DIR / "tasks_benign.json").read_text())
    ]
    all_attacked = [
        AttackedTask(**t) for t in json.loads((BENCH_DIR / "tasks_attacked.json").read_text())
    ]
    documents = [RAGDocument(**d) for d in json.loads((BENCH_DIR / "documents.json").read_text())]

    # Échantillons stratifiés N=100
    b_sample = stratified_sample_benign(all_benign, n=100)
    a_sample = stratified_sample_attacked(all_attacked, n=100)

    from collections import Counter

    fam_dist = Counter(t.scenario_family for t in a_sample)
    diff_dist = Counter(
        "easy"
        if "easy" in (t.attack_payload or "")
        else "medium"
        if "medium" in (t.attack_payload or "")
        else "hard"
        if "hard" in (t.attack_payload or "")
        else "unknown"
        for t in a_sample
    )
    print(f"\n=== B6 Stratified Sample N=100 ===")
    print(f"By family    : {dict(fam_dist)}")
    print(f"By difficulty: {dict(diff_dist)}")

    # Runner
    policy_engine = PolicyEngine.load_default_policies()
    mediator = ToolMediator(policy_engine=policy_engine, documents=documents)
    runner = BaselineRunner(mediator=mediator, output_dir=str(RESULTS_DIR))

    print("\n--- Running B6 on stratified N=100 (oracle intent) ---")
    log_file = runner.run("B6", b_sample, a_sample, documents, use_oracle=True)
    decisions = _load_jsonl(log_file)
    metrics = MetricsComputer.compute(str(log_file), "B6")
    intervals = compute_all_intervals(decisions, "B6")

    print(f"\n=== B6 N=100 Results ===")
    print(f"ASR  : {metrics.asr:.1f}% {intervals['ASR']}")
    print(f"BTSR : {metrics.btsr:.1f}% {intervals['BTSR']}")
    print(f"FBR  : {metrics.fbr:.1f}% {intervals['FBR']}")
    print(f"Lat med/p95 : {metrics.latency_median:.0f}ms / {metrics.latency_p95:.0f}ms")

    # Table résiduel ASR par type
    print("\n=== Residual ASR by Attack Type (B6, N=100) ===")
    print(f"{'Attack type':<42} {'N':>5} {'ASR':>6} {'95% CI'}")
    print("-" * 75)
    residual = compute_residual_asr_by_attack_type(str(log_file))
    for category, stats in residual.items():
        print(
            f"{category:<42} {stats['total']:>5} {stats['asr']:>5.1f}% "
            f"[{stats['ci_lower']:.1f}–{stats['ci_upper']:.1f}%]"
        )

    # Export CSV
    with open(RESULTS_DIR / "b6_n100_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["baseline", "n_benign", "n_attacked", "asr", "btsr", "fbr", "lat_median", "lat_p95"]
        )
        writer.writerow(
            [
                "B6",
                len(b_sample),
                len(a_sample),
                metrics.asr,
                metrics.btsr,
                metrics.fbr,
                metrics.latency_median,
                metrics.latency_p95,
            ]
        )

    logger.info(f"B6 N=100 results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
