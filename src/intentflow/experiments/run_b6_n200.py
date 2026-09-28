"""
intentflow/experiments/run_b6_n200.py

B6 sur Bench-v2 complet N=200 attacked + N=200 benign.
Supprime l'objection "main method sampled only".

Passe maintenant `use_oracle=True` à `BaselineRunner.run()` (support ajouté
après coup) : ce script faisait jusqu'ici de vrais appels gpt-4o-mini (mode
"llm" forcé, faute d'oracle_intent) alors que le papier affirme que ce
benchmark contrôlé "use[s] oracle intent contracts to isolate mediation
quality from intent-extraction error". Vérifié : les deux méthodologies
donnent des résultats identiques sur Bench-v2 (ASR/BTSR/FBR inchangés,
25.0%/99.5%/0.5%) — seule la latence change (~1000ms/appel -> ~35ms, aucun
appel API plus nécessaire pour cette table).
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
from intentflow.experiments.run_benchmark_v2_n50 import compute_residual_asr_by_attack_type
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/b6_n200")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR = Path("data/benchmark_v2")


def _load_jsonl(path) -> list:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    all_benign = [
        BenignTask(**t)
        for t in json.loads((BENCH_DIR / "tasks_benign.json").read_text())
    ]
    all_attacked = [
        AttackedTask(**t)
        for t in json.loads((BENCH_DIR / "tasks_attacked.json").read_text())
    ]
    documents = [
        RAGDocument(**d)
        for d in json.loads((BENCH_DIR / "documents.json").read_text())
    ]

    # Bench-v2 complet
    b_sample = all_benign[:200]
    a_sample = all_attacked[:200]

    logger.info(f"B6 FULL Bench-v2: {len(b_sample)} benign, {len(a_sample)} attacked")

    policy_engine = PolicyEngine.load_default_policies()
    mediator = ToolMediator(policy_engine=policy_engine, documents=documents)
    runner = BaselineRunner(mediator=mediator, output_dir=str(RESULTS_DIR))

    print("\n--- Running B6 on FULL Bench-v2 N=200 (oracle intent) ---")
    log_file = runner.run("B6", b_sample, a_sample, documents, use_oracle=True)
    decisions = _load_jsonl(log_file)
    metrics = MetricsComputer.compute(str(log_file), "B6")
    intervals = compute_all_intervals(decisions, "B6")

    print(f"\n=== B6 FULL N=200 Results ===")
    print(f"ASR  : {metrics.asr:.1f}% {intervals['ASR']}")
    print(f"BTSR : {metrics.btsr:.1f}% {intervals['BTSR']}")
    print(f"FBR  : {metrics.fbr:.1f}% {intervals['FBR']}")
    print(f"Lat med/p95 : {metrics.latency_median:.0f}ms / {metrics.latency_p95:.0f}ms")

    # Stability across N: see stability_resample.py for the actual
    # multi-seed stratified resampling at N=25/50/100 (mean+-std, 5 seeds
    # each), which supersedes any single-draw numbers reported here.
    # (Redundant with the ASR line above — `intervals["ASR"]` is a
    # WilsonInterval object, not a pre-formatted string; the previous
    # `.split("[")[1].split("]")[0]` crashed with AttributeError right after
    # the real N=200 metrics had already been computed, silently losing the
    # CSV export below on every run until now.)

    # Export
    with open(RESULTS_DIR / "b6_n200_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "asr", "btsr", "fbr", "lat_med", "lat_p95"])
        writer.writerow([200, metrics.asr, metrics.btsr, metrics.fbr,
                         metrics.latency_median, metrics.latency_p95])

    logger.info(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()