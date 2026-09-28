"""
intentflow/experiments/run_benchmark_v2.py

Script d'expérience pour IntentFlow-Bench-v2.

Usage : poetry run python -m intentflow.experiments.run_benchmark_v2

Lance B0–B6 sur le benchmark élargi (400 tâches) et calcule
les métriques avec intervalles de confiance Wilson.
Les résultats sont exportés dans data/results/benchmark_v2/.

`use_oracle=True` (voir `run_b6_n200.py`) pour B4/B6 : vérifié sur N=200 que
le mode oracle et le mode LLM réel donnent des résultats identiques pour ce
benchmark contrôlé — plus d'appel API nécessaire. Sans effet sur B0/B1/B2/B3/B5,
qui n'utilisent jamais `oracle_intent`.
"""

import csv
import json
from pathlib import Path

from intentflow.baselines.runner import BaselineRunner
from intentflow.data.benchmark_v2 import BenchmarkV2Generator
from intentflow.metrics.confidence_intervals import (
    compute_all_intervals,
)
from intentflow.metrics.evaluator import MetricsComputer
from intentflow.models.models import AttackedTask, BenignTask, RAGDocument
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/benchmark_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BENCH_DIR = Path("data/benchmark_v2")


def _load_jsonl(path) -> list:
    """Charge un fichier JSONL de façon sûre."""
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    # 1. Générer ou charger le benchmark
    if not (BENCH_DIR / "tasks_benign.json").exists():
        logger.info("Generating IntentFlow-Bench-v2...")
        gen = BenchmarkV2Generator(seed=42)
        gen.generate_all(str(BENCH_DIR))

    benign_tasks = [
        BenignTask(**t) for t in json.loads((BENCH_DIR / "tasks_benign.json").read_text())
    ]
    attacked_tasks = [
        AttackedTask(**t) for t in json.loads((BENCH_DIR / "tasks_attacked.json").read_text())
    ]
    documents = [RAGDocument(**d) for d in json.loads((BENCH_DIR / "documents.json").read_text())]

    logger.info(
        f"Loaded benchmark: {len(benign_tasks)} benign, "
        f"{len(attacked_tasks)} attacked, {len(documents)} docs"
    )

    # 2. Initialiser le pipeline
    policy_engine = PolicyEngine.load_default_policies()
    mediator = ToolMediator(policy_engine=policy_engine, documents=documents)
    runner = BaselineRunner(mediator=mediator, output_dir=str(RESULTS_DIR))

    # 3. Exécuter toutes les baselines
    # B4 et B6 sont coûteux (LLM) — on les limite à un sous-ensemble
    # pour le benchmark v2 afin de garder un temps raisonnable.
    # On peut passer FULL=True pour tout exécuter.
    full_llm = False  # Mettre True pour exécuter B4/B6 complets sur 400 tâches

    if full_llm:
        baselines = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]
        b_tasks = benign_tasks
        a_tasks = attacked_tasks
    else:
        # Mode rapide : B0-B3 et B5 sur tout, B4 et B6 sur 50 tâches
        baselines_fast = ["B0", "B1", "B2", "B3", "B5"]
        baselines_llm = ["B4", "B6"]
        b_sample = benign_tasks[:25]
        a_sample = attacked_tasks[:25]

    all_metrics = {}
    intervals_by_baseline = {}

    if full_llm:
        for baseline in baselines:
            log_file = runner.run(baseline, b_tasks, a_tasks, documents, use_oracle=True)
            decisions = _load_jsonl(log_file)
            metrics = MetricsComputer.compute(str(log_file), baseline)
            all_metrics[baseline] = metrics
            intervals_by_baseline[baseline] = compute_all_intervals(decisions, baseline)
    else:
        for baseline in baselines_fast:
            log_file = runner.run(
                baseline, benign_tasks, attacked_tasks, documents, use_oracle=True
            )
            decisions = _load_jsonl(log_file)
            metrics = MetricsComputer.compute(str(log_file), baseline)
            all_metrics[baseline] = metrics
            intervals_by_baseline[baseline] = compute_all_intervals(decisions, baseline)

        for baseline in baselines_llm:
            logger.info(f"Running {baseline} on sample (n=50)...")
            log_file = runner.run(baseline, b_sample, a_sample, documents, use_oracle=True)
            decisions = _load_jsonl(log_file)
            metrics = MetricsComputer.compute(str(log_file), baseline)
            all_metrics[baseline] = metrics
            intervals_by_baseline[baseline] = compute_all_intervals(decisions, baseline)

    # 4. Afficher les résultats
    print("\n" + "=" * 70)
    print("IntentFlow-Bench-v2 — Main Results")
    print("=" * 70)
    header = f"{'Defense':<22} {'PCR↓':>6} {'TMR↓':>6} {'ASR↓':>6} {'BTSR↑':>6} {'FBR↓':>6} {'Lat(ms)':>9}"
    print(header)
    print("-" * 70)
    for baseline, m in all_metrics.items():
        print(
            f"{baseline:<22} {m.pcr:>5.1f}% {m.tmr:>5.1f}% {m.asr:>5.1f}% "
            f"{m.btsr:>5.1f}% {m.fbr:>5.1f}% {m.latency_median:>8.1f}"
        )

    print("\n" + "=" * 70)
    print("Wilson 95% Confidence Intervals — ASR")
    print("=" * 70)
    for baseline, intervals in intervals_by_baseline.items():
        if "ASR" in intervals:
            print(f"  {baseline}: {intervals['ASR']}")

    # 5. Export CSV
    csv_file = RESULTS_DIR / "metrics_v2.csv"
    fieldnames = [
        "baseline",
        "num_benign",
        "num_attacked",
        "pcr",
        "tmr",
        "asr",
        "btsr",
        "pbr",
        "fbr",
        "latency_median",
        "latency_p95",
    ]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics.values():
            writer.writerow(m.model_dump())

    logger.info(f"Results exported → {csv_file}")
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
