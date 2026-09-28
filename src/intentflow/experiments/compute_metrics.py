"""
Calcule et exporte les métriques pour toutes les baselines.
Usage : python -m intentflow.experiments.compute_metrics
"""

import csv
from pathlib import Path

from intentflow.metrics.evaluator import MetricsComputer
from intentflow.pipeline.mediator import ToolMediator
from intentflow.utils.logger import logger


def main() -> None:
    results_dir = Path("data/results")
    metrics_file = results_dir / "metrics.csv"

    all_metrics = []
    for baseline in ToolMediator.BASELINES:
        decisions_file = results_dir / f"decisions_{baseline}.jsonl"
        if not decisions_file.exists():
            logger.warning(f"Missing {decisions_file}, skipping {baseline}")
            continue
        metrics = MetricsComputer.compute(str(decisions_file), baseline)
        all_metrics.append(metrics)

    if not all_metrics:
        logger.error("No metrics computed. Run run_baseline.py first.")
        return

    # Export CSV
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
    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m.model_dump())

    logger.info(f"Metrics exported to {metrics_file}")

    # Afficher la table
    print("\n=== Table 1 — IntentFlow Evaluation Results ===")
    header = f"{'Defense':<20} {'PCR↓':>6} {'TMR↓':>6} {'ASR↓':>6} {'BTSR↑':>6} {'FBR↓':>6} {'Lat(ms)':>8}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        print(
            f"{m.baseline:<20} {m.pcr:>5.1f}% {m.tmr:>5.1f}% {m.asr:>5.1f}% "
            f"{m.btsr:>5.1f}% {m.fbr:>5.1f}% {m.latency_median:>7.1f}"
        )


if __name__ == "__main__":
    main()
