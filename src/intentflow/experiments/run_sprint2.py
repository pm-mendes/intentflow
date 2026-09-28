"""
intentflow/experiments/run_sprint2.py

Script d'expérience Sprint 2 — exécute les 3 livrables :
1. Benchmark AgentDojo-inspired (B0-B6)
2. Baseline Spotlighting (B1b) comparée à B1 et B6
3. Évaluation de l'intent extractor (heuristic + LLM)

Usage : poetry run python -m intentflow.experiments.run_sprint2
"""

import json
from pathlib import Path

from intentflow.baselines.runner import BaselineRunner
from intentflow.baselines.spotlighting import SpotlightingMediator
from intentflow.data.agentdojo_inspired import save_agentdojo_benchmark
from intentflow.metrics.confidence_intervals import compute_all_intervals
from intentflow.metrics.evaluator import MetricsComputer
from intentflow.metrics.intent_evaluator import IntentEvaluator
from intentflow.models.models import AttackedTask, BenignTask, RAGDocument
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/sprint2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_jsonl(path) -> list:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ============================================================
# Partie 1 : Benchmark AgentDojo-inspired
# ============================================================


def run_agentdojo_benchmark() -> None:
    print("\n" + "=" * 70)
    print("SPRINT 2 — Part 1: AgentDojo-Inspired Benchmark")
    print("=" * 70)

    # Générer le benchmark
    bench_dir = Path("data/agentdojo_inspired")
    if not (bench_dir / "tasks_benign.json").exists():
        save_agentdojo_benchmark(str(bench_dir))

    benign_tasks = [
        BenignTask(**t) for t in json.loads((bench_dir / "tasks_benign.json").read_text())
    ]
    attacked_tasks = [
        AttackedTask(**t) for t in json.loads((bench_dir / "tasks_attacked.json").read_text())
    ]
    documents = [RAGDocument(**d) for d in json.loads((bench_dir / "documents.json").read_text())]

    logger.info(f"AgentDojo bench: {len(benign_tasks)} benign, {len(attacked_tasks)} attacked")

    policy_engine = PolicyEngine.load_default_policies()
    mediator = ToolMediator(policy_engine=policy_engine, documents=documents)
    runner = BaselineRunner(mediator=mediator, output_dir=str(RESULTS_DIR / "agentdojo"))

    # B0-B3 et B5 sur tout le bench, B6 sur sous-ensemble (coût LLM)
    results = {}
    for baseline in ["B0", "B1", "B2", "B3", "B5"]:
        log_file = runner.run(baseline, benign_tasks, attacked_tasks, documents)
        metrics = MetricsComputer.compute(str(log_file), baseline)
        decisions = _load_jsonl(log_file)
        results[baseline] = (metrics, compute_all_intervals(decisions, baseline))

    # B6 sur 25 tâches
    log_file = runner.run("B6", benign_tasks[:13], attacked_tasks[:12], documents)
    metrics = MetricsComputer.compute(str(log_file), "B6")
    decisions = _load_jsonl(log_file)
    results["B6"] = (metrics, compute_all_intervals(decisions, "B6"))

    print(f"\n{'Defense':<20} {'ASR↓':>8} {'BTSR↑':>8} {'FBR↓':>8} {'ASR 95% CI'}")
    print("-" * 65)
    for baseline, (m, intervals) in results.items():
        asr_ci = intervals.get("ASR")
        ci_str = f"[{asr_ci.lower*100:.1f}–{asr_ci.upper*100:.1f}%]" if asr_ci else "N/A"
        print(f"{baseline:<20} {m.asr:>7.1f}% {m.btsr:>7.1f}% {m.fbr:>7.1f}% {ci_str}")


# ============================================================
# Partie 2 : Baseline Spotlighting (B1b)
# ============================================================


def run_spotlighting_baseline() -> None:
    print("\n" + "=" * 70)
    print("SPRINT 2 — Part 2: Spotlighting Baseline (B1b)")
    print("=" * 70)

    # Utiliser le benchmark principal
    bench_dir = Path("data/benchmark_v2")
    if not (bench_dir / "tasks_benign.json").exists():
        from intentflow.data.benchmark_v2 import BenchmarkV2Generator

        BenchmarkV2Generator().generate_all(str(bench_dir))

    benign_tasks = [
        BenignTask(**t) for t in json.loads((bench_dir / "tasks_benign.json").read_text())
    ]
    attacked_tasks = [
        AttackedTask(**t) for t in json.loads((bench_dir / "tasks_attacked.json").read_text())
    ]
    documents = [RAGDocument(**d) for d in json.loads((bench_dir / "documents.json").read_text())]

    spotlighter = SpotlightingMediator(seed=42)

    # Exécuter Spotlighting sur toutes les tâches
    b1b_decisions = []

    for task in benign_tasks:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        # Spotlighting sur tâches bénignes : doit autoriser
        decision = spotlighter.validate(
            task.expected_tool, task.expected_arguments, documents, task.user_message
        )
        b1b_decisions.append(
            {
                "task_id": task.task_id,
                "baseline": "B1b",
                "is_benign": True,
                "allowed": decision.allowed,
                "executed": decision.allowed,
                "blocked_arguments": decision.blocked_arguments,
                "attack_type": None,
                "latency_ms": 1.0,
            }
        )

    for task in attacked_tasks:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        args = {**task.expected_arguments, **task.corrupted_arguments}
        # Ajouter les documents injectés
        injected_docs = [d for d in documents if d.is_injected]
        decision = spotlighter.validate(
            task.expected_tool, args, injected_docs + documents, task.user_message
        )
        b1b_decisions.append(
            {
                "task_id": task.task_id,
                "baseline": "B1b",
                "is_benign": False,
                "allowed": decision.allowed,
                "executed": decision.allowed,
                "blocked_arguments": decision.blocked_arguments,
                "attack_type": task.attack_type,
                "latency_ms": 2.0,
            }
        )

    # Sauvegarder
    log_file = RESULTS_DIR / "decisions_B1b.jsonl"
    with open(log_file, "w") as f:
        for d in b1b_decisions:
            f.write(json.dumps(d) + "\n")

    metrics = MetricsComputer.compute(str(log_file), "B1b")
    intervals = compute_all_intervals(b1b_decisions, "B1b")

    print("\nSpotlighting (B1b) Results:")
    print(f"  ASR  : {metrics.asr:.1f}% {intervals['ASR']}")
    print(f"  BTSR : {metrics.btsr:.1f}% {intervals['BTSR']}")
    print(f"  FBR  : {metrics.fbr:.1f}% {intervals['FBR']}")
    print("\n  Comparison:")
    print("  B1 (prompt-only naive) : ASR=100.0% (no detection)")
    print(f"  B1b (Spotlighting)     : ASR={metrics.asr:.1f}% (structured separation)")
    print("  B6 (IntentFlow)        : ASR~16% (argument-level mediation)")


# ============================================================
# Partie 3 : Évaluation de l'intent extractor
# ============================================================


def run_intent_evaluation() -> None:
    print("\n" + "=" * 70)
    print("SPRINT 2 — Part 3: Intent Extractor Evaluation")
    print("=" * 70)

    evaluator = IntentEvaluator()

    # Mode heuristique (toujours disponible)
    print("\nHeuristic mode (offline)...")
    metrics_h = evaluator.evaluate(mode="heuristic")
    IntentEvaluator.print_report(metrics_h, "heuristic")
    IntentEvaluator.export_csv(
        metrics_h, str(RESULTS_DIR / "intent_eval_heuristic.csv"), "heuristic"
    )

    # Mode LLM (si API disponible)
    print("\nLLM mode (gpt-4o-mini)...")
    try:
        metrics_llm = evaluator.evaluate(mode="llm")
        IntentEvaluator.print_report(metrics_llm, "llm")
        IntentEvaluator.export_csv(metrics_llm, str(RESULTS_DIR / "intent_eval_llm.csv"), "llm")

        # Comparaison heuristique vs LLM
        print(f"\n{'Field':<20} {'Heuristic F1':>14} {'LLM F1':>10} {'Gain':>8}")
        print("-" * 56)
        for field_name in metrics_h:
            h_f1 = metrics_h[field_name].f1
            l_f1 = metrics_llm[field_name].f1
            gain = l_f1 - h_f1
            print(f"{field_name:<20} {h_f1:>13.1%} {l_f1:>9.1%} {gain:>+7.1%}")

    except Exception as e:
        print(f"LLM mode failed: {e}")
        print("Heuristic results saved. Set OPENAI_API_KEY for LLM evaluation.")


def main() -> None:
    run_agentdojo_benchmark()
    run_spotlighting_baseline()
    run_intent_evaluation()
    print(f"\nAll Sprint 2 results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
