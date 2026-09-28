"""
intentflow/experiments/run_disambiguation.py

Script d'expérience dédié aux scénarios de disambiguation B5 vs B6.

Usage : poetry run python -m intentflow.experiments.run_disambiguation

Ce script :
1. Charge les scénarios "trusted but unauthorized"
2. Exécute B5 (provenance-only) et B6 (IntentFlow complet)
3. Affiche côte à côte les décisions B5 vs B6
4. Calcule les métriques avec intervalles de confiance Wilson
5. Exporte les résultats dans data/results/disambiguation_*.jsonl
   et data/results/disambiguation_summary.csv

Résultat attendu :
- B5 accepte les arguments trusted-but-unauthorized (FBR élevé sur ces scénarios)
- B6 bloque grâce à l'intent conditioning (FBR faible)
→ Démonstration que l'intention apporte un gain net mesurable.
"""

import csv
import json
from pathlib import Path

from intentflow.data.disambiguation_scenarios import get_disambiguation_data
from intentflow.intent.extractor import IntentExtractor
from intentflow.metrics.confidence_intervals import compute_all_intervals, format_intervals_table
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.policy.models import ArgumentPolicy
from intentflow.tools.registry import get_tool
from intentflow.utils.logger import logger


def load_strict_policies() -> "PolicyEngine":
    """
    Politique stricte pour les scénarios de disambiguation.
    participants et endpoint restreints à user_intent uniquement,
    ce qui force B6 à utiliser l'intent pour valider ces arguments.
    """
    from intentflow.policy.engine import PolicyEngine

    engine = PolicyEngine.load_default_policies()

    # Restreindre participants à user_intent
    engine.policies["create_calendar_event"].arguments = [
        p
        if p.argument_name != "participants"
        else ArgumentPolicy(
            argument_name="participants",
            allowed_origins=["user_intent"],
            min_confidence=0.80,
            role="recipient",
        )
        for p in engine.policies["create_calendar_event"].arguments
    ]

    # Restreindre endpoint à user_intent
    engine.policies["run_query"].arguments = [
        p
        if p.argument_name != "endpoint"
        else ArgumentPolicy(
            argument_name="endpoint",
            allowed_origins=["user_intent"],
            min_confidence=0.85,
            role="locator",
        )
        for p in engine.policies["run_query"].arguments
    ]

    # Restreindre recipient à user_intent (déjà le cas, on renforce)
    engine.policies["send_email"].arguments = [
        p
        if p.argument_name != "recipient"
        else ArgumentPolicy(
            argument_name="recipient",
            allowed_origins=["user_intent"],
            min_confidence=0.85,
            role="recipient",
        )
        for p in engine.policies["send_email"].arguments
    ]

    return engine


RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_baseline_on_tasks(
    baseline: str,
    benign_tasks: list,
    attacked_tasks: list,
    documents: list,
    mediator: ToolMediator,
) -> list[dict]:
    """Exécute une baseline sur les scénarios de disambiguation."""
    decisions = []

    for task in benign_tasks:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        result = mediator.execute(
            tool_name=task.expected_tool,
            arguments=task.expected_arguments,
            user_message=task.user_message,
            tool_fn=tool_fn,
            baseline=baseline,
            context_documents=documents,
        )
        decisions.append(
            {
                "task_id": task.task_id,
                "baseline": baseline,
                "tool": task.expected_tool,
                "user_message": task.user_message,
                "arguments": task.expected_arguments,
                "allowed": result["allowed"],
                "executed": result["executed"],
                "blocked_arguments": result.get("decision", {}).get("blocked_arguments", [])
                if result.get("decision")
                else [],
                "reasons": result.get("decision", {}).get("reasons", [])
                if result.get("decision")
                else [],
                "is_benign": True,
                "attack_type": None,
                "latency_ms": result["latency_ms"],
            }
        )

    for task in attacked_tasks:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        # Fusionner arguments légitimes + arguments corrompus
        args = {**task.expected_arguments, **task.corrupted_arguments}
        result = mediator.execute(
            tool_name=task.expected_tool,
            arguments=args,
            user_message=task.user_message,
            tool_fn=tool_fn,
            baseline=baseline,
            context_documents=documents,
        )
        decisions.append(
            {
                "task_id": task.task_id,
                "baseline": baseline,
                "tool": task.expected_tool,
                "user_message": task.user_message,
                "arguments": args,
                "allowed": result["allowed"],
                "executed": result["executed"],
                "blocked_arguments": result.get("decision", {}).get("blocked_arguments", [])
                if result.get("decision")
                else [],
                "reasons": result.get("decision", {}).get("reasons", [])
                if result.get("decision")
                else [],
                "is_benign": False,
                "attack_type": task.attack_type,
                "latency_ms": result["latency_ms"],
            }
        )

    return decisions


def main() -> None:
    benign_tasks, attacked_tasks, documents = get_disambiguation_data()

    policy_engine = load_strict_policies()
    intent_extractor = IntentExtractor()
    mediator = ToolMediator(
        policy_engine=policy_engine,
        intent_extractor=intent_extractor,
        documents=documents,
    )

    results = {}

    for baseline in ["B5", "B6"]:
        logger.info(f"Running disambiguation experiment: {baseline}")
        decisions = run_baseline_on_tasks(
            baseline, benign_tasks, attacked_tasks, documents, mediator
        )
        results[baseline] = decisions

        # Sauvegarder JSONL
        log_file = RESULTS_DIR / f"disambiguation_{baseline}.jsonl"
        with open(log_file, "w") as f:
            for d in decisions:
                f.write(json.dumps(d) + "\n")
        logger.info(f"Saved {len(decisions)} decisions → {log_file}")

    # ---- Affichage comparatif B5 vs B6 ----
    print("\n" + "=" * 80)
    print("DISAMBIGUATION EXPERIMENT — B5 (Provenance-only) vs B6 (IntentFlow)")
    print("=" * 80)
    print(f"\n{'Task ID':<30} {'Attack Type':<45} {'B5':>6} {'B6':>6}")
    print("-" * 90)

    attacked_b5 = {d["task_id"]: d for d in results["B5"] if not d["is_benign"]}
    attacked_b6 = {d["task_id"]: d for d in results["B6"] if not d["is_benign"]}

    b5_blocks = 0
    b6_blocks = 0

    for task_id in sorted(attacked_b5.keys()):
        d5 = attacked_b5[task_id]
        d6 = attacked_b6.get(task_id, {})
        b5_dec = "BLOCK" if not d5.get("allowed") else "ALLOW"
        b6_dec = "BLOCK" if not d6.get("allowed") else "ALLOW"
        attack_type = d5.get("attack_type", "")[:44]
        print(f"{task_id:<30} {attack_type:<45} {b5_dec:>6} {b6_dec:>6}")
        if b5_dec == "BLOCK":
            b5_blocks += 1
        if b6_dec == "BLOCK":
            b6_blocks += 1

    n_attacked = len(attacked_b5)
    print("-" * 90)
    print(f"{'TOTAL BLOCKED':<76} {b5_blocks:>3}/{n_attacked} {b6_blocks:>3}/{n_attacked}")

    # ---- Intervalles de confiance ----
    print("\n" + "=" * 80)
    print("WILSON CONFIDENCE INTERVALS (95%)")
    print("=" * 80)

    intervals_by_baseline = {}
    for baseline, decisions in results.items():
        intervals_by_baseline[baseline] = compute_all_intervals(decisions, baseline)

    print(format_intervals_table(intervals_by_baseline))

    # ---- Export CSV résumé ----
    summary_file = RESULTS_DIR / "disambiguation_summary.csv"
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "baseline",
                "n_benign",
                "n_attacked",
                "asr_count",
                "asr_total",
                "asr_pct",
                "asr_ci_lower",
                "asr_ci_upper",
                "btsr_count",
                "btsr_total",
                "btsr_pct",
                "btsr_ci_lower",
                "btsr_ci_upper",
                "fbr_count",
                "fbr_total",
                "fbr_pct",
                "fbr_ci_lower",
                "fbr_ci_upper",
            ]
        )
        for baseline, intervals in intervals_by_baseline.items():
            asr = intervals["ASR"]
            btsr = intervals["BTSR"]
            fbr = intervals["FBR"]
            n_b = len([d for d in results[baseline] if d["is_benign"]])
            n_a = len([d for d in results[baseline] if not d["is_benign"]])
            writer.writerow(
                [
                    baseline,
                    n_b,
                    n_a,
                    asr.count,
                    asr.total,
                    round(asr.proportion * 100, 1),
                    round(asr.lower * 100, 1),
                    round(asr.upper * 100, 1),
                    btsr.count,
                    btsr.total,
                    round(btsr.proportion * 100, 1),
                    round(btsr.lower * 100, 1),
                    round(btsr.upper * 100, 1),
                    fbr.count,
                    fbr.total,
                    round(fbr.proportion * 100, 1),
                    round(fbr.lower * 100, 1),
                    round(fbr.upper * 100, 1),
                ]
            )

    logger.info(f"Summary exported → {summary_file}")
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
