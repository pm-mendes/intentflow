"""
intentflow/experiments/run_hard_benign_eval.py

Évaluation des hard-benign tasks avec annuaire alias — Points 17 et 18.
Mesure FBR et Confirm Rate sur 50 tâches bénignes difficiles.
"""

import json
import csv
from pathlib import Path
from collections import Counter

from intentflow.data.hard_benign_tasks import get_hard_benign_data
from intentflow.intent.alias_directory import AliasDirectory
from intentflow.models.models import Intent
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.policy.models import ArgumentPolicy
from intentflow.tools.registry import get_tool
from intentflow.metrics.confidence_intervals import wilson_interval
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/hard_benign_eval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_permissive_policies():
    """
    Politiques permettant les valeurs user_intent (alias résolus)
    ET les valeurs explicitement mentionnées dans le message.
    """
    engine = PolicyEngine.load_default_policies()
    # Pour les hard-benign, autoriser user_intent + rag_inferred
    # (les arguments viennent du message utilisateur, pas de rag_document)
    for tool_name in engine.policies:
        for i, arg_policy in enumerate(engine.policies[tool_name].arguments):
            if "user_intent" in arg_policy.allowed_origins:
                # Ajouter user_provided comme origine valide
                if "user_provided" not in arg_policy.allowed_origins:
                    engine.policies[tool_name].arguments[i] = ArgumentPolicy(
                        argument_name=arg_policy.argument_name,
                        allowed_origins=arg_policy.allowed_origins + ["user_provided"],
                        min_confidence=arg_policy.min_confidence,
                        role=arg_policy.role,
                    )
    return engine


def run_task(task, documents, mediator, directory: AliasDirectory) -> dict:
    """Exécute une tâche bénigne difficile avec résolution d'alias."""
    tool_fn = get_tool(task.expected_tool)
    if not tool_fn:
        return {"task_id": task.task_id, "executed": False, "blocked": False,
                "confirm": False, "alias_resolved": False, "scenario_family": task.scenario_family}

    # 1. Construire l'intent de base
    base_intent = Intent(
        action=task.expected_tool,
        parameters={k: v for k, v in task.expected_arguments.items() if v is not None},
        confidence=1.0,
    )

    # 2. Enrichir avec l'annuaire alias
    enriched_params, resolutions = directory.resolve_intent_fields(
        task.user_message, base_intent.parameters
    )
    alias_resolved = len(resolutions) > 0
    oracle_intent = Intent(action=task.expected_tool,
                           parameters=enriched_params, confidence=1.0)

    # 3. Médiateur
    result = mediator.execute(
        task.expected_tool, task.expected_arguments,
        task.user_message, tool_fn, baseline="B6",
        context_documents=documents, oracle_intent=oracle_intent,
    )

    return {
        "task_id": task.task_id,
        "scenario_family": task.scenario_family,
        "executed": result["executed"],
        "blocked": not result["allowed"],
        "confirm": result.get("decision", {}).get("requires_confirm", False)
                   if result.get("decision") else False,
        "alias_resolved": alias_resolved,
        "alias_count": len(resolutions),
        "latency_ms": result.get("latency_ms", 0),
    }


def main():
    hard_benign, documents = get_hard_benign_data()
    directory = AliasDirectory()
    policy_engine = load_permissive_policies()
    mediator = ToolMediator(policy_engine=policy_engine, documents=documents)

    logger.info(f"Hard-benign eval: {len(hard_benign)} tasks")

    decisions = []
    for task in hard_benign:
        result = run_task(task, documents, mediator, directory)
        decisions.append(result)

    # Métriques globales
    n = len(decisions)
    executed = sum(1 for d in decisions if d["executed"])
    blocked = sum(1 for d in decisions if d["blocked"])
    confirmed = sum(1 for d in decisions if d["confirm"])
    alias_resolved = sum(1 for d in decisions if d["alias_resolved"])

    btsr = executed / n * 100 if n > 0 else 0
    fbr = blocked / n * 100 if n > 0 else 0
    confirm_rate = confirmed / n * 100 if n > 0 else 0

    btsr_ci = wilson_interval(executed, n)
    fbr_ci = wilson_interval(blocked, n)

    print(f"\n{'='*65}")
    print(f"HARD-BENIGN EVALUATION (N={n})")
    print(f"{'='*65}")
    print(f"BTSR        : {executed}/{n}={btsr:.1f}% [{btsr_ci.lower*100:.1f},{btsr_ci.upper*100:.1f}%]")
    print(f"FBR         : {blocked}/{n}={fbr:.1f}% [{fbr_ci.lower*100:.1f},{fbr_ci.upper*100:.1f}%]")
    print(f"Confirm rate: {confirmed}/{n}={confirm_rate:.1f}%")
    print(f"Alias resolved: {alias_resolved}/{n}")

    # Par catégorie
    print(f"\nBy scenario family:")
    by_fam = {}
    for d in decisions:
        fam = d["scenario_family"]
        by_fam.setdefault(fam, []).append(d)
    for fam, tasks in sorted(by_fam.items()):
        n_f = len(tasks)
        exec_f = sum(1 for t in tasks if t["executed"])
        block_f = sum(1 for t in tasks if t["blocked"])
        print(f"  {fam:<12} BTSR={exec_f}/{n_f}={exec_f/n_f*100:.0f}% "
              f"FBR={block_f}/{n_f}={block_f/n_f*100:.0f}%")

    # Export
    with open(RESULTS_DIR / "hard_benign_decisions.jsonl", "w") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")

    with open(RESULTS_DIR / "hard_benign_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "btsr", "fbr", "confirm_rate", "alias_resolved",
                         "btsr_ci_lower", "btsr_ci_upper", "fbr_ci_lower", "fbr_ci_upper"])
        writer.writerow([n, round(btsr,1), round(fbr,1), round(confirm_rate,1),
                         alias_resolved, round(btsr_ci.lower*100,1), round(btsr_ci.upper*100,1),
                         round(fbr_ci.lower*100,1), round(fbr_ci.upper*100,1)])

    logger.info(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()