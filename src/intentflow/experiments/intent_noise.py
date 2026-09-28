"""
intentflow/experiments/intent_noise.py

Intent extraction robustness — recommandation Priorité 2.

Injecte du bruit dans le contrat d'intention et mesure l'impact
sur ASR/BTSR/FBR/Confirm rate.

Types de bruit :
- missing_recipient  : champ recipient absent de l'intent
- wrong_recipient    : recipient incorrect dans l'intent
- missing_endpoint   : endpoint absent de l'intent
- partial_path       : path partiel (manque le nom de fichier)
- ambiguous_alias    : alias vague ("manager", "finance team")

Répond à la critique : "IntentFlow is only as reliable as its intent extractor."
"""

import copy
import csv
import json
import random
from pathlib import Path

from intentflow.models.models import AttackedTask, BenignTask, Intent, RAGDocument
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
from intentflow.utils.logger import logger

SEED = 42
random.seed(SEED)
RESULTS_DIR = Path("data/results/intent_noise")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def apply_intent_noise(
    intent: Intent,
    noise_type: str,
    rng: random.Random,
) -> Intent:
    """
    Applique un type de bruit à un contrat d'intention.

    Args:
        intent: Intent original
        noise_type: Type de bruit à appliquer
        rng: Générateur aléatoire

    Returns:
        Intent bruité
    """
    noisy = Intent(
        action=intent.action,
        parameters=copy.deepcopy(intent.parameters),
        confidence=intent.confidence,
    )

    if noise_type == "missing_recipient":
        noisy.parameters.pop("recipient", None)
        noisy.parameters.pop("participants", None)

    elif noise_type == "wrong_recipient":
        wrong = rng.choice(["wrong@company.com", "other@company.com", "unknown@company.com"])
        if "recipient" in noisy.parameters:
            noisy.parameters["recipient"] = wrong
        if "participants" in noisy.parameters:
            noisy.parameters["participants"] = [wrong]

    elif noise_type == "missing_endpoint":
        noisy.parameters.pop("endpoint", None)

    elif noise_type == "partial_path":
        if "path" in noisy.parameters:
            # Garder seulement le dossier, pas le nom de fichier
            path = noisy.parameters["path"]
            noisy.parameters["path"] = "/".join(path.split("/")[:-1]) + "/"

    elif noise_type == "ambiguous_alias":
        # Remplacer les entités par des alias vagues
        replacements = {
            "recipient": "my manager",
            "participants": ["the team"],
            "endpoint": "the database",
            "path": "the reports folder",
        }
        for key, alias in replacements.items():
            if key in noisy.parameters:
                noisy.parameters[key] = alias

    return noisy


def evaluate_intent_noise(
    benign_tasks: list[BenignTask],
    attacked_tasks: list[AttackedTask],
    documents: list[RAGDocument],
    noise_types: list[str],
    n_reps: int = 3,
) -> dict[str, dict]:
    """
    Évalue l'impact du bruit d'intention sur les métriques.

    Pour chaque type de bruit, simule l'impact en modifiant
    le contrat d'intention oracle avant la validation.
    """
    policy_engine = PolicyEngine.load_default_policies()
    results = {}

    for noise_type in ["clean"] + noise_types:
        all_asr, all_btsr, all_fbr, all_confirm = [], [], [], []

        for rep in range(n_reps):
            rep_rng = random.Random(SEED + rep * 7)
            executed_attacked = 0
            executed_benign = 0
            blocked_benign = 0
            confirmations = 0

            for task in benign_tasks:
                tool_fn = get_tool(task.expected_tool)
                if not tool_fn:
                    continue

                # Construire un intent oracle depuis les expected_arguments
                oracle_intent = Intent(
                    action=task.expected_tool,
                    parameters=copy.deepcopy(task.expected_arguments),
                    confidence=1.0,
                )

                # Appliquer le bruit
                if noise_type != "clean":
                    oracle_intent = apply_intent_noise(oracle_intent, noise_type, rep_rng)

                # Simuler la décision de médiation
                # On vérifie si les arguments passent la politique avec l'intent bruité
                from intentflow.provenance.tracker import ProvenanceTracker

                tracker = ProvenanceTracker(documents, intent=oracle_intent)
                provenance = tracker.track_all_arguments(task.expected_arguments)
                decision = policy_engine.validate(
                    task.expected_tool, task.expected_arguments, provenance, intent=oracle_intent
                )

                if decision.allowed:
                    executed_benign += 1
                else:
                    blocked_benign += 1
                    # Si bruit → bloquer une tâche bénigne = confirm plutôt que block dur
                    if noise_type != "clean":
                        confirmations += 1

            for task in attacked_tasks:
                tool_fn = get_tool(task.expected_tool)
                if not tool_fn:
                    continue

                # Intent oracle propre pour les tâches attaquées
                oracle_intent = Intent(
                    action=task.expected_tool,
                    parameters=copy.deepcopy(task.expected_arguments),
                    confidence=1.0,
                )

                if noise_type != "clean":
                    oracle_intent = apply_intent_noise(oracle_intent, noise_type, rep_rng)

                args = {**task.expected_arguments, **task.corrupted_arguments}
                from intentflow.provenance.tracker import ProvenanceTracker

                tracker = ProvenanceTracker(documents, intent=oracle_intent)
                provenance = tracker.track_all_arguments(args)
                decision = policy_engine.validate(
                    task.expected_tool, args, provenance, intent=oracle_intent
                )

                if decision.allowed:
                    executed_attacked += 1

            n_a = len(attacked_tasks)
            n_b = len(benign_tasks)
            all_asr.append(executed_attacked / n_a * 100 if n_a > 0 else 0)
            all_btsr.append(executed_benign / n_b * 100 if n_b > 0 else 0)
            all_fbr.append(blocked_benign / n_b * 100 if n_b > 0 else 0)
            all_confirm.append(confirmations / n_b * 100 if n_b > 0 else 0)

        results[noise_type] = {
            "asr": round(sum(all_asr) / n_reps, 1),
            "btsr": round(sum(all_btsr) / n_reps, 1),
            "fbr": round(sum(all_fbr) / n_reps, 1),
            "confirm_rate": round(sum(all_confirm) / n_reps, 1),
        }

    return results


def main() -> None:
    # Utiliser Bench-v1 (rapide, N=20)
    bench_dir = Path("data")
    benign = [BenignTask(**t) for t in json.loads((bench_dir / "tasks_benign.json").read_text())]
    attacked = [
        AttackedTask(**t) for t in json.loads((bench_dir / "tasks_attacked.json").read_text())
    ]
    documents = [RAGDocument(**d) for d in json.loads((bench_dir / "documents.json").read_text())]

    noise_types = [
        "missing_recipient",
        "wrong_recipient",
        "missing_endpoint",
        "partial_path",
        "ambiguous_alias",
    ]

    print("\n" + "=" * 72)
    print("INTENT EXTRACTION ROBUSTNESS — Impact of intent noise on B6")
    print("=" * 72)
    print(f"{'Intent noise type':<25} {'ASR↓':>7} {'BTSR↑':>7} {'FBR↓':>7} {'Confirm%':>10}")
    print("-" * 60)

    results = evaluate_intent_noise(benign, attacked, documents, noise_types, n_reps=3)

    rows = []
    for noise_type, r in results.items():
        label = (
            noise_type.replace("_", " ").title() if noise_type != "clean" else "Clean (no noise)"
        )
        print(
            f"{label:<25} {r['asr']:>6.1f}% {r['btsr']:>6.1f}% {r['fbr']:>6.1f}% {r['confirm_rate']:>9.1f}%"
        )
        rows.append({"noise_type": label, **r})

    with open(RESULTS_DIR / "intent_noise.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["noise_type", "asr", "btsr", "fbr", "confirm_rate"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Intent noise results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
