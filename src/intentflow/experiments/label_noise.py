"""
intentflow/experiments/label_noise.py

Expérience de robustesse au bruit de labels de provenance.

Répond à la recommandation #14 du professeur :
"Tester la robustesse à des labels imparfaits : 0%, 5%, 10%, 20%, 30%"

Dans le monde réel, les labels de provenance sont dérivés de métadonnées
(source domain, sender, file path) qui peuvent être incorrects :
- document externe marqué interne (le plus dangereux)
- document interne marqué unknown
- forwarded external content marqué trusted
- source domain manquant → unknown_origin

Ce bruit affecte les décisions de B5 et B6 différemment :
- B5 (provenance-only) : dégradation linéaire car tout repose sur les labels
- B6 (IntentFlow) : plus robuste car l'intent cross-référence les valeurs

Métriques mesurées par niveau de bruit :
- ASR : fraction d'attaques exécutées malgré le médiateur
- BTSR : fraction de tâches bénignes réussies
- FBR : fraction de tâches bénignes bloquées à tort
- Confirm rate : fraction de décisions nécessitant confirmation
"""

import csv
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Literal

from intentflow.models.models import AttackedTask, BenignTask, RAGDocument
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
from intentflow.utils.logger import logger

SEED = 42
RESULTS_DIR = Path("data/results/label_noise")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NoiseType = Literal[
    "external_marked_internal",  # injected doc marqué trusted_internal → le plus dangereux
    "internal_marked_unknown",  # benign doc marqué unknown_origin
    "forwarded_marked_trusted",  # external forwarded marqué trusted_internal
]


def apply_label_noise(
    documents: list[RAGDocument],
    noise_rate: float,
    seed: int = SEED,
) -> list[RAGDocument]:
    """
    Applique du bruit aux labels de provenance des documents.

    Types de bruit appliqués proportionnellement :
    - 60% : external_marked_internal (injected → trusted_internal)
             C'est le bruit le plus dangereux : l'attaquant passe pour interne
    - 25% : internal_marked_unknown (benign → unknown_origin)
             Génère des faux positifs : des docs légitimes sont traités avec méfiance
    - 15% : forwarded_marked_trusted (external → trusted_internal)
             Simule un email forwardé mal étiqueté

    Args:
        documents: Liste de RAGDocument originaux
        noise_rate: Fraction de documents dont le label est corrompu [0.0, 1.0]
        seed: Graine pour la reproductibilité

    Returns:
        Liste de RAGDocument avec labels bruités
    """
    if noise_rate == 0.0:
        return documents

    rng = random.Random(seed)
    noisy_docs = [deepcopy(d) for d in documents]

    n_noisy = max(1, int(len(noisy_docs) * noise_rate))
    indices_to_corrupt = rng.sample(range(len(noisy_docs)), min(n_noisy, len(noisy_docs)))

    for idx in indices_to_corrupt:
        doc = noisy_docs[idx]
        roll = rng.random()

        if doc.is_injected:
            # Document injecté → le marquer comme trusted_internal (pire cas)
            # Probabilité 60% : external_marked_internal
            if roll < 0.60:
                noisy_docs[idx] = RAGDocument(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    is_injected=False,  # Le médiateur ne sait plus que c'est injecté
                    injection_type=doc.injection_type,
                    source_domain="trusted_internal",  # Label corrompu
                )
            # Probabilité 40% : laisser tel quel (external mais non marqué injecté)
            else:
                noisy_docs[idx] = RAGDocument(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    is_injected=False,
                    injection_type=doc.injection_type,
                    source_domain="untrusted_external",
                )
        else:
            # Document bénin → le marquer comme unknown (faux positifs potentiels)
            if roll < 0.25:
                noisy_docs[idx] = RAGDocument(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    is_injected=doc.is_injected,
                    source_domain="unknown_origin",
                )
            # Sinon laisser tel quel

    n_corrupted = sum(
        1
        for orig, noisy in zip(documents, noisy_docs, strict=False)
        if orig.source_domain != noisy.source_domain or orig.is_injected != noisy.is_injected
    )
    logger.info(f"Label noise {noise_rate:.0%}: {n_corrupted}/{len(documents)} docs corrupted")
    return noisy_docs


def run_baseline_with_noise(
    baseline: str,
    benign_tasks: list[BenignTask],
    attacked_tasks: list[AttackedTask],
    documents_noisy: list[RAGDocument],
    mediator: ToolMediator,
) -> list[dict]:
    """Exécute une baseline sur des documents avec labels bruités."""
    decisions = []

    for task in benign_tasks:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        result = mediator.execute(
            task.expected_tool,
            task.expected_arguments,
            task.user_message,
            tool_fn,
            baseline=baseline,
            context_documents=documents_noisy,
        )
        decisions.append(
            {
                "task_id": task.task_id,
                "baseline": baseline,
                "is_benign": True,
                "allowed": result["allowed"],
                "executed": result["executed"],
                "blocked_arguments": result.get("decision", {}).get("blocked_arguments", [])
                if result.get("decision")
                else [],
                "latency_ms": result["latency_ms"],
            }
        )

    for task in attacked_tasks:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        args = {**task.expected_arguments, **task.corrupted_arguments}
        result = mediator.execute(
            task.expected_tool,
            args,
            task.user_message,
            tool_fn,
            baseline=baseline,
            context_documents=documents_noisy,
        )
        decisions.append(
            {
                "task_id": task.task_id,
                "baseline": baseline,
                "is_benign": False,
                "allowed": result["allowed"],
                "executed": result["executed"],
                "blocked_arguments": result.get("decision", {}).get("blocked_arguments", [])
                if result.get("decision")
                else [],
                "latency_ms": result["latency_ms"],
            }
        )

    return decisions


def run_label_noise_experiment(
    noise_rates: list[float] = None,
    baselines: list[str] = None,
    n_reps: int = 3,
) -> None:
    """
    Lance l'expérience complète de robustesse au bruit.

    Args:
        noise_rates: Niveaux de bruit à tester
        baselines: Baselines à évaluer (B5 et B6 sont les plus intéressantes)
        n_reps: Nombre de répétitions par niveau (pour stabiliser les moyennes)
    """
    if noise_rates is None:
        noise_rates = [0.0, 0.05, 0.10, 0.20, 0.30]
    if baselines is None:
        baselines = ["B5", "B6"]

    # Charger le benchmark principal (Bench-v1, N=20, rapide)
    bench_dir = Path("data")
    benign_tasks = [
        BenignTask(**t) for t in json.loads((bench_dir / "tasks_benign.json").read_text())
    ]
    attacked_tasks = [
        AttackedTask(**t) for t in json.loads((bench_dir / "tasks_attacked.json").read_text())
    ]
    documents = [RAGDocument(**d) for d in json.loads((bench_dir / "documents.json").read_text())]

    policy_engine = PolicyEngine.load_default_policies()
    mediator = ToolMediator(policy_engine=policy_engine, documents=documents)

    results = {}  # (baseline, noise_rate) → averaged metrics

    for baseline in baselines:
        for noise_rate in noise_rates:
            logger.info(f"Running {baseline} at noise={noise_rate:.0%}...")

            # Répéter avec différentes graines pour stabiliser
            all_asr, all_btsr, all_fbr = [], [], []

            for rep in range(n_reps):
                noisy_docs = apply_label_noise(documents, noise_rate, seed=SEED + rep * 100)
                decisions = run_baseline_with_noise(
                    baseline, benign_tasks, attacked_tasks, noisy_docs, mediator
                )

                attacked = [d for d in decisions if not d["is_benign"]]
                benign = [d for d in decisions if d["is_benign"]]
                n_a = len(attacked)
                n_b = len(benign)

                asr = sum(1 for d in attacked if d["executed"]) / n_a * 100 if n_a > 0 else 0
                btsr = sum(1 for d in benign if d["executed"]) / n_b * 100 if n_b > 0 else 0
                fbr = sum(1 for d in benign if not d["allowed"]) / n_b * 100 if n_b > 0 else 0

                all_asr.append(asr)
                all_btsr.append(btsr)
                all_fbr.append(fbr)

            # Moyennes sur les répétitions
            avg_asr = sum(all_asr) / n_reps
            avg_btsr = sum(all_btsr) / n_reps
            avg_fbr = sum(all_fbr) / n_reps

            results[(baseline, noise_rate)] = {
                "baseline": baseline,
                "noise_rate": noise_rate,
                "asr": round(avg_asr, 1),
                "btsr": round(avg_btsr, 1),
                "fbr": round(avg_fbr, 1),
            }

            logger.info(
                f"  {baseline} noise={noise_rate:.0%}: "
                f"ASR={avg_asr:.1f}% BTSR={avg_btsr:.1f}% FBR={avg_fbr:.1f}%"
            )

    # Afficher les tables
    print("\n" + "=" * 70)
    print("LABEL NOISE ROBUSTNESS — ASR by noise level")
    print("=" * 70)
    print(f"{'Noise':>8}", end="")
    for baseline in baselines:
        print(f"  {baseline+' ASR':>12} {baseline+' BTSR':>12} {baseline+' FBR':>10}", end="")
    print()
    print("-" * 70)

    for noise_rate in noise_rates:
        print(f"{noise_rate:>7.0%}", end="")
        for baseline in baselines:
            r = results[(baseline, noise_rate)]
            print(f"  {r['asr']:>11.1f}% {r['btsr']:>11.1f}% {r['fbr']:>9.1f}%", end="")
        print()

    # Export CSV
    csv_file = RESULTS_DIR / "label_noise_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["baseline", "noise_rate", "asr", "btsr", "fbr"])
        writer.writeheader()
        for r in results.values():
            writer.writerow(r)

    logger.info(f"Results saved to {csv_file}")


if __name__ == "__main__":
    run_label_noise_experiment(
        noise_rates=[0.0, 0.05, 0.10, 0.20, 0.30],
        baselines=["B5", "B6"],
        n_reps=3,
    )
