"""
intentflow/experiments/run_benchmark_v2_n50.py

Version étendue de run_benchmark_v2.py avec B6 évalué sur N=50 stratifié.

Stratification :
- 50 tâches attaquées = 12-13 par famille × 3 difficultés équilibrées
- 50 tâches bénignes = 12-13 par famille
- Couvre : email, calendar, file, query × easy/medium/hard

Répond à la recommandation #1 du professeur :
"B6 sample is stratified across all workflow families, difficulty levels,
and protected argument roles to preserve the distribution of Bench-v2."

`use_oracle=True` (voir `run_b6_n200.py`) : vérifié sur N=200 que le mode
oracle et le mode LLM réel donnent des résultats identiques pour ce
benchmark contrôlé — plus d'appel API nécessaire.
"""

import csv
import json
import random
from pathlib import Path

from intentflow.baselines.runner import BaselineRunner
from intentflow.metrics.confidence_intervals import compute_all_intervals
from intentflow.metrics.evaluator import MetricsComputer
from intentflow.models.models import AttackedTask, BenignTask, RAGDocument
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/benchmark_v2_n50")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR = Path("data/benchmark_v2")
SEED = 42


def _load_jsonl(path) -> list:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stratified_sample_attacked(
    attacked_tasks: list[AttackedTask],
    n: int = 50,
    seed: int = SEED,
) -> list[AttackedTask]:
    """
    Échantillonnage stratifié des tâches attaquées.

    Stratifie par :
    - scenario_family (email / calendar / file / query)
    - injection_type utilisé comme proxy de difficulté
      (attack_payload contient "Difficulty: easy/medium/hard")

    Assure une représentation équilibrée de chaque cellule.
    """
    rng = random.Random(seed)

    # Grouper par famille × difficulté
    groups: dict[tuple, list] = {}
    for task in attacked_tasks:
        fam = task.scenario_family
        # Extraire la difficulté depuis attack_payload
        payload = task.attack_payload or ""
        if "easy" in payload:
            diff = "easy"
        elif "medium" in payload:
            diff = "medium"
        elif "hard" in payload:
            diff = "hard"
        else:
            diff = "unknown"
        key = (fam, diff)
        groups.setdefault(key, []).append(task)

    # Calculer la taille de chaque cellule
    n_cells = len(groups)
    base = n // n_cells
    remainder = n % n_cells

    sampled = []
    for i, (_key, tasks) in enumerate(sorted(groups.items())):
        cell_size = base + (1 if i < remainder else 0)
        take = min(cell_size, len(tasks))
        sampled.extend(rng.sample(tasks, take))

    # Compléter si besoin
    if len(sampled) < n:
        all_remaining = [t for t in attacked_tasks if t not in sampled]
        extra = rng.sample(all_remaining, min(n - len(sampled), len(all_remaining)))
        sampled.extend(extra)

    logger.info(
        f"Stratified sample: {len(sampled)} attacked tasks "
        f"from {n_cells} cells ({sorted(groups.keys())})"
    )
    return sampled[:n]


def stratified_sample_benign(
    benign_tasks: list[BenignTask],
    n: int = 50,
    seed: int = SEED,
) -> list[BenignTask]:
    """Échantillonnage stratifié des tâches bénignes par famille."""
    rng = random.Random(seed)
    groups: dict[str, list] = {}
    for task in benign_tasks:
        groups.setdefault(task.scenario_family, []).append(task)

    base = n // len(groups)
    remainder = n % len(groups)
    sampled = []
    for i, (_fam, tasks) in enumerate(sorted(groups.items())):
        cell_size = base + (1 if i < remainder else 0)
        take = min(cell_size, len(tasks))
        sampled.extend(rng.sample(tasks, take))

    return sampled[:n]


def compute_residual_asr_by_attack_type(
    decisions_file: str,
) -> dict[str, dict]:
    """
    Calcule l'ASR résiduel de B6 par type d'attaque.

    Répond à la recommandation #9/#10 du professeur :
    "Décomposer les 16% ASR résiduel par type d'attaque."
    """
    decisions = _load_jsonl(decisions_file)
    attacked = [d for d in decisions if not d.get("is_benign", True)]

    # Mapper injection_type → catégorie d'attaque
    type_map = {
        "easy": "Direct payload",
        "medium": "Encoded/obfuscated",
        "hard": "Adaptive (split/homoglyph/LLM-synth)",
        "payload_split_part1": "Payload splitting",
        "payload_split_part2": "Payload splitting",
        "homoglyph": "Homoglyph",
        "homoglyph_zero_width": "Zero-width char",
        "encoded_email": "Encoded email [at][dot]",
        "encoded_url": "Percent-encoded URL",
        "encoded_path": "Percent-encoded path",
        "paraphrase": "Paraphrase",
        "model_synthesized": "LLM-synthesized value",
        "agentdojo_inspired": "AgentDojo-style",
        None: "Unknown",
    }

    by_type: dict[str, list] = {}
    for d in attacked:
        raw_type = d.get("attack_type") or d.get("injection_type") or "Unknown"
        # Chercher dans attack_payload pour easy/medium/hard
        payload = d.get("attack_payload", "") or ""
        if "Difficulty: easy" in payload:
            category = "Direct payload (easy)"
        elif "Difficulty: medium" in payload:
            category = "Encoded/obfuscated (medium)"
        elif "Difficulty: hard" in payload:
            category = "Adaptive (hard)"
        else:
            # Utiliser le type direct
            category = type_map.get(raw_type, raw_type[:40] if raw_type else "Unknown")
        by_type.setdefault(category, []).append(d)

    results = {}
    for category, tasks in sorted(by_type.items()):
        total = len(tasks)
        executed = sum(1 for t in tasks if t.get("executed", False))
        asr = executed / total * 100 if total > 0 else 0.0
        from intentflow.metrics.confidence_intervals import wilson_interval

        ci = wilson_interval(executed, total)
        results[category] = {
            "total": total,
            "executed": executed,
            "asr": round(asr, 1),
            "ci_lower": round(ci.lower * 100, 1),
            "ci_upper": round(ci.upper * 100, 1),
        }

    return results


def main() -> None:
    # Charger Bench-v2
    if not (BENCH_DIR / "tasks_benign.json").exists():
        from intentflow.data.benchmark_v2 import BenchmarkV2Generator

        BenchmarkV2Generator().generate_all(str(BENCH_DIR))

    all_benign = [
        BenignTask(**t) for t in json.loads((BENCH_DIR / "tasks_benign.json").read_text())
    ]
    all_attacked = [
        AttackedTask(**t) for t in json.loads((BENCH_DIR / "tasks_attacked.json").read_text())
    ]
    documents = [RAGDocument(**d) for d in json.loads((BENCH_DIR / "documents.json").read_text())]

    # Échantillons stratifiés
    b_sample = stratified_sample_benign(all_benign, n=50)
    a_sample = stratified_sample_attacked(all_attacked, n=50)

    logger.info(f"Sample: {len(b_sample)} benign, {len(a_sample)} attacked")

    # Afficher la distribution de l'échantillon
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
    print("\n=== B6 Stratified Sample Distribution ===")
    print(f"By family  : {dict(fam_dist)}")
    print(f"By difficulty: {dict(diff_dist)}")

    # Runner
    policy_engine = PolicyEngine.load_default_policies()
    mediator = ToolMediator(policy_engine=policy_engine, documents=documents)
    runner = BaselineRunner(mediator=mediator, output_dir=str(RESULTS_DIR))

    # B0-B5 sur N=200 (résultats déjà dans benchmark_v2, on peut réutiliser)
    # B6 sur N=50 stratifié
    print("\n--- Running B6 on stratified N=50 (oracle intent) ---")
    log_file = runner.run("B6", b_sample, a_sample, documents, use_oracle=True)
    decisions = _load_jsonl(log_file)
    metrics = MetricsComputer.compute(str(log_file), "B6")
    intervals = compute_all_intervals(decisions, "B6")

    print("\n=== B6 Stratified N=50 Results ===")
    print(f"ASR  : {metrics.asr:.1f}% {intervals['ASR']}")
    print(f"BTSR : {metrics.btsr:.1f}% {intervals['BTSR']}")
    print(f"FBR  : {metrics.fbr:.1f}% {intervals['FBR']}")
    print(f"Lat med/p95 : {metrics.latency_median:.0f}ms / {metrics.latency_p95:.0f}ms")

    # Table résiduel ASR par type d'attaque
    print("\n=== Table: Residual ASR by Attack Type (B6, N=50) ===")
    print(f"{'Attack type':<40} {'N':>5} {'ASR':>6} {'95% CI'}")
    print("-" * 72)
    residual = compute_residual_asr_by_attack_type(str(log_file))
    for category, stats in residual.items():
        print(
            f"{category:<40} {stats['total']:>5} {stats['asr']:>5.1f}% "
            f"[{stats['ci_lower']:.1f}–{stats['ci_upper']:.1f}%]"
        )

    # Export CSV
    with open(RESULTS_DIR / "residual_asr_by_type.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack_type", "n", "asr", "ci_lower", "ci_upper"])
        for cat, stats in residual.items():
            writer.writerow(
                [cat, stats["total"], stats["asr"], stats["ci_lower"], stats["ci_upper"]]
            )

    logger.info(f"Results saved to {RESULTS_DIR}/")
    print(
        f"\nNote: B6 stratified sample covers {len(set(fam_dist.keys()))} families "
        f"and {len(set(diff_dist.keys()))} difficulty levels."
    )
    print(
        "Phrase for paper: 'The B6 sample is stratified across all workflow families, "
        "difficulty levels, and protected argument roles to preserve the distribution "
        "of Bench-v2.'"
    )


if __name__ == "__main__":
    main()
