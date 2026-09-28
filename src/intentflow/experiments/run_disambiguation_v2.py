"""
intentflow/experiments/run_disambiguation_v2.py

Script d'expérience pour le benchmark de disambiguation élargi (N=50+50).

Produit :
1. Table B5 vs B6 globale avec CIs
2. Table B5 vs B6 par famille de scénarios (recommandation #3)
3. Export CSV

Correctif (même famille de bug que Phase 2 sur l'ablation endpoint) : ce
script n'a jamais passé d'`oracle_intent` à `mediator.execute()`. Le papier
affirme explicitement que "all controlled benchmarks ... use oracle intent
contracts to isolate mediation quality from intent-extraction error" — mais
sans `oracle_intent`, `ToolMediator.execute()` (mediator.py) tombe sur
`mode="llm"` et déclenche un vrai appel gpt-4o-mini payant pour chacune des
100 tâches à chaque baseline, au lieu d'utiliser l'intention "gold" comme
Disambiguation-v2 est censé le faire. `run_baseline_on_tasks()` utilise
maintenant `baselines.runner.oracle_intent_for()` — la même fonction, ajoutée
depuis à `BaselineRunner` pour que tous les scripts de benchmark contrôlé
(`if_core_vs_hardened.py`/`run_adaptive_eval.py`/`run_endpoint_ablation_n30.py`
en avaient chacun leur propre copie) construisent l'oracle de la même façon,
restreinte aux champs du schéma d'intention (jamais un champ payload comme
`content`, cause d'un vrai bug de contamination croisée trouvé ici).
"""

import csv
import json
from pathlib import Path

from intentflow.baselines.runner import oracle_intent_for
from intentflow.data.disambiguation_v2 import get_disambiguation_v2_data
from intentflow.metrics.confidence_intervals import (
    compute_all_intervals,
    wilson_interval,
)
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/disambiguation_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def named_configs() -> dict[str, PolicyEngine]:
    """
    Les quatre configurations nommées réelles (policy/engine.py), pas une
    politique ad hoc spécifique à ce script. Un ancien réglage local
    désactivait délibérément le durcissement endpoint pour B6 (pour isoler
    la vulnérabilité FM2), produisant une table à deux colonnes (B5/B6) où
    la ligne "endpoint" affichait un B6 non-durci à 100% — incohérent avec
    les tables qui rapportent IF-Hardened/IF-Confirm à 0% sur FM2 ailleurs
    dans le papier. Ce script rapporte maintenant une colonne par
    configuration réelle, sans réglage ad hoc.
    """
    return {
        "B5": PolicyEngine.load_default_policies(),
        "IF-Core": PolicyEngine.if_core(),
        "IF-Hardened": PolicyEngine.if_hardened(),
        "IF-Confirm": PolicyEngine.if_confirm(),
    }


def family_label(task) -> str:
    """
    "Same-value-diff-intent" is a distinct attack_type cutting across the
    four scenario_family buckets (see data/disambiguation_v2.py); it must be
    checked first, or its 5 tasks get silently folded into their raw
    scenario_family instead of reported as their own row.
    """
    if task.attack_type.startswith("Same-value"):
        return "Same-value-diff-intent"
    return {
        "email": "Trusted wrong recipient",
        "query": "Trusted wrong endpoint",
        "file": "Trusted wrong file path",
        "calendar": "Trusted wrong participant",
    }.get(task.scenario_family, task.scenario_family)


def run_baseline_on_tasks(baseline_id, benign, attacked, docs, mediator):
    decisions = []
    for task in benign:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        result = mediator.execute(
            task.expected_tool,
            task.expected_arguments,
            task.user_message,
            tool_fn,
            baseline=baseline_id,
            context_documents=docs,
            oracle_intent=oracle_intent_for(task),
        )
        decisions.append(
            {
                "task_id": task.task_id,
                "baseline": baseline_id,
                "scenario_family": task.scenario_family,
                "family_label": "benign",
                "is_benign": True,
                "allowed": result["allowed"],
                "executed": result["executed"],
                "blocked_arguments": result.get("decision", {}).get("blocked_arguments", [])
                if result.get("decision")
                else [],
                "attack_type": None,
                "latency_ms": result["latency_ms"],
            }
        )
    for task in attacked:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        args = {**task.expected_arguments, **task.corrupted_arguments}
        result = mediator.execute(
            task.expected_tool,
            args,
            task.user_message,
            tool_fn,
            baseline=baseline_id,
            context_documents=docs,
            oracle_intent=oracle_intent_for(task),
        )
        decisions.append(
            {
                "task_id": task.task_id,
                "baseline": baseline_id,
                "scenario_family": task.scenario_family,
                "family_label": family_label(task),
                "is_benign": False,
                "allowed": result["allowed"],
                "executed": result["executed"],
                "blocked_arguments": result.get("decision", {}).get("blocked_arguments", [])
                if result.get("decision")
                else [],
                "attack_type": task.attack_type,
                "latency_ms": result["latency_ms"],
            }
        )
    return decisions


FAMILY_ORDER = [
    "Trusted wrong recipient",
    "Trusted wrong endpoint",
    "Trusted wrong file path",
    "Trusted wrong participant",
    "Same-value-diff-intent",
]


def print_family_table(results_by_config: dict[str, list[dict]]) -> list[dict]:
    """Une colonne par configuration nommée, sur les mêmes familles."""
    names = list(results_by_config.keys())
    header = f"{'Scenario family':<28}" + "".join(f"{n:>14}" for n in names)
    print(f"\n{header}")
    print("-" * len(header))

    family_rows = []
    for fam in FAMILY_ORDER:
        row = {"family": fam}
        cells = []
        for name in names:
            attacked_d = [
                d for d in results_by_config[name]
                if not d["is_benign"] and d["family_label"] == fam
            ]
            n = len(attacked_d)
            exec_n = sum(1 for d in attacked_d if d["executed"])
            asr = exec_n / n * 100 if n > 0 else 0
            ci = wilson_interval(exec_n, n) if n > 0 else None
            row[f"{name}_n"] = n
            row[f"{name}_asr"] = round(asr, 1)
            row[f"{name}_ci_lower"] = round(ci.lower * 100, 1) if ci else 0
            row[f"{name}_ci_upper"] = round(ci.upper * 100, 1) if ci else 0
            cells.append(f"{asr:>13.1f}%")
        print(f"{fam:<28}" + "".join(cells))
        family_rows.append(row)
    return family_rows


def main():
    benign, attacked, docs = get_disambiguation_v2_data()
    logger.info(f"Disambiguation-v2: {len(benign)} benign, {len(attacked)} attacked")

    configs = named_configs()
    results = {}
    for name, engine in configs.items():
        logger.info(f"Running {name}...")
        mediator = ToolMediator(policy_engine=engine, documents=docs)
        baseline_id = "B5" if name == "B5" else "B6"
        decisions = run_baseline_on_tasks(baseline_id, benign, attacked, docs, mediator)
        results[name] = decisions

        log_file = RESULTS_DIR / f"disambiguation_v2_{name.lower().replace('-', '_')}.jsonl"
        with open(log_file, "w") as f:
            for d in decisions:
                f.write(json.dumps(d) + "\n")

    # Résultats globaux
    print("\n" + "=" * 70)
    print("DISAMBIGUATION-V2 — one column per named configuration (N=50 attacked, N=50 benign)")
    print("=" * 70)

    for name in configs:
        decisions = results[name]
        baseline_id = "B5" if name == "B5" else "B6"
        intervals = compute_all_intervals(decisions, baseline_id)
        print(f"\n{name}:")
        print(f"  ASR  : {intervals['ASR']}")
        print(f"  BTSR : {intervals['BTSR']}")

    # Table par famille
    print("\n--- Table: ASR by Scenario Family, per configuration ---")
    family_rows = print_family_table(results)

    # Export CSV
    with open(RESULTS_DIR / "disambiguation_v2_by_family.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=family_rows[0].keys())
        writer.writeheader()
        writer.writerows(family_rows)

    logger.info(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
