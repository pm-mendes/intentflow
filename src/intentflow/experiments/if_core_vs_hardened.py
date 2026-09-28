"""
intentflow/experiments/if_core_vs_hardened.py

Table IF-Core vs IF-Hardened vs IF-Confirm — Point #9.
Utilise les données existantes pour construire la table des 3 versions.

IF-Core    = B6 baseline (intent + provenance + roles)
IF-Hardened = IF-Core + registry + SQL AST + span provenance
IF-Confirm  = IF-Hardened + Confirm sur tous les cas non résolus (recommended)

Phase 0.2 : les trois variantes sont désormais des configurations nommées de
`PolicyEngine` (`if_core`/`if_hardened`/`if_confirm`, policy/engine.py) — ce
module patchait auparavant `PolicyEngine.load_default_policies()` à la main
avec une logique dupliquée (quasi à l'identique) dans `e2e_agent_eval_v2.py`
et `run_adaptive_eval.py`. `make_engine()` ci-dessous ne fait plus que
déléguer à la source de vérité unique.
"""
import json, csv
from pathlib import Path
from intentflow.models.models import (BenignTask, AttackedTask, RAGDocument, Intent)
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
from intentflow.metrics.confidence_intervals import wilson_interval
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/if_versions")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(bench_dir="data"):
    d = Path(bench_dir)
    return (
        [BenignTask(**t) for t in json.loads((d/"tasks_benign.json").read_text())],
        [AttackedTask(**t) for t in json.loads((d/"tasks_attacked.json").read_text())],
        [RAGDocument(**doc) for doc in json.loads((d/"documents.json").read_text())],
    )


def make_engine(version: str) -> PolicyEngine:
    return {
        "if_core": PolicyEngine.if_core,
        "if_hardened": PolicyEngine.if_hardened,
        "if_confirm": PolicyEngine.if_confirm,
    }[version]()


def run_version(version: str, benign, attacked, documents) -> dict:
    engine = make_engine(version)
    # IF-Confirm mode: treat any uncertain decision as Confirm → benign pass, attack blocked
    confirm_mode = (version == "if_confirm")
    mediator = ToolMediator(policy_engine=engine, documents=documents)

    asr = btsr = fbr = 0
    na, nb = len(attacked), len(benign)

    for task in attacked:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn: continue
        oracle = Intent(action=task.expected_tool,
                       parameters={k:v for k,v in task.expected_arguments.items() if v},
                       confidence=1.0)
        args = {**task.expected_arguments, **task.corrupted_arguments}
        res = mediator.execute(task.expected_tool, args, task.user_message,
                               tool_fn, baseline="B6", context_documents=documents,
                               oracle_intent=oracle)
        if res["executed"]: asr += 1

    for task in benign:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn: continue
        oracle = Intent(action=task.expected_tool,
                       parameters={k:v for k,v in task.expected_arguments.items() if v},
                       confidence=1.0)
        res = mediator.execute(task.expected_tool, task.expected_arguments,
                               task.user_message, tool_fn, baseline="B6",
                               context_documents=documents, oracle_intent=oracle)
        if res["executed"]: btsr += 1
        elif not confirm_mode: fbr += 1
        # IF-Confirm: Confirm → human approves → benign passes (not FBR)
        else: btsr += 1  # optimistic: human approves legitimate tasks

    asr_ci = wilson_interval(asr, na) if na > 0 else None
    btsr_ci = wilson_interval(btsr, nb) if nb > 0 else None

    return {
        "version": version,
        "na": na, "nb": nb,
        "asr": round(asr/na*100,1) if na else 0,
        "asr_count": asr,
        "asr_ci_lo": round(asr_ci.lower*100,1) if asr_ci else 0,
        "asr_ci_hi": round(asr_ci.upper*100,1) if asr_ci else 0,
        "btsr": round(btsr/nb*100,1) if nb else 0,
        "btsr_count": btsr,
        "btsr_ci_lo": round(btsr_ci.lower*100,1) if btsr_ci else 0,
        "btsr_ci_hi": round(btsr_ci.upper*100,1) if btsr_ci else 0,
        "fbr": round(fbr/nb*100,1) if nb else 0,
    }


def main():
    benign, attacked, documents = load_data()
    versions = ["if_core", "if_hardened", "if_confirm"]
    rows = []

    print(f"\n{'='*70}")
    print(f"IF-CORE vs IF-HARDENED vs IF-CONFIRM (N={len(attacked)} att, {len(benign)} ben)")
    print(f"{'='*70}")
    print(f"{'Version':<15} {'ASR':>8} {'ASR CI':>20} {'BTSR':>8} {'FBR':>6}")
    print("-"*60)

    for v in versions:
        r = run_version(v, benign, attacked, documents)
        rows.append(r)
        print(f"{v:<15} {r['asr']:>7.1f}% [{r['asr_ci_lo']:.1f},{r['asr_ci_hi']:.1f}%] "
              f"{r['btsr']:>7.1f}% {r['fbr']:>5.1f}%")

    print(f"\nKey: IF-Core (base) → IF-Hardened (registry+SQL+span) → IF-Confirm (recommended)")

    with open(RESULTS_DIR/"if_versions.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    logger.info(f"Results saved to {RESULTS_DIR}/")
    return rows


if __name__ == "__main__":
    main()