"""
intentflow/experiments/run_adaptive_eval.py

Évaluation adaptative contre IntentFlow (Point 10) +
Official AgentDojo subset (Point 9).
"""

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

from intentflow.data.agentdojo_official import get_official_agentdojo_data
from intentflow.data.adaptive_attacks_intentflow import get_adaptive_intentflow_data
from intentflow.models.models import RAGDocument
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
from intentflow.metrics.confidence_intervals import wilson_interval
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/adaptive_eval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_strict_policies() -> PolicyEngine:
    """
    Phase 0.2: was a hand-rolled policy-tightening patch duplicated across
    this script, `e2e_agent_eval_v2.py` and `if_core_vs_hardened.py`. Now
    delegates to `PolicyEngine.if_hardened()`, the single named IF-Hardened
    configuration (registry + SQL AST + span provenance + fail-closed).
    """
    return PolicyEngine.if_hardened()


def run_benchmark(name, benign, attacked, documents, mediator, strict=True):
    decisions = []
    for task in benign:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        from intentflow.models.models import Intent
        oracle = Intent(action=task.expected_tool,
            parameters={k: v for k, v in task.expected_arguments.items() if v is not None},
            confidence=1.0)
        result = mediator.execute(task.expected_tool, task.expected_arguments,
            task.user_message, tool_fn, baseline="B6",
            context_documents=documents, oracle_intent=oracle)
        decisions.append({"task_id": task.task_id, "is_benign": True,
            "allowed": result["allowed"], "executed": result["executed"],
            "scenario_family": task.scenario_family, "injection_type": None})

    for task in attacked:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        args = {**task.expected_arguments, **task.corrupted_arguments}
        from intentflow.models.models import Intent
        oracle = Intent(action=task.expected_tool,
            parameters={k: v for k, v in task.expected_arguments.items() if v is not None},
            confidence=1.0)
        result = mediator.execute(task.expected_tool, args, task.user_message,
            tool_fn, baseline="B6", context_documents=documents, oracle_intent=oracle)
        decisions.append({"task_id": task.task_id, "is_benign": False,
            "allowed": result["allowed"], "executed": result["executed"],
            "scenario_family": task.scenario_family,
            "injection_type": getattr(task, "injection_type", task.attack_type.split(":")[0].strip())})
    return decisions


def print_results(name, decisions):
    att = [d for d in decisions if not d["is_benign"]]
    ben = [d for d in decisions if d["is_benign"]]
    na, nb = len(att), len(ben)
    asr_n = sum(1 for d in att if d["executed"])
    btsr_n = sum(1 for d in ben if d["executed"])
    fbr_n = sum(1 for d in ben if not d["allowed"])

    asr_ci = wilson_interval(asr_n, na) if na > 0 else None
    btsr_ci = wilson_interval(btsr_n, nb) if nb > 0 else None

    print(f"\n=== {name} ===")
    print(f"N: {na} attacked, {nb} benign")
    print(f"ASR  : {asr_n}/{na}={asr_n/na*100:.1f}% {f'[{asr_ci.lower*100:.1f},{asr_ci.upper*100:.1f}%]' if asr_ci else ''}")
    print(f"BTSR : {btsr_n}/{nb}={btsr_n/nb*100:.1f}% {f'[{btsr_ci.lower*100:.1f},{btsr_ci.upper*100:.1f}%]' if btsr_ci else ''}")
    print(f"FBR  : {fbr_n}/{nb}={fbr_n/nb*100:.1f}%")

    # Par famille d'attaque
    if any(d["injection_type"] for d in decisions):
        print(f"\n  By attack family:")
        by_family = defaultdict(list)
        for d in att:
            by_family[d.get("injection_type", "unknown")].append(d)
        for fam, tasks in sorted(by_family.items()):
            executed = sum(1 for t in tasks if t["executed"])
            ci = wilson_interval(executed, len(tasks))
            print(f"    {fam:<25} ASR={executed}/{len(tasks)}={executed/len(tasks)*100:.0f}% "
                  f"[{ci.lower*100:.1f},{ci.upper*100:.1f}%]")

    return {"name": name, "na": na, "nb": nb,
            "asr": round(asr_n/na*100, 1) if na > 0 else 0,
            "btsr": round(btsr_n/nb*100, 1) if nb > 0 else 0,
            "fbr": round(fbr_n/nb*100, 1) if nb > 0 else 0}


def main():
    policy_engine = load_strict_policies()
    all_results = []

    # ============================================================
    # Point 9 : Official AgentDojo subset
    # ============================================================
    print("\n" + "="*65)
    print("POINT 9: Official AgentDojo Subset")
    print("="*65)
    ag_ben, ag_att, ag_docs = get_official_agentdojo_data()
    mediator_ag = ToolMediator(policy_engine=policy_engine, documents=ag_docs)
    decs_ag = run_benchmark("AgentDojo Official", ag_ben, ag_att, ag_docs, mediator_ag)
    all_results.append(print_results("Official AgentDojo Subset (N=20+20)", decs_ag))

    with open(RESULTS_DIR / "agentdojo_official.jsonl", "w") as f:
        for d in decs_ag:
            f.write(json.dumps(d) + "\n")

    # ============================================================
    # Point 10 : Adaptive attacks against IntentFlow
    # ============================================================
    print("\n" + "="*65)
    print("POINT 10: Adaptive Attacks Against IntentFlow")
    print("="*65)
    adp_ben, adp_att, adp_docs = get_adaptive_intentflow_data()
    mediator_adp = ToolMediator(policy_engine=policy_engine, documents=adp_docs)
    decs_adp = run_benchmark("Adaptive", adp_ben, adp_att, adp_docs, mediator_adp)
    all_results.append(print_results("Adaptive Attacks (6 families, N=42)", decs_adp))

    with open(RESULTS_DIR / "adaptive_attacks.jsonl", "w") as f:
        for d in decs_adp:
            f.write(json.dumps(d) + "\n")

    # Export CSV
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "na", "nb", "asr", "btsr", "fbr"])
        writer.writeheader()
        writer.writerows(all_results)

    logger.info(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()