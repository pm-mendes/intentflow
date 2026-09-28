"""
intentflow/experiments/run_agentdojo_sota_comparison.py

Direct comparison on the AgentDojo subset (data/agentdojo_official.py) of
IntentFlow (IF-Hardened) against five simplified SOTA reimplementations
(TaskShield, Progent, ClawGuard, CaMeL, PACT), same tasks, same protocol
(oracle intent for IntentFlow; the SOTA baselines are heuristic checks with
no LLM call), same model where applicable (gpt-4o-mini-2024-07-18, unused
here since everything runs oracle/heuristic -- no live LLM calls, no cost).

Requested explicitly (ICAART review, point B.4): a same-protocol comparison
against PACT/CaMeL/Progent/TaskShield on AgentDojo, not just Bench-v2.
"""

import csv
import json
from pathlib import Path

from intentflow.baselines.runner import oracle_intent_for
from intentflow.data.agentdojo_official import get_official_agentdojo_data
from intentflow.experiments.sota_baselines import (
    CaMeLBaseline,
    ClawGuardBaseline,
    PACTBaseline,
    ProgentBaseline,
    TaskShieldBaseline,
)
from intentflow.metrics.confidence_intervals import wilson_interval
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/agentdojo_sota_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_heuristic_baseline(name, baseline, benign, attacked, documents):
    na = nb = asr = btsr = fbr = 0
    for task in attacked:
        args = {**task.expected_arguments, **task.corrupted_arguments}
        allowed, _ = baseline.check(task.expected_tool, args, task.user_message, documents)
        na += 1
        if allowed:
            asr += 1
    for task in benign:
        allowed, _ = baseline.check(
            task.expected_tool, task.expected_arguments, task.user_message, documents
        )
        nb += 1
        if allowed:
            btsr += 1
        else:
            fbr += 1
    return {"name": name, "na": na, "nb": nb, "asr": asr, "btsr": btsr, "fbr": fbr}


def run_intentflow(benign, attacked, documents):
    engine = PolicyEngine.if_hardened()
    mediator = ToolMediator(policy_engine=engine, documents=documents)
    na = nb = asr = btsr = fbr = 0
    for task in attacked:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        args = {**task.expected_arguments, **task.corrupted_arguments}
        res = mediator.execute(
            task.expected_tool, args, task.user_message, tool_fn,
            baseline="B6", context_documents=documents, oracle_intent=oracle_intent_for(task),
        )
        na += 1
        if res["executed"]:
            asr += 1
    for task in benign:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        res = mediator.execute(
            task.expected_tool, task.expected_arguments, task.user_message, tool_fn,
            baseline="B6", context_documents=documents, oracle_intent=oracle_intent_for(task),
        )
        nb += 1
        if res["executed"]:
            btsr += 1
        else:
            fbr += 1
    return {"name": "IntentFlow (IF-Hardened)", "na": na, "nb": nb, "asr": asr, "btsr": btsr, "fbr": fbr}


def main():
    benign, attacked, documents = get_official_agentdojo_data()
    logger.info(f"AgentDojo subset SOTA comparison: {len(benign)} benign, {len(attacked)} attacked")

    rows = [run_intentflow(benign, attacked, documents)]
    for name, baseline in [
        ("TaskShield", TaskShieldBaseline()),
        ("Progent", ProgentBaseline()),
        ("ClawGuard", ClawGuardBaseline()),
        ("CaMeL", CaMeLBaseline()),
        ("PACT", PACTBaseline()),
    ]:
        rows.append(run_heuristic_baseline(name, baseline, benign, attacked, documents))

    print(f"\n{'='*78}")
    print(f"AGENTDOJO SUBSET SOTA COMPARISON (N={len(attacked)} attacked, {len(benign)} benign)")
    print(f"{'='*78}")
    print(f"{'Defense':<26} {'ASR':>8} {'ASR CI':>18} {'BTSR':>8} {'FBR':>8}")
    print("-" * 78)

    csv_rows = []
    for r in rows:
        asr_pct = r["asr"] / r["na"] * 100 if r["na"] else 0
        btsr_pct = r["btsr"] / r["nb"] * 100 if r["nb"] else 0
        fbr_pct = r["fbr"] / r["nb"] * 100 if r["nb"] else 0
        ci = wilson_interval(r["asr"], r["na"]) if r["na"] else None
        ci_str = f"[{ci.lower*100:.1f},{ci.upper*100:.1f}%]" if ci else "N/A"
        print(f"{r['name']:<26} {asr_pct:>7.1f}% {ci_str:>18} {btsr_pct:>7.1f}% {fbr_pct:>7.1f}%")
        csv_rows.append(
            {
                "defense": r["name"],
                "na": r["na"],
                "nb": r["nb"],
                "asr": round(asr_pct, 1),
                "asr_ci_lower": round(ci.lower * 100, 1) if ci else 0,
                "asr_ci_upper": round(ci.upper * 100, 1) if ci else 0,
                "btsr": round(btsr_pct, 1),
                "fbr": round(fbr_pct, 1),
            }
        )

    with open(RESULTS_DIR / "agentdojo_sota_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(RESULTS_DIR / "agentdojo_sota_comparison.json", "w") as f:
        json.dump(csv_rows, f, indent=2)

    logger.info(f"Results saved to {RESULTS_DIR}/")
    return csv_rows


if __name__ == "__main__":
    main()
