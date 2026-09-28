"""
intentflow/experiments/run_mixed_provenance.py
FM4 Mixed-Provenance experiment — B5 vs B6
"""

import csv
import json
from pathlib import Path

from intentflow.data.mixed_provenance import get_mixed_provenance_data
from intentflow.metrics.confidence_intervals import compute_all_intervals
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.policy.models import ArgumentPolicy
from intentflow.tools.registry import get_tool
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/mixed_provenance")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_strict_policies():
    engine = PolicyEngine.load_default_policies()
    for tool_name in ["send_email", "create_calendar_event"]:
        if tool_name not in engine.policies:
            continue
        engine.policies[tool_name].arguments = [
            p
            if p.argument_name not in ("recipient", "participants")
            else ArgumentPolicy(
                argument_name=p.argument_name,
                allowed_origins=["user_intent"],
                min_confidence=p.min_confidence,
                role=p.role,
            )
            for p in engine.policies[tool_name].arguments
        ]
    if "write_file" in engine.policies:
        engine.policies["write_file"].arguments = [
            p
            if p.argument_name != "path"
            else ArgumentPolicy(
                argument_name="path",
                allowed_origins=["user_intent"],
                min_confidence=0.90,
                role="locator",
            )
            for p in engine.policies["write_file"].arguments
        ]
    return engine


def _make_oracle(task):
    from intentflow.models.models import Intent

    return Intent(
        action=task.expected_tool,
        parameters={k: v for k, v in task.expected_arguments.items() if v is not None},
        confidence=1.0,
    )


def run_baseline(baseline, benign, attacked, docs, mediator):
    decisions = []
    for task in benign:
        tool_fn = get_tool(task.expected_tool)
        if not tool_fn:
            continue
        oracle = _make_oracle(task) if baseline == "B6" else None
        result = mediator.execute(
            task.expected_tool,
            task.expected_arguments,
            task.user_message,
            tool_fn,
            baseline=baseline,
            context_documents=docs,
            oracle_intent=oracle,
        )
        decisions.append(
            {
                "task_id": task.task_id,
                "baseline": baseline,
                "is_benign": True,
                "allowed": result["allowed"],
                "executed": result["executed"],
                "scenario_family": task.scenario_family,
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
            baseline=baseline,
            context_documents=docs,
        )
        decisions.append(
            {
                "task_id": task.task_id,
                "baseline": baseline,
                "is_benign": False,
                "allowed": result["allowed"],
                "executed": result["executed"],
                "scenario_family": task.scenario_family,
                "attack_type": task.attack_type,
            }
        )
    return decisions


def main():
    benign, attacked, docs = get_mixed_provenance_data()
    logger.info(f"Mixed-provenance: {len(benign)} benign, {len(attacked)} attacked")

    policy_engine = load_strict_policies()
    mediator = ToolMediator(policy_engine=policy_engine, documents=docs)

    results = {}
    for baseline in ["B5", "B6"]:
        logger.info(f"Running {baseline}...")
        decisions = run_baseline(baseline, benign, attacked, docs, mediator)
        results[baseline] = decisions
        with open(RESULTS_DIR / f"mixed_{baseline}.jsonl", "w") as f:
            for d in decisions:
                f.write(json.dumps(d) + "\n")

    print("\n" + "=" * 65)
    print("FM4 MIXED-PROVENANCE — B5 vs B6")
    print("All attacked values from trusted_internal documents")
    print("=" * 65)

    for baseline in ["B5", "B6"]:
        intervals = compute_all_intervals(results[baseline], baseline)
        print(f"\n{baseline}: ASR={intervals['ASR']} BTSR={intervals['BTSR']}")

    print(f"\n{'Family':<15} {'B5 ASR':>10} {'B6 ASR':>10}")
    print("-" * 38)
    rows = []
    for fam in ["email", "query", "file", "calendar"]:
        b5_a = [d for d in results["B5"] if not d["is_benign"] and d["scenario_family"] == fam]
        b6_a = [d for d in results["B6"] if not d["is_benign"] and d["scenario_family"] == fam]
        b5_asr = sum(1 for d in b5_a if d["executed"]) / len(b5_a) * 100 if b5_a else 0
        b6_asr = sum(1 for d in b6_a if d["executed"]) / len(b6_a) * 100 if b6_a else 0
        print(f"{fam:<15} {b5_asr:>9.1f}% {b6_asr:>9.1f}%")
        rows.append({"family": fam, "b5_asr": b5_asr, "b6_asr": b6_asr})

    with open(RESULTS_DIR / "mixed_by_family.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "b5_asr", "b6_asr"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
