"""50 hard-benign Bench-v2 tasks under the paper's configurations (IF-Hardened, IF-Confirm, B5), oracle intent
built exactly as everywhere else in the paper (baselines.runner.oracle_intent_for). Offline.
Usage: PYTHONPATH=src python run_hard_benign_ifh.py OUTDIR"""
import json, sys
from pathlib import Path
from intentflow.baselines.runner import oracle_intent_for
from intentflow.data.hard_benign_tasks import get_hard_benign_data
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
tasks, docs = get_hard_benign_data()
rows = []
for name, eng, bl in (("IF-Hardened", PolicyEngine.if_hardened(), "B6"), ("IF-Confirm", PolicyEngine.if_confirm(), "B6"),
                      ("B5", PolicyEngine.load_default_policies(), "B5")):
    med = ToolMediator(policy_engine=eng, documents=docs)
    for t in tasks:
        r = med.execute(t.expected_tool, dict(t.expected_arguments), t.user_message, get_tool(t.expected_tool),
                        baseline=bl, context_documents=docs, oracle_intent=oracle_intent_for(t))
        dec = r.get("decision") or {}
        rows.append({"config": name, "task_id": t.task_id, "family": t.scenario_family, "executed": bool(r["executed"]),
                     "confirm": bool(dec.get("requires_confirm", False)), "error": r.get("error")})
with open(out / "hard_benign_paper_configs.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
for name in ("IF-Hardened", "IF-Confirm", "B5"):
    rs = [r for r in rows if r["config"] == name]
    print(name, "executed", sum(r["executed"] for r in rs), "/", len(rs), "confirm", sum(r["confirm"] for r in rs),
          "errors", sum(1 for r in rs if r["error"]), "blocked:", [r["task_id"] for r in rs if not r["executed"]])
