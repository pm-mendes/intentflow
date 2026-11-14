"""B4 (action-level intent) on full Bench-v2 seed 42, oracle intent, offline. Usage: PYTHONPATH=src python run_b4_benchv2.py OUTDIR"""
import json, sys
from pathlib import Path
from intentflow.baselines.runner import oracle_intent_for
from intentflow.models.models import AttackedTask, BenignTask, RAGDocument
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
root = Path("data/benchmark_v2"); out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
b = [BenignTask(**t) for t in json.load(open(root / "tasks_benign.json"))]
a = [AttackedTask(**t) for t in json.load(open(root / "tasks_attacked.json"))]
d = [RAGDocument(**x) for x in json.load(open(root / "documents.json"))]
med = ToolMediator(policy_engine=PolicyEngine.load_default_policies(), documents=d)
rows = []
for ben, tasks in ((True, b), (False, a)):
    for t in tasks:
        args = dict(t.expected_arguments) if ben else {**t.expected_arguments, **t.corrupted_arguments}
        r = med.execute(t.expected_tool, args, t.user_message, get_tool(t.expected_tool), baseline="B4",
                        context_documents=d, oracle_intent=oracle_intent_for(t))
        rows.append({"task_id": t.task_id, "is_benign": ben, "executed": bool(r["executed"]), "error": r.get("error")})
with open(out / "decisions_B4_full.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
att = [r for r in rows if not r["is_benign"]]; bn = [r for r in rows if r["is_benign"]]
print("B4 ASR", sum(r["executed"] for r in att), "/", len(att), "BTSR", sum(r["executed"] for r in bn), "/", len(bn), "errors", sum(1 for r in rows if r["error"]))
