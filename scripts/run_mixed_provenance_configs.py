"""Mixed-provenance set (data.mixed_provenance_v2, 40 attacked / 20 benign) under the paper's named configurations,
oracle intent via baselines.runner.oracle_intent_for. Offline. Usage: PYTHONPATH=src python run_mixed_provenance_configs.py OUTDIR"""
import json, sys
from pathlib import Path
from collections import Counter
from intentflow.baselines.runner import oracle_intent_for
from intentflow.data.mixed_provenance_v2 import get_mixed_v2_data
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
benign, attacked, docs = get_mixed_v2_data()
rows = []
for name, eng, bl in (("B5", PolicyEngine.load_default_policies(), "B5"), ("IF-Core", PolicyEngine.if_core(), "B6"),
                      ("IF-Hardened", PolicyEngine.if_hardened(), "B6"), ("IF-Confirm", PolicyEngine.if_confirm(), "B6")):
    med = ToolMediator(policy_engine=eng, documents=docs)
    for ben, tasks in ((True, benign), (False, attacked)):
        for t in tasks:
            args = dict(t.expected_arguments) if ben else {**t.expected_arguments, **t.corrupted_arguments}
            r = med.execute(t.expected_tool, args, t.user_message, get_tool(t.expected_tool), baseline=bl,
                            context_documents=docs, oracle_intent=oracle_intent_for(t))
            rows.append({"config": name, "task_id": t.task_id, "is_benign": ben,
                         "attack_type": None if ben else t.attack_type,
                         "corrupted_keys": [] if ben else sorted(t.corrupted_arguments), "executed": bool(r["executed"]),
                         "error": r.get("error")})
with open(out / "mixed_provenance_configs.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
for name in ("B5", "IF-Core", "IF-Hardened", "IF-Confirm"):
    a = [r for r in rows if r["config"] == name and not r["is_benign"]]; b = [r for r in rows if r["config"] == name and r["is_benign"]]
    print(name, "ASR", sum(r["executed"] for r in a), "/", len(a), "BTSR", sum(r["executed"] for r in b), "/", len(b),
          "errors", sum(1 for r in rows if r["config"] == name and r["error"]),
          "succ keys", Counter(k for r in a if r["executed"] for k in r["corrupted_keys"]))
