"""Bench-v2 (seed 42, 200 attacked / 200 benign) under B0 (no defense), B2 (retrieval filter) and B3 (tool
allowlist), offline. Regenerates data/results/benchmark_v2/decisions_B{0,2,3}.jsonl (Table 3 of the ICAART 2027
paper). These baselines never use the intent contract.

Usage (from the repository root):  PYTHONPATH=src python scripts/run_bench_v2_baselines.py OUTDIR
"""
import json
import sys
from pathlib import Path

from intentflow.baselines.runner import BaselineRunner
from intentflow.models.models import AttackedTask, BenignTask, RAGDocument
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine

ROOT = Path("data/benchmark_v2")

if __name__ == "__main__":
    out = Path(sys.argv[1])
    benign = [BenignTask(**t) for t in json.loads((ROOT / "tasks_benign.json").read_text())]
    attacked = [AttackedTask(**t) for t in json.loads((ROOT / "tasks_attacked.json").read_text())]
    docs = [RAGDocument(**d) for d in json.loads((ROOT / "documents.json").read_text())]
    runner = BaselineRunner(mediator=ToolMediator(policy_engine=PolicyEngine.load_default_policies(), documents=docs),
                            output_dir=str(out))
    for baseline in ("B0", "B2", "B3"):
        log = runner.run(baseline, benign, attacked, docs, use_oracle=True)
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        att = [r for r in rows if not r["is_benign"]]
        ben = [r for r in rows if r["is_benign"]]
        print(f"{baseline} ASR {sum(r['executed'] for r in att)}/{len(att)} BTSR {sum(r['executed'] for r in ben)}/{len(ben)}")
