"""Bench-v2 (seeds 42, 123, 777) under B5, IF-Core, IF-Hardened and IF-Confirm, oracle intent, offline.

Regenerates the decision logs released as data/results/partfix_seed{SEED}_{CONFIG}/decisions_{B5|B6}.jsonl
(Table 3 of the ICAART 2027 paper: B5, IF-Core, IF-Hardened; the multi-seed paragraph; IF-Confirm).
Written for the camera-ready (14/11/2026): the script that produced the released logs on 28/09/2026 was not
kept. It goes through the same entry point as every other Bench-v2 run (BaselineRunner.run with
use_oracle=True, i.e. baselines.runner.oracle_intent_for). Its decisions are compared with the released
logs by scripts/check_paper_numbers.py.

Usage (from the repository root):  PYTHONPATH=src python scripts/run_bench_v2_configs.py OUTDIR
"""
import json
import sys
from pathlib import Path

from intentflow.baselines.runner import BaselineRunner
from intentflow.models.models import AttackedTask, BenignTask, RAGDocument
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine

SEEDS = {"42": Path("data/benchmark_v2"), "123": Path("data/benchmark_v2_seed123"),
         "777": Path("data/benchmark_v2_seed777")}
CONFIGS = (("b5", PolicyEngine.load_default_policies, "B5"), ("if_core", PolicyEngine.if_core, "B6"),
           ("if_hardened", PolicyEngine.if_hardened, "B6"), ("if_confirm", PolicyEngine.if_confirm, "B6"))


def load(root: Path):
    b = [BenignTask(**t) for t in json.loads((root / "tasks_benign.json").read_text())][:200]
    a = [AttackedTask(**t) for t in json.loads((root / "tasks_attacked.json").read_text())][:200]
    d = [RAGDocument(**x) for x in json.loads((root / "documents.json").read_text())]
    return b, a, d


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for seed, root in SEEDS.items():
        benign, attacked, docs = load(root)
        for name, make_engine, baseline in CONFIGS:
            runner = BaselineRunner(mediator=ToolMediator(policy_engine=make_engine(), documents=docs),
                                    output_dir=str(out / f"partfix_seed{seed}_{name}"))
            log = runner.run(baseline, benign, attacked, docs, use_oracle=True)
            rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            att = [r for r in rows if not r["is_benign"]]
            ben = [r for r in rows if r["is_benign"]]
            print(f"seed {seed} {name:12s} ASR {sum(r['executed'] for r in att)}/{len(att)} "
                  f"BTSR {sum(r['executed'] for r in ben)}/{len(ben)}")
