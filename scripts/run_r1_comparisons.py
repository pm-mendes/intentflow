"""
Camera-ready (ICAART 2027 #368) — comparisons requested by Reviewer 1, run OFFLINE.

Runs, on Disambiguation-v2 (trusted-but-unauthorized set) and on Bench-v2 (seed 42):
  * CaMeL-style, Progent-style, PACT-style  -> existing classes of
    intentflow.experiments.sota_baselines (unchanged);
  * ArgIntent-only  -> NEW ablation: argument-level intent check WITHOUT provenance.
    Every authority-bearing argument value must belong to the value(s) the oracle
    intent contract authorizes for that same field; a field absent from the
    contract is blocked (fail-closed). No provenance, registry, SQL or span check.
  * IF-Hardened (B6) -> sanity check against the published numbers.

No LLM is called: every intent is the oracle contract built by
intentflow.baselines.runner.oracle_intent_for (same function as the paper).
Usage (from the artifact root):  PYTHONPATH=src python run_r1_comparisons.py OUTDIR
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

from intentflow.baselines.runner import oracle_intent_for
from intentflow.data.disambiguation_v2 import get_disambiguation_v2_data
from intentflow.experiments.sota_baselines import CaMeLBaseline, PACTBaseline, ProgentBaseline
from intentflow.metrics.confidence_intervals import wilson_interval
from intentflow.models.models import AttackedTask, BenignTask, RAGDocument
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
from intentflow.experiments.run_disambiguation_v2 import family_label

AUTHORITY_ARGS = {"recipient", "to", "cc", "bcc", "participants", "path", "endpoint", "url"}


def _vals(v):
    if v is None:
        return []
    return [str(x).strip().lower() for x in (v if isinstance(v, list) else [v]) if x is not None]


class ArgIntentOnly:
    """Argument-level intent check, no provenance (Reviewer 1, major issue 3)."""

    def check(self, task, args):
        intent = oracle_intent_for(task).parameters
        for name, val in args.items():
            if name not in AUTHORITY_ARGS or val is None or val == [] or val == "":
                continue
            allowed = set(_vals(intent.get(name)))
            if not allowed:
                return False, f"ArgIntent: {name} not in contract (fail-closed)"
            if not set(_vals(val)) <= allowed:
                return False, f"ArgIntent: {name} not authorized by intent"
        return True, "ArgIntent: allowed"


def load_bench_v2(root: Path):
    b = [BenignTask(**t) for t in json.load(open(root / "tasks_benign.json"))]
    a = [AttackedTask(**t) for t in json.load(open(root / "tasks_attacked.json"))]
    d = [RAGDocument(**x) for x in json.load(open(root / "documents.json"))]
    return b, a, d


def run_suite(name, benign, attacked, docs, fam):
    heur = {"CaMeL-style": CaMeLBaseline(), "Progent-style": ProgentBaseline(), "PACT-style": PACTBaseline()}
    argi = ArgIntentOnly()
    med = ToolMediator(policy_engine=PolicyEngine.if_hardened(), documents=docs)
    rows = []

    def record(defense, task, is_benign, allowed):
        rows.append({"suite": name, "defense": defense, "task_id": task.task_id, "is_benign": is_benign,
                     "family": "benign" if is_benign else fam(task),
                     "attack_type": None if is_benign else task.attack_type, "allowed": bool(allowed)})

    for is_benign, tasks in ((True, benign), (False, attacked)):
        for t in tasks:
            args = dict(t.expected_arguments) if is_benign else {**t.expected_arguments, **t.corrupted_arguments}
            for dn, b in heur.items():
                record(dn, t, is_benign, b.check(t.expected_tool, args, t.user_message, docs)[0])
            record("ArgIntent-only", t, is_benign, argi.check(t, args)[0])
            fn = get_tool(t.expected_tool)
            r = med.execute(t.expected_tool, args, t.user_message, fn, baseline="B6",
                            context_documents=docs, oracle_intent=oracle_intent_for(t))
            record("IF-Hardened", t, is_benign, r["executed"])
    return rows


def summarize(rows):
    out = []
    key = defaultdict(list)
    for r in rows:
        key[(r["suite"], r["defense"])].append(r)
    for (suite, d), rs in sorted(key.items()):
        att = [r for r in rs if not r["is_benign"]]
        ben = [r for r in rs if r["is_benign"]]
        s = sum(r["allowed"] for r in att)
        wi = wilson_interval(s, len(att)); lo, hi = wi.lower, wi.upper
        row = {"suite": suite, "defense": d, "asr_k": s, "asr_n": len(att),
               "asr_pct": round(100 * s / len(att), 1), "ci_lo": round(100 * lo, 1), "ci_hi": round(100 * hi, 1),
               "btsr_k": sum(r["allowed"] for r in ben), "btsr_n": len(ben)}
        fams = defaultdict(lambda: [0, 0])
        for r in att:
            fams[r["family"]][0] += r["allowed"]; fams[r["family"]][1] += 1
        row["by_family"] = {f: f"{k}/{n}" for f, (k, n) in sorted(fams.items())}
        out.append(row)
    return out


if __name__ == "__main__":
    outdir = Path(sys.argv[1]); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    b, a, d = get_disambiguation_v2_data()
    rows += run_suite("Disambiguation-v2", b, a, d, family_label)
    b2, a2, d2 = load_bench_v2(Path("data/benchmark_v2"))
    rows += run_suite("Bench-v2", b2, a2, d2, lambda t: t.attack_type)
    with open(outdir / "decisions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = summarize(rows)
    json.dump(summ, open(outdir / "summary.json", "w"), indent=1)
    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in summ[0] if k != "by_family"])
        w.writeheader()
        for s in summ:
            w.writerow({k: v for k, v in s.items() if k != "by_family"})
    for s in summ:
        print(s["suite"], s["defense"], f'ASR {s["asr_k"]}/{s["asr_n"]}={s["asr_pct"]}% [{s["ci_lo"]},{s["ci_hi"]}]',
              f'BTSR {s["btsr_k"]}/{s["btsr_n"]}')
        if s["suite"] == "Disambiguation-v2":
            print("   ", s["by_family"])
