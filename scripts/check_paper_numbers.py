"""Recompute every number of the ICAART 2027 camera-ready from the released result files and compare it with
the value printed in the paper. Writes the metrics manifest (paper location -> result file -> script).

Usage (from the repository root, no API key, about 5 s):
    PYTHONPATH=src python scripts/check_paper_numbers.py [--manifest artifact/results/paper_metrics_manifest.csv]

Exit code 0 if every recomputed value equals the paper's value, 1 otherwise. The decision latency is
machine-dependent: it matches only on the released log and is never counted as a failure.
Rates are printed with one decimal; 95% intervals are Wilson intervals (intentflow.metrics.confidence_intervals).
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from intentflow.metrics.confidence_intervals import wilson_interval

R = Path("data/results")
AUTHORITY = {"recipient", "participants", "path", "endpoint", "attachment"}
CONTENT = {"subject", "date", "description", "content"}


def jl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def rate(k, n):
    return f"{k}/{n} = {100 * k / n:.1f}%"


def ci(k, n):
    w = wilson_interval(k, n)
    return f"[{100 * w.lower:.1f}, {100 * w.upper:.1f}]"


def asr_btsr(rows, ex="executed"):
    att = [r for r in rows if not r["is_benign"]]
    ben = [r for r in rows if r["is_benign"]]
    return sum(bool(r[ex]) for r in att), len(att), sum(bool(r[ex]) for r in ben), len(ben)


def csv_rows(path):
    return list(csv.DictReader(open(path)))


ENTRIES = []  # (paper location, quantity, paper value, recomputed value, result file, script)


def add(loc, what, paper, got, src, script):
    ENTRIES.append((loc, what, paper, got, src, script))


def main():
    # --- Table 2: benchmark sizes --------------------------------------------------------------------------
    sizes = {"Bench-v2": ("data/benchmark_v2", 200), "Disambiguation-v2": ("data/disambiguation_v2", None),
             "AgentDojo-derived": ("data/agentdojo_official", None),
             "Adaptive white-box": ("data/adaptive_attacks_intentflow", None)}
    total = 0
    for name, (d, cap) in sizes.items():
        a = json.load(open(f"{d}/tasks_attacked.json"))
        b = json.load(open(f"{d}/tasks_benign.json"))
        a, b = (a[:cap], b[:cap]) if cap else (a, b)
        extra = len(json.load(open("data/hard_benign/tasks_benign.json"))) if name == "Bench-v2" else 0
        n = len(a) + len(b) + extra
        total += n
        add("Table 2", f"{name}: attacked / benign / total",
            {"Bench-v2": "200 / 250 / 450", "Disambiguation-v2": "50 / 50 / 100",
             "AgentDojo-derived": "20 / 20 / 40", "Adaptive white-box": "42 / 5 / 47"}[name],
            f"{len(a)} / {len(b) + extra} / {n}", d + "/", "src/intentflow/data/*")
    add("Table 2", "Total tasks", "637", str(total), "data/*/tasks_*.json", "src/intentflow/data/*")

    # --- Table 3: Bench-v2 seed 42 ---------------------------------------------------------------------------
    def row(label, path, paper_asr, paper_ci, paper_btsr, script, ex="executed"):
        ka, na, kb, nb = asr_btsr(jl(path), ex)
        add("Table 3", f"{label}: ASR", paper_asr, f"{100 * ka / na:.1f}%", str(path), script)
        add("Table 3", f"{label}: 95% CI", paper_ci, ci(ka, na), str(path), script)
        add("Table 3", f"{label}: BTSR", paper_btsr, f"{100 * kb / nb:.0f}%", str(path), script)

    rb = "intentflow.experiments.run_benchmark_v2"
    row("B0", R / "benchmark_v2/decisions_B0.jsonl", "100.0%", "[98.1, 100.0]", "100%", rb)
    row("B3", R / "benchmark_v2/decisions_B3.jsonl", "100.0%", "[98.1, 100.0]", "100%", rb)
    row("B4 action-level intent", R / "benchmark_v2/decisions_B4_full.jsonl", "100.0%", "[98.1, 100.0]", "100%",
        "scripts/run_b4_benchv2.py")
    row("B2 retrieval filter", R / "benchmark_v2/decisions_B2.jsonl", "74.5%", "[68.0, 80.0]", "100%", rb)
    bc = "scripts/run_bench_v2_configs.py"
    row("B5 provenance-only", R / "partfix_seed42_b5/decisions_B5.jsonl", "50.5%", "[43.6, 57.4]", "100%", bc)
    row("IF-Core", R / "partfix_seed42_if_core/decisions_B6.jsonl", "24.0%", "[18.6, 30.4]", "100%", bc)
    row("IntentFlow (IF-Hardened)", R / "partfix_seed42_if_hardened/decisions_B6.jsonl", "12.0%", "[8.2, 17.2]",
        "100%", bc)

    r1 = jl(R / "r1_comparisons/decisions.jsonl")
    for label, d, pa, pc, pb in (("Arg.-level provenance proxy", "PACT-style", "72.0%", "[65.4, 77.8]", "100%"),
                                 ("Arg.-level intent only", "ArgIntent-only", "37.0%", "[30.6, 43.9]", "100%"),
                                 ("CaMeL*", "CaMeL-style", "37.0%", "[30.6, 43.9]", "100%"),
                                 ("Progent*", "Progent-style", "18.0%", "[13.3, 23.9]", "93%")):
        rows = [r for r in r1 if r["suite"] == "Bench-v2" and r["defense"] == d]
        ka, na, kb, nb = asr_btsr(rows, "allowed")
        src = str(R / "r1_comparisons/decisions.jsonl")
        add("Table 3", f"{label}: ASR", pa, f"{100 * ka / na:.1f}%", src, "scripts/run_r1_comparisons.py")
        add("Table 3", f"{label}: 95% CI", pc, ci(ka, na), src, "scripts/run_r1_comparisons.py")
        add("Table 3", f"{label}: BTSR", pb, f"{100 * kb / nb:.0f}%", src, "scripts/run_r1_comparisons.py")

    sota = {r["baseline"]: r for r in csv_rows(R / "sota_baselines/sota_baselines.csv")}
    for label, key, pa, pc, pb in (("TaskShield*", "B7_TaskShield", "89.0%", "[83.9, 92.6]", "100%"),
                                   ("ClawGuard*", "B10_ClawGuard", "27.5%", "[21.8, 34.1]", "42%")):
        s = sota[key]
        k = round(float(s["asr"]) * int(s["na"]) / 100)
        add("Table 3", f"{label}: ASR", pa, f"{float(s['asr']):.1f}%", str(R / "sota_baselines/sota_baselines.csv"),
            "intentflow.experiments.sota_baselines (aggregate only, no decision log)")
        add("Table 3", f"{label}: 95% CI", pc, ci(k, int(s["na"])), str(R / "sota_baselines/sota_baselines.csv"),
            "intentflow.experiments.sota_baselines")
        add("Table 3", f"{label}: BTSR", pb, f"{float(s['btsr']):.0f}%", str(R / "sota_baselines/sota_baselines.csv"),
            "intentflow.experiments.sota_baselines")

    # --- Section 5 text: roles, hard-benign, intent vs provenance, seeds -------------------------------------
    tasks = {t["task_id"]: t for t in json.load(open("data/benchmark_v2/tasks_attacked.json"))[:200]}
    ifh = {r["task_id"]: r for r in jl(R / "partfix_seed42_if_hardened/decisions_B6.jsonl") if not r["is_benign"]}
    by_role = Counter()
    n_role = Counter()
    for tid, t in tasks.items():
        keys = set(t["corrupted_arguments"])
        role = "authority" if keys & AUTHORITY else "sql" if "query" in keys else "content"
        n_role[role] += 1
        by_role[role] += ifh[tid]["executed"]
    src = str(R / "partfix_seed42_if_hardened/decisions_B6.jsonl")
    add("Sec. 5, by role", "IF-Hardened, authority-bearing attacks executed", "0/134",
        f"{by_role['authority']}/{n_role['authority']}", src, bc)
    add("Sec. 5, by role", "95% CI (authority; printed [0, 2.8%])", "[0.0, 2.8]", ci(by_role["authority"], n_role["authority"]), src, bc)
    add("Sec. 5, by role", "IF-Hardened, SQL query attacks executed", "0/34", f"{by_role['sql']}/{n_role['sql']}",
        src, bc)
    add("Sec. 5, by role", "IF-Hardened, content/payload attacks executed", "24/32",
        f"{by_role['content']}/{n_role['content']}", src, bc)

    hb = jl(R / "hard_benign/hard_benign_paper_configs.jsonl")
    for cfg, paper in (("IF-Hardened", "46/50"), ("B5", "46/50")):
        rs = [r for r in hb if r["config"] == cfg]
        add("Sec. 5, main result", f"Hard-benign executed, {cfg}", paper, f"{sum(r['executed'] for r in rs)}/{len(rs)}",
            str(R / "hard_benign/hard_benign_paper_configs.jsonl"), "scripts/run_hard_benign_ifh.py")
    k = sum(r["executed"] for r in hb if r["config"] == "IF-Hardened")
    add("Sec. 5, main result", "Hard-benign 95% CI, IF-Hardened", "[81.2, 96.8]", ci(k, 50),
        str(R / "hard_benign/hard_benign_paper_configs.jsonl"), "scripts/run_hard_benign_ifh.py")

    for seed, b5, core, hard in (("123", "62.0%", "24.0%", "12.0%"), ("777", "64.0%", "26.0%", "12.0%")):
        for name, sub, ex, paper in (("B5", "b5", "B5", b5), ("IF-Core", "if_core", "B6", core),
                                     ("IF-Hardened", "if_hardened", "B6", hard)):
            p = R / f"partfix_seed{seed}_{sub}/decisions_{ex}.jsonl"
            ka, na, kb, nb = asr_btsr(jl(p))
            add("Sec. 5, stability", f"Seed {seed}, {name} ASR", paper, f"{100 * ka / na:.1f}%", str(p), bc)

    # --- Live extraction vs oracle (Sec. 4) ------------------------------------------------------------------
    live = {r["task_id"]: r for r in jl(R / "if_hardened_livellm_postfix/decisions_B6.jsonl")}
    orac = {r["task_id"]: r for r in jl(R / "partfix_seed42_if_hardened/decisions_B6.jsonl")}
    diff = sum((live[t]["allowed"], live[t]["executed"]) != (orac[t]["allowed"], orac[t]["executed"]) for t in orac)
    add("Sec. 4, oracle intent", "Live gpt-4o-mini extraction vs oracle, decisions changed", "0/400",
        f"{diff}/{len(orac)}", str(R / "if_hardened_livellm_postfix/decisions_B6.jsonl"),
        "live run, 21/09/2026 (hosted model; log only)")

    # --- Table 4: Disambiguation-v2 ---------------------------------------------------------------------------
    fam_order = ["Trusted wrong recipient", "Trusted wrong endpoint", "Trusted wrong file path",
                 "Trusted wrong participant", "Same-value-diff-intent"]
    paper_t4 = {"B5": [15, 10, 10, 10, 5], "Prg": [15, 10, 0, 10, 4], "Int": [0, 2, 0, 0, 0], "CaM": [0, 0, 0, 0, 0],
                "IFC": [0, 10, 0, 0, 1], "IFH": [0, 0, 0, 0, 0]}
    paper_btsr = {"B5": "100%", "Prg": "92%", "Int": "100%", "CaM": "68%", "IFC": "100%", "IFH": "100%"}
    logs = {"B5": jl(R / "disambiguation_v2/disambiguation_v2_b5.jsonl"),
            "IFC": jl(R / "disambiguation_v2/disambiguation_v2_if_core.jsonl"),
            "IFH": jl(R / "disambiguation_v2/disambiguation_v2_if_hardened.jsonl")}
    r1d = [r for r in r1 if r["suite"] == "Disambiguation-v2"]
    for col, d in (("Prg", "Progent-style"), ("Int", "ArgIntent-only"), ("CaM", "CaMeL-style"), ("PACT", "PACT-style")):
        logs[col] = [{**r, "executed": r["allowed"], "family_label": r["family"]} for r in r1d if r["defense"] == d]
    for col in paper_t4:
        rows = logs[col]
        att = [r for r in rows if not r["is_benign"]]
        got = [sum(r["executed"] for r in att if r["family_label"] == f) for f in fam_order]
        src = "scripts/run_r1_comparisons.py" if col in ("Prg", "Int", "CaM") else \
            "intentflow.experiments.run_disambiguation_v2"
        f = (str(R / "r1_comparisons/decisions.jsonl") if col in ("Prg", "Int", "CaM")
             else str(R / f"disambiguation_v2/disambiguation_v2_{ {'B5': 'b5', 'IFC': 'if_core', 'IFH': 'if_hardened'}[col]}.jsonl"))
        add("Table 4", f"{col}: executed per family (recipient, endpoint, path, participant, same value)",
            " ".join(map(str, paper_t4[col])), " ".join(map(str, got)), f, src)
        ka, na, kb, nb = asr_btsr(rows)
        add("Table 4", f"{col}: BTSR", paper_btsr[col], f"{100 * kb / nb:.0f}%", f, src)
    ka, na, _, _ = asr_btsr(logs["PACT"])
    add("Table 4 note", "Provenance proxy, attacks executed", "50/50", f"{ka}/{na}",
        str(R / "r1_comparisons/decisions.jsonl"), "scripts/run_r1_comparisons.py")
    conf = jl(R / "disambiguation_v2/disambiguation_v2_if_confirm.jsonl")
    ka, na, kb, nb = asr_btsr(conf)
    add("Table 4 note", "IF-Confirm ASR / BTSR", "0/50, 50/50", f"{ka}/{na}, {kb}/{nb}",
        str(R / "disambiguation_v2/disambiguation_v2_if_confirm.jsonl"), "intentflow.experiments.run_disambiguation_v2")
    n_conf = sum(1 for r in conf if not r["is_benign"] and not r["executed"] and not r["blocked_arguments"])
    add("Table 4 note", "IF-Confirm attack decisions that are Confirm (not executed, no argument blocked)", "40/50",
        f"{n_conf}/50", str(R / "disambiguation_v2/disambiguation_v2_if_confirm.jsonl"),
        "intentflow.experiments.run_disambiguation_v2 (Confirm inferred: the log has no decision-type field)")

    # --- Endpoint ablation (Sec. 5.1) -------------------------------------------------------------------------
    ea = {r["mode"]: r for r in csv_rows(R / "endpoint_ablation_n30/endpoint_ablation_n30.csv")}
    src = str(R / "endpoint_ablation_n30/endpoint_ablation_n30.csv")
    sc = "intentflow.experiments.run_endpoint_ablation_n30 (aggregate only)"
    m = {k.split(" (")[0]: v for k, v in ea.items()}
    add("Sec. 5.1, endpoint ablation", "IF-Hardened (fail-closed): ASR, BTSR", "0.0%, 46.7%",
        f"{float(m['Block unresolved']['asr']):.1f}%, {float(m['Block unresolved']['btsr']):.1f}%", src, sc)
    add("Sec. 5.1, endpoint ablation", "Fail-open: ASR, BTSR", "16.7%, 46.7%",
        f"{float(m['Allow unresolved']['asr']):.1f}%, {float(m['Allow unresolved']['btsr']):.1f}%", src, sc)
    add("Sec. 5.1, endpoint ablation", "IF-Confirm: ASR, BTSR", "0.0%, 93.3%",
        f"{float(m['Registry + Confirm']['asr']):.1f}%, {float(m['Registry + Confirm']['btsr']):.1f}%", src, sc)

    # --- Table 5: AgentDojo-derived replay -------------------------------------------------------------------
    ad = {r["defense"]: r for r in csv_rows(R / "agentdojo_sota_comparison/agentdojo_sota_comparison.csv")}
    paper_t5 = {"TaskShield": ("95.0%", "[76.4, 99.1]", "95%"), "ClawGuard": ("90.0%", "[69.9, 97.2]", "75%"),
                "PACT": ("75.0%", "[53.1, 88.8]", "100%"), "Progent": ("5.0%", "[0.9, 23.6]", "100%"),
                "CaMeL": ("0.0%", "[0.0, 16.1]", "75%"), "IntentFlow (IF-Hardened)": ("0.0%", "[0.0, 16.1]", "75%")}
    for d, (pa, pc, pb) in paper_t5.items():
        r = ad[d]
        add("Table 5", f"{d}: ASR, 95% CI, BTSR", f"{pa} {pc} {pb}",
            f"{float(r['asr']):.1f}% [{float(r['asr_ci_lower']):.1f}, {float(r['asr_ci_upper']):.1f}] "
            f"{float(r['btsr']):.0f}%", str(R / "agentdojo_sota_comparison/agentdojo_sota_comparison.csv"),
            "intentflow.experiments.run_agentdojo_sota_comparison")

    # --- Adaptive attacks (Sec. 5.2) -------------------------------------------------------------------------
    adp = jl(R / "adaptive_eval/adaptive_attacks.jsonl")
    ka, na, kb, nb = asr_btsr(adp)
    fam = Counter(r["injection_type"] for r in adp if not r["is_benign"] and r["executed"])
    src = str(R / "adaptive_eval/adaptive_attacks.jsonl")
    add("Sec. 5.2, adaptive", "Attacks executed / benign executed", "4/42, 4/5", f"{ka}/{na}, {kb}/{nb}", src,
        "intentflow.experiments.run_adaptive_eval")
    add("Sec. 5.2, adaptive", "Families with a success", "Role-confusion: 4", ", ".join(f"{k}: {v}" for k, v in fam.items()),
        src, "intentflow.experiments.run_adaptive_eval")

    # --- Mixed provenance (Sec. 5.2) -------------------------------------------------------------------------
    mp = jl(R / "mixed_provenance/mixed_provenance_configs.jsonl")
    src = str(R / "mixed_provenance/mixed_provenance_configs.jsonl")
    for cfg, paper in (("B5", "40/40, 20/20"), ("IF-Core", "11/40, 20/20"), ("IF-Hardened", "0/40, 20/20")):
        ka, na, kb, nb = asr_btsr([r for r in mp if r["config"] == cfg])
        add("Sec. 5.2, mixed provenance", f"{cfg}: attacks, benign executed", paper, f"{ka}/{na}, {kb}/{nb}", src,
            "scripts/run_mixed_provenance_configs.py")
    ka, na, _, _ = asr_btsr([r for r in mp if r["config"] == "IF-Core"])
    add("Sec. 5.2, mixed provenance", "IF-Core 95% CI", "[16.1, 42.8]", ci(ka, na), src,
        "scripts/run_mixed_provenance_configs.py")

    # --- Section 6: latency and live campaign ----------------------------------------------------------------
    lat = [r["latency_ms"] for r in jl(R / "partfix_seed42_if_hardened/decisions_B6.jsonl")]
    add("Sec. 6, latency", "IF-Hardened decision latency p50 / p95 (ms)", "63.6 / 242.2",
        f"{np.percentile(lat, 50):.1f} / {np.percentile(lat, 95):.1f}",
        str(R / "partfix_seed42_if_hardened/decisions_B6.jsonl"), "machine-dependent; numpy linear percentiles")
    e2e = jl(R / "e2e_phase3/decisions.jsonl")
    att = [r for r in e2e if r["is_attacked"]]
    ben = [r for r in e2e if not r["is_attacked"]]
    corr = [r for r in att if r.get("argument_corrupted")]
    mh = [r for r in att if r["technique"] == "multi_hop"]
    blocked = sum(r["mediation_blocked"] for r in corr)
    src = str(R / "e2e_phase3/decisions.jsonl")
    sc = "intentflow.experiments.e2e_phase3 (hosted model; log only)"
    add("Sec. 6, live campaign", "Corrupted calls proposed", "17/110 = 15.5%", rate(len(corr), len(att)), src, sc)
    add("Sec. 6, live campaign", "Multi-hop corrupted calls", "16/22 = 72.7%",
        rate(sum(r.get("argument_corrupted", False) for r in mh), len(mh)), src, sc)
    add("Sec. 6, live campaign", "Corrupted calls blocked, 95% CI", "17/17 [81.6, 100.0]",
        f"{blocked}/{len(corr)} {ci(blocked, len(corr))}", src, sc)
    uer = len(corr) - blocked
    add("Sec. 6, live campaign", "Unauthorized executions, 95% CI", "0/110 [0.0, 3.4]",
        f"{uer}/{len(att)} {ci(uer, len(att))}", src, sc)
    okb = sum(1 for r in ben if not r["planner_failed"] and r["tool_correct"] and not r["mediation_blocked"])
    add("Sec. 6, live campaign", "Benign tasks executed", "48/50 = 96.0%", rate(okb, len(ben)), src, sc)
    add("Sec. 6, live campaign", "Planner tokens", "55873", str(sum(r.get("total_tokens", 0) for r in e2e)), src, sc)


if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--manifest") + 1]) if "--manifest" in sys.argv else \
        Path("artifact/results/paper_metrics_manifest.csv")
    main()
    bad = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paper_location", "quantity", "paper_value", "recomputed_value", "match", "result_file", "script"])
        for loc, what, paper, got, src, script in ENTRIES:
            timing = script.startswith("machine-dependent")
            ok = paper == got
            bad += not ok and not timing
            w.writerow([loc, what, paper, got, "yes" if ok else ("timing" if timing else "NO"), src, script])
            print(f"{'ok ' if ok else ('~~ ' if timing else 'XX ')} {loc:28s} {what:70s} paper={paper:22s} got={got}")
    print(f"\n{len(ENTRIES) - bad}/{len(ENTRIES)} values match. Manifest: {out}")
    sys.exit(1 if bad else 0)
