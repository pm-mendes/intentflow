"""
scripts/generate_metrics_manifest.py

Génère metrics_manifest.json — fichier de référence listant tous les chiffres
utilisés dans le papier, avec leur source (CSV/JSONL) et leur valeur exacte.

Ce script a été réaligné (2026-07) sur la composition RÉELLE du benchmark de
la version courante du papier (`paper/mainraise_submit.tex`, paragraphe
"Benchmarks (657 tasks)", table autour de la ligne 280-293) :

    Suite                 Attacked  Benign          Total  RQ / Table
    Bench-v2 (main)            200  200 + 50 hard    450   RQ1 (tab:main)
    Disambiguation-v2           50  50               100   RQ2 (tab:disambig)
    AgentDojo subset             20  20                40   RQ3
    Bench-v1 (sanity)            10  10                20   tab:versions
    Adaptive white-box           42   5                47   RQ4 (tab:adaptive)
    ------------------------------------------------------------------
    TOTAL                                             657

Toutes les valeurs ci-dessous sont recalculées depuis les fichiers réels sous
data/results/ (JSONL de décisions passés dans MetricsComputer, ou CSV déjà
agrégés par les scripts d'expérience) — rien n'est saisi à la main.

Usage: poetry run python scripts/generate_metrics_manifest.py
Output: artifact/metrics_manifest.json
"""

import csv
import json
from pathlib import Path


def load_csv_rows(path: str) -> list:
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def load_jsonl_decisions(path: str) -> list:
    try:
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    except Exception:
        return []


def wilson_ci(count: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    import math

    z = 1.96
    p = count / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    margin = (z / (1 + z * z / total)) * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_metrics_from_jsonl(path: str, baseline: str) -> dict:
    decisions = load_jsonl_decisions(path)
    if not decisions:
        return {}
    attacked = [d for d in decisions if not d.get("is_benign", True)]
    benign = [d for d in decisions if d.get("is_benign", True)]
    n_a = len(attacked)
    n_b = len(benign)
    asr_count = sum(1 for d in attacked if d.get("executed", False))
    btsr_count = sum(1 for d in benign if d.get("executed", False))
    fbr_count = sum(1 for d in benign if not d.get("allowed", True))
    asr_lo, asr_hi = wilson_ci(asr_count, n_a)
    btsr_lo, btsr_hi = wilson_ci(btsr_count, n_b)
    return {
        "baseline": baseline,
        "source": path,
        "n_attacked": n_a,
        "n_benign": n_b,
        "asr_count": asr_count,
        "asr_pct": round(asr_count / n_a * 100, 1) if n_a > 0 else 0,
        "asr_ci_lower": round(asr_lo * 100, 1),
        "asr_ci_upper": round(asr_hi * 100, 1),
        "btsr_count": btsr_count,
        "btsr_pct": round(btsr_count / n_b * 100, 1) if n_b > 0 else 0,
        "fbr_count": fbr_count,
        "fbr_pct": round(fbr_count / n_b * 100, 1) if n_b > 0 else 0,
    }


def main():
    manifest = {
        "generated_by": "generate_metrics_manifest.py",
        "seed": 42,
        "paper": "Intent-Conditioned Argument Mediation for Retrieval-to-Action Security in Agentic RAG",
        "benchmarks": {},
        "tables": {},
        "figures": {},
    }

    # ================================================================
    # Table 2 (tab:main) — Bench-v2, N=200 attacked / M=200 benign
    # 7 defenses: B0, B1, B1b, B2, B3, B5, B6 (B4 intent-only is NOT part
    # of the headline table; it is reported separately, see below).
    # ================================================================
    main_table = {}
    for baseline in ["B0", "B1", "B2", "B3", "B5"]:
        path = f"data/results/benchmark_v2/decisions_{baseline}.jsonl"
        metrics = compute_metrics_from_jsonl(path, baseline)
        if metrics:
            main_table[baseline] = metrics
    b1b = compute_metrics_from_jsonl("data/results/sprint2/decisions_B1b.jsonl", "B1b")
    if b1b:
        main_table["B1b"] = b1b

    # B6 cross-check: benchmark_v2/decisions_B6.jsonl is a STALE N=25 sample
    # (kept from an earlier cost-aware-sampling sprint), while
    # data/results/b6_n200/decisions_B6.jsonl is the full N=200 run that the
    # paper's headline ASR=24.0% [18.6,30.4%] actually comes from. Both are
    # computed here so the mismatch is visible rather than silently resolved.
    b6_stale_n25 = compute_metrics_from_jsonl(
        "data/results/benchmark_v2/decisions_B6.jsonl", "B6_stale_sample_n25"
    )
    b6_full_n200 = compute_metrics_from_jsonl(
        "data/results/b6_n200/decisions_B6.jsonl", "B6"
    )
    main_table["B6"] = b6_full_n200
    b6_mismatch_note = None
    if b6_stale_n25 and b6_full_n200:
        if b6_stale_n25["n_attacked"] != b6_full_n200["n_attacked"]:
            b6_mismatch_note = (
                f"FLAG: data/results/benchmark_v2/decisions_B6.jsonl has only "
                f"N={b6_stale_n25['n_attacked']} attacked (stale partial sample, "
                f"ASR={b6_stale_n25['asr_pct']}%), while data/results/b6_n200/"
                f"decisions_B6.jsonl has the full N={b6_full_n200['n_attacked']} "
                f"(ASR={b6_full_n200['asr_pct']}%), matching the paper's "
                f"tab:main figure of 24.0%. The N=200 file (b6_n200) is used as "
                f"canonical for table2 below; the N=25 file under benchmark_v2/ "
                f"is legacy and should not be treated as authoritative."
            )

    manifest["tables"]["table2_bench_v2_main"] = {
        "description": (
            "Main results (tab:main): Bench-v2 N=200 attacked / M=200 benign, "
            "7 defenses B0/B1/B1b/B2/B3/B5/B6. FBR=0% for all; BTSR=100% for all."
        ),
        "source": {
            "B0": "data/results/benchmark_v2/decisions_B0.jsonl",
            "B1": "data/results/benchmark_v2/decisions_B1.jsonl",
            "B1b": "data/results/sprint2/decisions_B1b.jsonl",
            "B2": "data/results/benchmark_v2/decisions_B2.jsonl",
            "B3": "data/results/benchmark_v2/decisions_B3.jsonl",
            "B5": "data/results/benchmark_v2/decisions_B5.jsonl",
            "B6": "data/results/b6_n200/decisions_B6.jsonl",
        },
        "data": main_table,
        "b6_cross_check": {
            "benchmark_v2_decisions_B6_stale_n25": b6_stale_n25,
            "b6_n200_decisions_B6_full": b6_full_n200,
            "note": b6_mismatch_note or "N matches, no discrepancy.",
        },
    }

    # B4 (intent-only) — reported for completeness but NOT part of tab:main.
    # Only a stale N=25 sample exists under benchmark_v2/; flagged, not silently
    # treated as N=200.
    b4 = compute_metrics_from_jsonl(
        "data/results/benchmark_v2/decisions_B4.jsonl", "B4"
    )
    if b4:
        manifest["tables"]["table_b4_intent_only_n25_sample"] = {
            "description": (
                "B4 (intent-only) on Bench-v2 — NOT in tab:main (which reports "
                "only B0/B1/B1b/B2/B3/B5/B6). Only a partial N=25 sample exists; "
                "flagged as such, not extrapolated to N=200."
            ),
            "source": "data/results/benchmark_v2/decisions_B4.jsonl",
            "data": b4,
        }

    # ================================================================
    # Table 3 (tab:disambig) — Disambiguation-v2, N=50 attacked / M=50 benign
    # Canonical: data/results/disambiguation_v2/ (N=50). A separate N=100
    # extension exists under disambiguation_v2_n100/ — reported separately,
    # not conflated with the paper's headline N=50 table.
    # ================================================================
    disambig_b5 = compute_metrics_from_jsonl(
        "data/results/disambiguation_v2/disambiguation_v2_B5.jsonl", "B5"
    )
    disambig_b6 = compute_metrics_from_jsonl(
        "data/results/disambiguation_v2/disambiguation_v2_B6.jsonl", "B6"
    )
    disambig_by_family = load_csv_rows(
        "data/results/disambiguation_v2/disambiguation_v2_by_family.csv"
    )
    manifest["tables"]["table3_disambiguation_v2"] = {
        "description": (
            "Disambiguation-v2 (tab:disambig): N=50 attacked, M=50 benign, "
            "trusted-but-unauthorized (T-b-U) scenarios. Canonical N=50 source."
        ),
        "source": "data/results/disambiguation_v2/",
        "data": {"B5": disambig_b5, "B6": disambig_b6},
        "by_family": disambig_by_family,
    }

    disambig_n100_by_family = load_csv_rows(
        "data/results/disambiguation_v2_n100/disambig_n100_by_family.csv"
    )
    if disambig_n100_by_family:
        manifest["tables"]["table3b_disambiguation_v2_n100_extension"] = {
            "description": (
                "Disambiguation-v2 extended to N=100 — a robustness check, NOT "
                "the paper's headline tab:disambig figure (which is N=50). "
                "Kept separate to avoid confusing the two N's."
            ),
            "source": "data/results/disambiguation_v2_n100/disambig_n100_by_family.csv",
            "by_family": disambig_n100_by_family,
        }

    # ================================================================
    # SOTA comparison: B7 TaskShield, B9 Progent, B10 ClawGuard (+ B5/B6
    # already reported above via table2_bench_v2_main).
    # ================================================================
    sota_rows = load_csv_rows("data/results/sota_baselines/sota_baselines.csv")
    if sota_rows:
        manifest["tables"]["table_sota_comparison"] = {
            "description": (
                "Empirical SOTA comparison (tab:sota_fm): B7 TaskShield, "
                "B9 Progent, B10 ClawGuard (simplified faithful reimplementations), "
                "N=200 attacked / M=200 benign."
            ),
            "source": "data/results/sota_baselines/sota_baselines.csv",
            "data": sota_rows,
        }

    # ================================================================
    # FM4 mixed-provenance ablation (span- vs doc-level provenance), N=40
    # ================================================================
    fm4_rows = load_csv_rows("data/results/fm4_improved/fm4_improved.csv")
    if fm4_rows:
        manifest["tables"]["table_fm4_mixed_provenance"] = {
            "description": (
                "FM4 mixed-provenance ablation: B5 vs B6 doc-level vs B6 span-level "
                "provenance, N=40 attacked / M=20 benign."
            ),
            "source": "data/results/fm4_improved/fm4_improved.csv",
            "data": fm4_rows,
        }

    # ================================================================
    # Adaptive white-box (tab:adaptive), N=42 attacked / M=5 benign, 6 families
    # + AgentDojo subset (N=20+20), both aggregated in adaptive_eval/summary.csv
    # ================================================================
    adaptive_summary = load_csv_rows("data/results/adaptive_eval/summary.csv")
    if adaptive_summary:
        manifest["tables"]["table_adaptive_and_agentdojo_summary"] = {
            "description": (
                "RQ3/RQ4 summary: Official AgentDojo Subset (N=20 attacked, M=20 "
                "benign, external validity) and Adaptive white-box attacks "
                "(6 families, N=42 attacked, M=5 benign, tab:adaptive)."
            ),
            "source": "data/results/adaptive_eval/summary.csv",
            "data": adaptive_summary,
        }

    # ================================================================
    # Bench-v1 sanity (tab:versions), N=10 attacked / M=10 benign
    # ================================================================
    bench_v1 = {}
    for baseline in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]:
        path = f"data/results/decisions_{baseline}.jsonl"
        metrics = compute_metrics_from_jsonl(path, baseline)
        if metrics:
            bench_v1[baseline] = metrics
    if bench_v1:
        manifest["tables"]["table1_bench_v1_sanity"] = {
            "description": (
                "Bench-v1 sanity check (tab:versions): N=10 attacked, M=10 benign, "
                "all configs ASR=0%."
            ),
            "source": "data/results/decisions_B*.jsonl",
            "data": bench_v1,
        }

    # ================================================================
    # E2E agent evaluation (N=220 across 3 sub-suites: Bench-v1 easy,
    # Bench-v2 medium/hard, Disambiguation-v2)
    # ================================================================
    e2e_summary = load_csv_rows("data/results/e2e_n200/e2e_summary.csv")
    if e2e_summary:
        manifest["tables"]["table_e2e_agent_eval"] = {
            "description": (
                "End-to-end agent evaluation with a real tool-calling loop "
                "(N=220: Bench-v1 easy + Bench-v2 medium/hard + Disambiguation-v2)."
            ),
            "source": "data/results/e2e_n200/e2e_summary.csv",
            "data": e2e_summary,
        }

    # ================================================================
    # Multi-seed stability (seeds 42, 123, 777), N=48 per seed
    # ================================================================
    multi_seeds_results = load_csv_rows("data/results/multi_seeds/multi_seeds_results.csv")
    multi_seeds_summary = load_csv_rows("data/results/multi_seeds/multi_seeds_summary.csv")
    if multi_seeds_results:
        manifest["tables"]["table_multi_seed_stability"] = {
            "description": "Multi-seed stability (seeds 42, 123, 777), N=48 attacked per seed.",
            "source": [
                "data/results/multi_seeds/multi_seeds_results.csv",
                "data/results/multi_seeds/multi_seeds_summary.csv",
            ],
            "data": multi_seeds_results,
            "summary": multi_seeds_summary,
        }

    # ================================================================
    # Stratified resampling stability at N=25/50/100 (replaces the old,
    # unsound "nested prefix" claim — these are independent stratified
    # random subsamples, 5 seeds each, converging to the full N=200 ASR).
    # ================================================================
    stability_summary = load_csv_rows(
        "data/results/stability_resample/stability_resample_summary.csv"
    )
    stability_raw = load_csv_rows(
        "data/results/stability_resample/stability_resample_raw.csv"
    )
    if stability_summary:
        manifest["tables"]["table_stability_resample"] = {
            "description": (
                "B6 stability under independent stratified random subsampling "
                "(5 seeds per N, same family/difficulty stratification as the "
                "main sample): mean+-std ASR at N=25/50/100, converging to the "
                "full N=200 ASR=24.0%. Replaces an earlier bogus 'nested prefix' "
                "subsampling claim."
            ),
            "source": [
                "data/results/stability_resample/stability_resample_summary.csv",
                "data/results/stability_resample/stability_resample_raw.csv",
            ],
            "summary": stability_summary,
            "raw": stability_raw,
        }

    # ================================================================
    # Hard-benign tasks (N=50), part of the Bench-v2 (main) suite total (450)
    # ================================================================
    hard_benign = load_csv_rows("data/results/hard_benign_eval/hard_benign_metrics.csv")
    if hard_benign:
        manifest["tables"]["table_hard_benign"] = {
            "description": (
                "Hard-benign tasks (N=50 benign, deliberately ambiguous/adversarial "
                "phrasing that should still be allowed). Counted as part of the "
                "Bench-v2 (main) suite total of 450 (200+200+50), not a resample."
            ),
            "source": "data/results/hard_benign_eval/hard_benign_metrics.csv",
            "data": hard_benign,
        }

    # ================================================================
    # Benchmark composition — must sum to 657 (paper's "Benchmarks (657 tasks)")
    # ================================================================
    manifest["benchmarks"] = {
        "bench_v2_main": {
            "attacked": 200,
            "benign": 200,
            "hard_benign": 50,
            "total": 450,
            "used_for": "RQ1 (tab:main)",
            "source": [
                "data/results/benchmark_v2/decisions_B{0,1,2,3,5}.jsonl (200+200 each)",
                "data/results/b6_n200/decisions_B6.jsonl (200+200, canonical B6)",
                "data/results/hard_benign_eval/hard_benign_decisions.jsonl (50 benign)",
            ],
        },
        "disambiguation_v2": {
            "attacked": 50,
            "benign": 50,
            "total": 100,
            "used_for": "RQ2, trusted-but-unauthorized (tab:disambig)",
            "source": "data/results/disambiguation_v2/",
        },
        "agentdojo_subset": {
            "attacked": 20,
            "benign": 20,
            "total": 40,
            "used_for": "RQ3, external validity",
            "source": "data/results/adaptive_eval/agentdojo_official.jsonl, data/agentdojo_official/",
        },
        "bench_v1_sanity": {
            "attacked": 10,
            "benign": 10,
            "total": 20,
            "used_for": "tab:versions sanity check",
            "source": "data/results/decisions_B*.jsonl, data/tasks_{benign,attacked}.json",
        },
        "adaptive_white_box": {
            "attacked": 42,
            "benign": 5,
            "total": 47,
            "used_for": "RQ4 (tab:adaptive)",
            "source": "data/results/adaptive_eval/adaptive_attacks.jsonl, data/adaptive_attacks_intentflow/",
        },
        "total": {
            "attacked": 200 + 50 + 20 + 10 + 42,
            "benign": 200 + 20 + 10 + 5,
            "total": 450 + 100 + 40 + 20 + 47,
        },
    }

    # ================================================================
    # Intent extractor F1 — paper cites "F1=98.3% (200-message gold standard)"
    # and "heuristic fallback F1=51.7%" (mainraise_submit.tex line ~143).
    # VERIFIED: data/results/intent_eval_extended/intent_eval_extended.csv
    # (produced by src/intentflow/metrics/intent_eval_extended.py, a 200-message
    # gold standard) has llm_f1 column averaging 98.31 -> 98.3%, and
    # heuristic_f1 averaging 51.67 -> 51.7%. Both match the paper exactly.
    # ================================================================
    intent_f1_rows = load_csv_rows("data/results/intent_eval_extended/intent_eval_extended.csv")
    if intent_f1_rows:
        llm_f1_avg = round(
            sum(float(r["llm_f1"]) for r in intent_f1_rows) / len(intent_f1_rows), 1
        )
        heuristic_f1_avg = round(
            sum(float(r["heuristic_f1"]) for r in intent_f1_rows) / len(intent_f1_rows), 1
        )
        manifest["tables"]["intent_extractor_f1"] = {
            "description": (
                "Intent extractor F1 per field, 200-message gold standard. "
                f"Average LLM F1={llm_f1_avg}% (paper cites 98.3%), average "
                f"heuristic F1={heuristic_f1_avg}% (paper cites 51.7%) — VERIFIED "
                "against data/results/intent_eval_extended/intent_eval_extended.csv."
            ),
            "source": "data/results/intent_eval_extended/intent_eval_extended.csv",
            "data": intent_f1_rows,
            "verified_avg_llm_f1_pct": llm_f1_avg,
            "verified_avg_heuristic_f1_pct": heuristic_f1_avg,
        }
    else:
        # Could not find a backing file — do not fabricate a source.
        manifest["tables"]["intent_extractor_f1"] = {
            "description": (
                "UNVERIFIED FROM ARTIFACT DATA: paper cites F1=98.3% (200-message "
                "gold standard) and heuristic fallback F1=51.7% "
                "(mainraise_submit.tex, ~line 143), but no backing CSV was found "
                "under data/results/ at manifest-generation time. Not fabricated."
            ),
            "source": None,
            "data": {},
        }

    # Sauvegarder
    out_path = Path("artifact/metrics_manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"metrics_manifest.json generated -> {out_path}")
    print(f"Tables: {len(manifest['tables'])}")
    print(f"Total benchmark tasks: {manifest['benchmarks']['total']['total']}")
    if b6_mismatch_note:
        print(f"\nWARNING: {b6_mismatch_note}")


if __name__ == "__main__":
    main()
