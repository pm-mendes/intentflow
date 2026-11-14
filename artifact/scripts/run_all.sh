#!/usr/bin/env bash
# ============================================================
# artifact/scripts/run_all.sh — IntentFlow Reproduction Script
# Supports three modes: --quick, --offline, --with-llm,
# and an optional --pseudonymize-logs flag.
#
# Usage:
#   bash artifact/scripts/run_all.sh --quick                       # ~5 min, no API, B0-B5 only
#   bash artifact/scripts/run_all.sh --offline                     # ~10 min, no API: every controlled result of the paper
#   bash artifact/scripts/run_all.sh --with-llm                    # ~90 min, requires OPENAI_API_KEY
#   bash artifact/scripts/run_all.sh --offline --pseudonymize-logs # also emit *.pseudonymized.jsonl
# ============================================================

set -euo pipefail

MODE="${1:---offline}"
PSEUDONYMIZE=0
for arg in "$@"; do
    if [ "$arg" = "--pseudonymize-logs" ]; then
        PSEUDONYMIZE=1
    fi
done

echo "=== IntentFlow ICAART 2027 — Reproduction ($MODE) ==="
echo "Started: $(date)"
echo ""

# Environment check
poetry run python -c "import intentflow; print('✓ intentflow loaded')"

case "$MODE" in

  --quick)
    echo "--- QUICK MODE (~5 min, no LLM, B0-B5 only) ---"
    poetry run python -m intentflow.data.generator
    for b in B0 B1 B2 B3 B5; do
        poetry run python -m intentflow.experiments.run_baseline --baseline "$b"
    done
    poetry run python -m intentflow.experiments.compute_metrics
    echo ""
    echo "Expected runtime: ~5 min"
    echo "Note: B4 and B6 skipped (require OPENAI_API_KEY)"
    echo "Results: data/results/metrics.csv"
    ;;

  --offline)
    echo "--- OFFLINE MODE: every controlled result of the ICAART 2027 paper (no API key, ~10 min) ---"
    # Regenerates the decision logs behind Tables 3-5 and Section 5 in place, from the released
    # benchmark files (data/*/tasks_*.json are not regenerated), then recomputes every number of the
    # paper and writes artifact/results/paper_metrics_manifest.csv. Only latency_ms changes from run to run.
    # Results that involve a hosted model (live-extraction check, live campaign) are released as logs only.
    export OPENAI_API_KEY=""
    export PYTHONPATH=src
    R=data/results
    poetry run python scripts/run_bench_v2_baselines.py "$R/benchmark_v2"      # Table 3: B0, B2, B3
    poetry run python scripts/run_b4_benchv2.py "$R/benchmark_v2"              # Table 3: B4
    poetry run python scripts/run_bench_v2_configs.py "$R"                     # Table 3: B5, IF-Core, IF-Hardened; seeds 123/777
    poetry run python scripts/run_r1_comparisons.py "$R/r1_comparisons"        # Tables 3-4: proxies, CaMeL*, Progent*, intent only
    poetry run python -m intentflow.experiments.sota_baselines                 # Table 3: TaskShield*, ClawGuard*
    poetry run python scripts/run_hard_benign_ifh.py "$R/hard_benign"          # Section 5: 50 hard-benign tasks
    poetry run python -m intentflow.experiments.run_disambiguation_v2          # Table 4: B5, IF-Core, IF-Hardened, IF-Confirm
    poetry run python -m intentflow.experiments.run_endpoint_ablation_n30      # Section 5.1: endpoint ablation
    poetry run python -m intentflow.experiments.run_agentdojo_sota_comparison  # Table 5
    poetry run python -m intentflow.experiments.run_adaptive_eval              # Section 5.2: adaptive attacks
    poetry run python scripts/run_mixed_provenance_configs.py "$R/mixed_provenance"  # Section 5.2: mixed provenance
    poetry run python scripts/check_paper_numbers.py                           # every number vs the paper
    echo ""
    echo "Results: data/results/ ; manifest: artifact/results/paper_metrics_manifest.csv"
    ;;

  --with-llm)
    echo "--- FULL LLM MODE (~90 min, requires OPENAI_API_KEY) ---"
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "ERROR: OPENAI_API_KEY not set. Set it in .env or export OPENAI_API_KEY=sk-..."
        exit 1
    fi
    poetry run python -m intentflow.data.generator
    poetry run python -m intentflow.data.benchmark_v2
    poetry run python -m intentflow.data.agentdojo_inspired
    for b in B0 B1 B1b B2 B3 B4 B5 B6; do
        poetry run python -m intentflow.experiments.run_baseline --baseline "$b"
    done
    poetry run python -m intentflow.experiments.compute_metrics
    poetry run python -m intentflow.experiments.run_benchmark_v2
    poetry run python -m intentflow.experiments.run_b6_n100
    poetry run python -m intentflow.experiments.run_disambiguation_v2
    poetry run python -m intentflow.experiments.run_sprint2
    poetry run python -m intentflow.experiments.run_sprint3
    poetry run python -m intentflow.experiments.endpoint_ablation
    poetry run python -m intentflow.experiments.intent_noise
    poetry run python -m intentflow.experiments.label_noise
    poetry run python -m intentflow.experiments.run_mixed_provenance
    poetry run python -m intentflow.experiments.policy_fuzzing_extended
    poetry run python scripts/check_paper_numbers.py
    echo ""
    echo "Expected runtime: ~90 min"
    echo "Estimated API cost: ~\$0.20 (gpt-4o-mini)"
    echo "Results: data/results/"
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash artifact/scripts/run_all.sh [--quick|--offline|--with-llm] [--pseudonymize-logs]"
    exit 1
    ;;
esac

if [ "$PSEUDONYMIZE" = "1" ]; then
    echo ""
    echo "--- Pseudonymizing JSONL decision logs (SHA-256, per-run salt) ---"
    poetry run python scripts/pseudonymize_logs.py data/results --recursive
fi

echo ""
echo "=== Done: $(date) ==="
echo ""
echo "Expected runtime:"
echo "  --quick    : ~5 min"
echo "  --offline  : ~10 min"
echo "  --with-llm : ~90 min"
