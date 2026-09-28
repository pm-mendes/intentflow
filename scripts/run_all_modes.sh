#!/usr/bin/env bash
# ============================================================
# artifact/scripts/run_all.sh — IntentFlow Reproduction Script
# Supports three modes: --offline, --with-llm, --quick
#
# Usage:
#   bash scripts/run_all.sh --quick      # ~5 min, no API, B0-B5 only
#   bash scripts/run_all.sh --offline    # ~60 min, no API, heuristic intent
#   bash scripts/run_all.sh --with-llm  # ~90 min, requires OPENAI_API_KEY
# ============================================================

set -euo pipefail

MODE="${1:---offline}"
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
    echo "--- OFFLINE MODE (~60 min, heuristic intent fallback) ---"
    export OPENAI_API_KEY=""
    poetry run python -m intentflow.data.generator
    poetry run python -m intentflow.data.benchmark_v2
    poetry run python -m intentflow.data.agentdojo_inspired
    for b in B0 B1 B1b B2 B3 B4 B5 B6; do
        poetry run python -m intentflow.experiments.run_baseline --baseline "$b"
    done
    poetry run python -m intentflow.experiments.compute_metrics
    poetry run python -m intentflow.experiments.run_disambiguation_v2
    poetry run python -m intentflow.experiments.endpoint_ablation
    poetry run python -m intentflow.experiments.intent_noise
    poetry run python -m intentflow.experiments.label_noise
    poetry run python -m intentflow.experiments.policy_fuzzing_extended
    echo ""
    echo "Expected runtime: ~60 min"
    echo "Note: B4/B6 use heuristic intent (no API cost, reduced coverage)"
    echo "Results: data/results/"
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
    python scripts/generate_metrics_manifest.py
    echo ""
    echo "Expected runtime: ~90 min"
    echo "Estimated API cost: ~\$0.20 (gpt-4o-mini)"
    echo "Results: data/results/"
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash scripts/run_all.sh [--quick|--offline|--with-llm]"
    exit 1
    ;;
esac

echo ""
echo "=== Done: $(date) ==="
echo ""
echo "Expected runtime:"
echo "  --quick    : ~5 min"
echo "  --offline  : ~60 min"
echo "  --with-llm : ~90 min"