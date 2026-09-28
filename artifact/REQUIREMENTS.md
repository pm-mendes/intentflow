# Reproduction Requirements — IntentFlow Artifact

## Hardware

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8 GB |
| Disk | 500 MB | 2 GB |
| GPU | Not required | Not required |

## Software

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.11+ | Required |
| Poetry | 1.8+ | Dependency manager |
| OpenAI API | — | Required for B4/B6 LLM mode only |

## Python Dependencies (key packages)

| Package | Version | Purpose |
|---|---|---|
| openai | ^1.40 | LLM-based intent extraction (B4, B6) |
| pydantic | ^2.8 | Data validation |
| pydantic-settings | ^2.3 | Configuration management |
| numpy | ^1.26 | Latency percentile computation |
| loguru | ^0.7 | Structured logging |
| tqdm | ^4.66 | Progress bars |
| pandas | ^2.2 | Results analysis |

Full dependency list with exact pinned versions: `poetry.lock`

## Reproduction Modes

The canonical entry point is `artifact/scripts/run_all.sh --quick|--offline|--with-llm`
(optionally with `--pseudonymize-logs`). Each mode runs a different subset of
`python -m intentflow.experiments.X` modules:

| Mode | Runtime | API key | What it runs |
|---|---|---|---|
| `--quick` | ~5 min | Not required | `data.generator`; baselines B0/B1/B2/B3/B5 via `run_baseline`; `compute_metrics` |
| `--offline` | ~60 min | Not required (B4/B6 use heuristic intent) | `data.generator`, `data.benchmark_v2`, `data.agentdojo_inspired`; baselines B0/B1/B1b/B2/B3/B4/B5/B6; `compute_metrics`; `run_disambiguation_v2`; `endpoint_ablation`; `intent_noise`; `label_noise`; `policy_fuzzing_extended` |
| `--with-llm` | ~90 min | Required (`OPENAI_API_KEY`) | Same data generation + baselines as `--offline`, plus `run_benchmark_v2`, `run_b6_n100`, `run_disambiguation_v2`, `run_sprint2`, `run_sprint3`, `endpoint_ablation`, `intent_noise`, `label_noise`, `run_mixed_provenance`, `policy_fuzzing_extended`, and regenerates `artifact/metrics_manifest.json` via `scripts/generate_metrics_manifest.py` |

`--quick` and `--offline` never call the OpenAI API — B4/B6 automatically fall
back to heuristic intent extraction (same security properties, reduced field
coverage). See `artifact/scripts/run_all.sh` for the exact, current command
list — the table above summarizes it, but that script is the source of truth.

Beyond `run_all.sh`, the full result set checked into `data/results/` also
includes suite-specific one-off scripts not wired into any single mode above
(e.g. `stability_resample`, `multi_seeds_eval`, `sota_baselines`,
`fm4_improved`, `run_adaptive_eval`, `run_hard_benign_eval`,
`e2e_agent_eval`/`e2e_n200`, `agentdojo_runner` — see
`src/intentflow/experiments/` for the complete list). These back the
robustness/ablation tables in the paper beyond the headline Table 2 and are
not required for a `--quick`/`--offline`/`--with-llm` run to succeed.

## API Cost Estimate (`--with-llm` mode)

Running B4/B6 with `OPENAI_API_KEY` set (gpt-4o-mini) across the full
`--with-llm` mode is inexpensive — on the order of a few hundred short
JSON-mode completions total across baselines + disambiguation + sprint2/3 +
mixed-provenance. The `--with-llm` banner in `artifact/scripts/run_all.sh`
itself estimates **~$0.20 total**. Actual cost depends on OpenAI pricing at
run time; check your OpenAI dashboard for a live figure rather than treating
this as a hard bound.

## Reproducibility Notes

- All experiments use `RANDOM_SEED=42` for deterministic data generation
- LLM outputs may vary slightly across API calls (temperature=0 minimizes variance)
- Latency measurements depend on hardware and network conditions
- Expected metrics for the headline Bench-v2 (N=200) Table 2 are provided in
  `artifact/results/expected_metrics.csv` (generated via
  `intentflow.metrics.evaluator.MetricsComputer` from the real decision logs
  under `data/results/benchmark_v2/`, `data/results/sprint2/`, and
  `data/results/b6_n200/` — see `README.md` for the exact source files and
  the Validation section for the `diff` command)
- `artifact/metrics_manifest.json` (via `scripts/generate_metrics_manifest.py`)
  tracks which result file backs every other paper table/figure, for
  reproducibility auditing beyond Table 2