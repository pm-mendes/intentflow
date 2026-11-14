# IntentFlow

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24233094.svg)](https://doi.org/10.5281/zenodo.24233094)

Research artifact of the paper

> **Intent-Conditioned Argument Mediation for Tool-Using LLM Agents: Blocking Trusted-but-Unauthorized Arguments**
> Yulliwas Ameur, Pedro Mendes, Samia Bouzefrane.
> ICAART 2027, 19th International Conference on Agents and Artificial Intelligence (short paper).

IntentFlow is a runtime mediator placed between an unmodified LLM planner and its tools. Before a tool call
executes, it checks each authority-bearing argument (recipient, participants, file path, endpoint) against an
intent contract built from the user's request and against the provenance of the value. It targets
*trusted-but-unauthorized* values: values that come from an admissible source (a directory, an internal
document) but are not the ones the user asked for.

This repository contains the implementation, the benchmarks, the decision logs and the scripts behind every
number of the paper. The version cited by the paper is the tag `icaart2027-camera-ready`, archived on Zenodo:
[10.5281/zenodo.24233094](https://doi.org/10.5281/zenodo.24233094) (this version) ;
[10.5281/zenodo.24233093](https://doi.org/10.5281/zenodo.24233093) (all versions).

## Benchmarks (637 tasks)

| Suite | Attacked | Benign | Total | Paper |
|---|---|---|---|---|
| Bench-v2 (main), `data/benchmark_v2/` | 200 | 200 + 50 hard-benign (`data/hard_benign/`) | 450 | Table 3, Section 5 |
| Disambiguation-v2, `data/disambiguation_v2/` | 50 | 50 | 100 | Table 4 |
| AgentDojo-derived replay, `data/agentdojo_official/` | 20 | 20 | 40 | Table 5 |
| Adaptive white-box, `data/adaptive_attacks_intentflow/` | 42 | 5 | 47 | Section 5.2 |
| **Total** | **312** | **325** | **637** | |

Section 5 also uses a mixed-provenance set (`data/mixed_provenance_v2/`, 40 + 20), an endpoint ablation
(30 + 30) and two further Bench-v2 instances (`data/benchmark_v2_seed123/`, `data/benchmark_v2_seed777/`).
The live campaign of Section 6 uses 110 attacked and 50 benign Bench-v2 tasks.

All controlled results use oracle intent contracts (`intentflow.baselines.runner.oracle_intent_for`) and
the schema-declared argument roles. The provenance tracker reads the benchmark's ground-truth
`is_injected` flag to label a document as injected; a deployment would have to infer it (see the paper,
Section 4).

## Reproducing the paper

Requirements: Python 3.11+, [Poetry](https://python-poetry.org/). No API key is needed for the controlled
results.

```bash
poetry install
bash artifact/scripts/run_all.sh --offline                 # ~10 min: regenerates every controlled result
PYTHONPATH=src poetry run python scripts/check_paper_numbers.py   # ~5 s: recomputes every number of the paper
```

`scripts/check_paper_numbers.py` recomputes each number of the paper (Tables 2–5, Sections 4–6) from the
files in `data/results/`, compares it with the printed value and writes
`artifact/results/paper_metrics_manifest.csv`: paper location, quantity, value in the paper, recomputed
value, result file and producing script. On this tag, all 95 values match. Decision latency is
machine-dependent; it matches only on the released log.

Every script below runs offline from the repository root and writes to the directory given as argument
(`PYTHONPATH=src python <script> <output_dir>`). Re-running them gives the released decisions exactly; only
`latency_ms` changes.

| Paper | Result files (`data/results/`) | Script |
|---|---|---|
| Table 3: B0, B2, B3 | `benchmark_v2/decisions_B{0,2,3}.jsonl` | `scripts/run_bench_v2_baselines.py` |
| Table 3: B4 action-level intent (all 400 tasks) | `benchmark_v2/decisions_B4_full.jsonl` | `scripts/run_b4_benchv2.py` |
| Table 3: B5, IF-Core, IF-Hardened; stability (seeds 123, 777); IF-Confirm | `partfix_seed{42,123,777}_{b5,if_core,if_hardened,if_confirm}/` | `scripts/run_bench_v2_configs.py` |
| Tables 3 and 4: provenance proxy, argument-level intent only, CaMeL\*, Progent\* | `r1_comparisons/` | `scripts/run_r1_comparisons.py` |
| Table 3: TaskShield\*, ClawGuard\* | `sota_baselines/sota_baselines.csv` | `python -m intentflow.experiments.sota_baselines` |
| Section 5: 50 hard-benign tasks (IF-Hardened, IF-Confirm, B5) | `hard_benign/hard_benign_paper_configs.jsonl` | `scripts/run_hard_benign_ifh.py` |
| Table 4: B5, IF-Core, IF-Hardened, IF-Confirm | `disambiguation_v2/` | `python -m intentflow.experiments.run_disambiguation_v2` |
| Section 5.1: endpoint ablation | `endpoint_ablation_n30/` | `python -m intentflow.experiments.run_endpoint_ablation_n30` |
| Table 5: AgentDojo-derived replay | `agentdojo_sota_comparison/` | `python -m intentflow.experiments.run_agentdojo_sota_comparison` |
| Section 5.2: adaptive attacks | `adaptive_eval/adaptive_attacks.jsonl` | `python -m intentflow.experiments.run_adaptive_eval` |
| Section 5.2: mixed provenance, per configuration | `mixed_provenance/mixed_provenance_configs.jsonl` | `scripts/run_mixed_provenance_configs.py` |

Scripts added for the camera-ready (November 2026): `run_r1_comparisons.py`, `run_b4_benchv2.py`,
`run_hard_benign_ifh.py`, `run_mixed_provenance_configs.py`, `run_bench_v2_baselines.py`,
`run_bench_v2_configs.py` and `check_paper_numbers.py`. The script that produced the `partfix_seed*` logs
on 28/09/2026 was not kept; `run_bench_v2_configs.py` regenerates all twelve logs with identical decisions
and provenance labels.

### Results that involve a hosted model (released as logs only)

| Paper | Result files (`data/results/`) | Producing code |
|---|---|---|
| Section 4: live `gpt-4o-mini` intent extraction on the 400 Bench-v2 tasks under IF-Hardened (0/400 decisions differ from oracle intent) | `if_hardened_livellm_postfix/decisions_B6.jsonl` | `BaselineRunner.run(..., use_oracle=False)` with `OPENAI_API_KEY` |
| Section 6: live planner campaign (`gpt-4o-mini-2024-07-18`, IF-Confirm, 110 + 50 tasks) | `e2e_phase3/` | `python -m intentflow.experiments.e2e_phase3` |

Hosted decoding is not deterministic: re-running these gives different logs.

### Files in `data/results/` that the paper does not use

- `benchmark_v2/decisions_B1.jsonl` (prompt-only) and `b1b_spotlighting/` (Spotlighting): these baselines
  were not run against a model. `baselines/spotlighting.py` simulates Spotlighting with fixed
  probabilities. Neither is reported in the paper.
- `fm4_improved/` (mixed provenance under the default policy, without SQL checks): superseded by
  `mixed_provenance/mixed_provenance_configs.jsonl`, which reports each named configuration.

## Known limitations of the prototype

- **Value–intent matching.** Since the camera-ready, an argument value matches the intent contract only if
  it is equal to an authorized value after role-specific normalization (membership for lists). The
  submitted version also accepted substring inclusion, so `manager@company.com.evil.io` matched a contract
  authorizing `manager@company.com`. The fix changes none of the paper's decisions (see `CHANGELOG.md`).
- **Injection labels.** The provenance tracker takes the benchmark's `is_injected` flag (Section 4 of the
  paper).
- **Disambiguation-v2 labels.** In `dv2_attacked_020` and `dv2_attacked_024`, the reference contract wrongly
  lists the attacker's endpoint. The published numbers are computed on the benchmark as released; the
  labels will be corrected in a later version of the benchmark.
- **Provenance source identifier.** When several injected documents contain the same value, the
  `source_id` recorded in a log may differ between runs (set iteration order depends on
  `PYTHONHASHSEED`). The origin label and the decision do not.

## Repository layout

```
src/intentflow/
  pipeline/mediator.py     ToolMediator.execute(), the entry point every configuration runs through
  intent/                  intent extraction (oracle, LLM, heuristic), endpoint registry, aliases
  provenance/tracker.py    argument provenance (intent match, user message, documents, spans)
  policy/engine.py         per-tool, per-argument policies; IF-Core, IF-Hardened, IF-Confirm
  policy/sql_hardening.py  SQL syntax-tree checks
  baselines/               BaselineRunner, oracle_intent_for, baseline helpers
  experiments/             one script per experiment of the paper (see table above)
data/                      benchmarks (tasks and documents) and data/results/ (decision logs, metrics)
scripts/                   camera-ready scripts, check_paper_numbers.py, log tools
artifact/                  run_all.sh, policies (YAML copies), Sigma export, paper_metrics_manifest.csv
tests/                     pytest suite
```

See `ARCHITECTURE.md` for the components and the data flow. Source comments and docstrings are in French.

## Tests

```bash
poetry run pytest tests/
```

## Citation

```bibtex
@inproceedings{ameur2027intentflow,
  author    = {Ameur, Yulliwas and Mendes, Pedro and Bouzefrane, Samia},
  title     = {Intent-Conditioned Argument Mediation for Tool-Using {LLM} Agents:
               Blocking Trusted-but-Unauthorized Arguments},
  booktitle = {Proceedings of the 19th International Conference on Agents and Artificial Intelligence (ICAART 2027)},
  publisher = {SciTePress},
  year      = {2027}
}

@software{ameur2026intentflow_artifact,
  author    = {Ameur, Yulliwas and Mendes, Pedro and Bouzefrane, Samia},
  title     = {{IntentFlow}: artifact of the {ICAART} 2027 paper on intent-conditioned argument mediation},
  version   = {icaart2027-camera-ready},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.24233094}
}
```

## License

MIT License, Copyright (c) 2026 Yulliwas Ameur, Pedro Mendes, Samia Bouzefrane. See `LICENSE`.
