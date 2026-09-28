# Architecture

Reference documentation for the IntentFlow codebase (baselines, components, data flow).

## Project

IntentFlow is a research artifact for a ICAART 2027 paper: "Intent-Conditioned Argument Mediation for
Retrieval-to-Action Security in Agentic RAG." It implements and benchmarks a runtime argument-level
mediation layer that defends agentic RAG tool-calling pipelines against prompt-injection / retrieval
corruption attacks, compared against baselines B0–B6 (see Architecture below). The repo is the
anonymized submission source (`README.md`, `pyproject.toml` authors/URLs are intentionally
anonymized placeholders — do not "fix" them).

Source code and docstrings are written in French; this is intentional project style, not an error.

## Commands

Dependency management is via Poetry.

```bash
poetry install                                    # install deps
poetry run python -c "import intentflow"          # sanity check
```

Tests:

```bash
poetry run pytest tests/unit/ -v                          # unit tests (no API key needed, heuristic mode)
poetry run pytest tests/unit/test_policy.py -v             # single file
poetry run pytest tests/unit/test_policy.py::TestPolicyEngine::test_blocks_injected_recipient -v  # single test
poetry run pytest --cov=src --cov-report=term-missing       # with coverage (this is the default addopts)
```

Lint / type-check (CI runs these three as separate blocking/non-blocking stages, see `.gitlab-ci.yml`):

```bash
poetry run ruff check src/ tests/
poetry run ruff format --check src/ tests/
poetry run mypy src/            # advisory only in CI (allow_failure)
```

Running experiments (baselines are B0–B6; B4/B6 need `OPENAI_API_KEY` or fall back to heuristic intent
extraction with reduced coverage but same security properties):

```bash
poetry run python -m intentflow.data.generator                          # generate synthetic tasks/docs
poetry run python -m intentflow.experiments.run_baseline --baseline B6  # run one baseline
poetry run python -m intentflow.experiments.compute_metrics             # aggregate metrics.csv from JSONL logs
bash scripts/run_all_experiments.sh                                     # full repro (data/results/metrics.csv)
bash artifact/scripts/run_all.sh --quick|--offline|--with-llm           # artifact repro modes (README-facing)
```

Numerous one-off experiment entry points live under `src/intentflow/experiments/` (e.g.
`run_disambiguation_v2`, `endpoint_ablation`, `intent_noise`, `label_noise`, `policy_fuzzing_extended`,
`run_sprint2`, `run_sprint3`, `run_mixed_provenance`) — each is a `python -m intentflow.experiments.X`
module producing JSONL decision logs under `data/results/`. Check `artifact/scripts/run_all.sh` for the
current canonical list and order when adding a new one.

Offline mode (no OpenAI key) works for everything except that B4/B6 degrade to heuristic intent
extraction: `OPENAI_API_KEY="" bash scripts/run_all_experiments.sh`.

## Architecture

### The core pipeline (`src/intentflow/pipeline/mediator.py`)

`ToolMediator.execute()` is the single entry point that every baseline runs through. Given a proposed
tool call `(tool_name, arguments, user_message)`, it dispatches to one of seven strategies selected by
the `baseline` argument, producing an `allowed` decision before optionally invoking the real tool
function:

- **B0** — no defense, always allow.
- **B1** — prompt-only (assumes the LLM was instructed to refuse injections); always allow in practice.
- **B2** — retrieval filter: blocks if an argument value appears verbatim in a document flagged
  `is_injected` (assumes a priori knowledge of malicious docs — an idealized upper bound).
- **B3** — tool allow-list only (`tool_name in TOOL_REGISTRY`).
- **B4** — intent-only: extracts `Intent` via `IntentExtractor` and checks `intent.action == tool_name`
  (no argument-level checking).
- **B5** — provenance-only (`_validate_provenance_only`): blocks an argument only if its provenance is
  `injected_document`; everything else (including untrusted-but-not-flagged internal docs) is allowed.
  This is the key baseline that IntentFlow (B6) improves upon.
- **B6** — IntentFlow full (`_validate_intentflow_full`): extracts intent → tracks provenance
  *conditioned on that intent* → runs `PolicyEngine.validate()`. This is what distinguishes B6 from B5:
  a value that traces to a trusted internal document is still blocked if the user's stated intent
  doesn't authorize it (e.g., `legal-team@company.com` found in an internal doc, but the user asked to
  email "my manager" — B5 allows it, B6 blocks it on intent mismatch).

### Supporting components (each consumed by the mediator)

- **`intent/extractor.py`** (`IntentExtractor`) — turns a raw user message into a structured `Intent`
  (`action`, `parameters`, `context`, `confidence`). Three modes: `oracle` (manual/ground-truth, used
  for controlled experiments), `llm` (gpt-4o-mini, temperature 0, strict-JSON system prompt), and
  `heuristic` (keyword/regex fallback, used in tests and when no API key is set — deliberately low
  confidence, 0.4, to reflect its unreliability).
- **`provenance/tracker.py`** (`ProvenanceTracker`) — assigns each tool-call argument an origin label
  from the integrity lattice in `models/models.py::PROVENANCE_LATTICE` (`injected_document` <
  `untrusted_external` < `user_provided` < `rag_document` < `trusted_internal` < `trusted_system` <
  `user_intent`). Cascade: match against extracted `Intent.parameters` → substring match against the
  raw user message → substring/fuzzy match (`SequenceMatcher`, threshold 0.7) against the RAG corpus →
  fallback `rag_inferred` with low confidence. Errs conservative: unclear provenance gets a low-trust
  label rather than being assumed safe.
- **`policy/engine.py`** (`PolicyEngine`) — the argument-level policy table (paper Section 4.5).
  Per-tool, per-argument rules of `allowed_origins` + `min_confidence`, categorized by argument *role*:
  `recipient`/`locator`/`principal` arguments (e.g. `send_email.recipient`, `write_file.path`,
  `run_query.endpoint`) require `user_intent` only at high confidence; `payload` arguments (e.g.
  `body`, `content`, `description`) allow `rag_document`/`rag_inferred` too since the LLM synthesizes
  them rather than copying verbatim. An argument sourced from `injected_document` is *always* blocked
  regardless of any other rule — this check runs first in `validate()`. Default policies are defined in
  `PolicyEngine.load_default_policies()`; `artifact/policies/default_policies.yaml` is the
  paper-facing/artifact copy of the same table.
- **`tools/registry.py`** — simulated tool implementations (`send_email`, `create_calendar_event`,
  `write_file`, `post_url`, `run_query`) in `TOOL_REGISTRY`. No real network/filesystem side effects by
  design (so the artifact runs with zero external credentials) — each does structural/format validation
  only (e.g. `write_file` rejects `..` path traversal, `send_email` regex-validates the address) and
  returns a synthetic result dict.
- **`models/models.py`** — the shared Pydantic contracts every component passes data through: `Intent`,
  `ProvenanceSource`, `ToolCall`, `RAGDocument` (`is_injected`/`injection_type` mark ground-truth attack
  content, used only for experiment construction — a real deployment would infer this rather than know
  it a priori), `BenignTask`/`AttackedTask` (task definitions with `expected_block` ground truth), and
  `MediationDecision` (the JSONL-loggable record consumed by the metrics pipeline).

### Metrics (`metrics/evaluator.py`, `metrics/models.py`)

`MetricsComputer.compute()` reads a JSONL decision log and derives (Section 5.4): PCR (Parameter
Corruption Rate), TMR (Tool Misuse Rate), ASR (Attack Success Rate — fraction of attacks executed
unblocked), BTSR (Benign Task Success Rate), PBR (Policy Block Rate), FBR (False Block Rate, i.e.
false positives on benign tasks), and latency median/p95 (via `numpy.percentile`, not naive indexing,
since indexing breaks for N<20). Lower is better for PCR/TMR/ASR/FBR; higher is better for BTSR.

### Data flow for an experiment run

1. `data/generator.py` (or one of the specialized generators in `src/intentflow/data/`, e.g.
   `benchmark_v2.py`, `agentdojo_official.py`, `mixed_provenance_v2.py`, `hard_benign_tasks.py`,
   `adaptive_attacks_intentflow.py`) produces `BenignTask`/`AttackedTask`/`RAGDocument` fixtures as JSON
   under `data/`.
2. `baselines/runner.py` (`BaselineRunner`) drives `ToolMediator.execute()` over every task for a given
   baseline and writes one `MediationDecision` per task as a JSONL line under `data/results/`.
3. `experiments/compute_metrics.py` (or a per-experiment variant) aggregates JSONL → `MetricsResult` →
   CSV.
4. `scripts/generate_metrics_manifest.py` / `artifact/metrics_manifest.json` track which result files
   back which paper table/figure, for artifact reproducibility auditing.

Many files under `src/intentflow/experiments/` are one-off scripts for a specific paper
section/ablation/robustness check (label noise, intent noise, endpoint ablation, multi-seed, McNemar
significance tests, SOTA baseline comparisons, AgentDojo integration) rather than shared library code —
when adding a new ablation, follow the pattern of an existing sibling script rather than generalizing
the mediator/policy engine itself.

### Configuration

`src/intentflow/utils/config.py` (`Settings`, pydantic-settings) loads from `.env` — copy
`.env.example` to `.env` and set `OPENAI_API_KEY` for LLM-based intent extraction (B4/B6). Everything
else has documented defaults (`RANDOM_SEED=42` for reproducibility, `INJECTED_DOCUMENT_RATIO=0.3`,
etc.).
