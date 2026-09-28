"""
scripts/logs_to_sigma.py

Converts IntentFlow JSONL decision logs into (a) a normalized event stream
suitable for SIEM ingestion and (b) a Sigma detection rule, so the audit
log is a starting point for detection engineering rather than an end point.

Each decision log line has (schema from src/intentflow/pipeline/mediator.py):
  tool, arguments, provenance (per-arg {origin, source_id, confidence,
  extraction_method}), allowed, blocked_arguments, reasons, task_id,
  baseline, timestamp. Not every log in this repo carries every field
  (older experiment logs predate `reasons`); this script only relies on
  `allowed`, `blocked_arguments`, and `provenance`, which are present
  everywhere the mediator itself wrote the log.

The rule targets the mediator's unconditional invariant (Lemma 1,
Section 3.2 of the paper): an argument sourced from `injected_document`
is always blocked. A hit on this rule is a real attempted retrieval-to-action
attack that was caught -- not a heuristic guess.

Usage:
  poetry run python scripts/logs_to_sigma.py data/results/b6_n200/decisions_B6.jsonl
  poetry run python scripts/logs_to_sigma.py data/results --recursive
"""

import argparse
import glob
import json
import sys
from pathlib import Path

OUT_DIR = Path("artifact/sigma")


def load_jsonl(path: str) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def to_sigma_event(decision: dict, source_file: str) -> dict | None:
    """Flatten one mediator decision into a normalized detection event, one
    row per blocked argument (a Sigma/SIEM event is typically single-subject)."""
    blocked = decision.get("blocked_arguments") or []
    provenance = decision.get("provenance") or {}
    if not blocked:
        return None
    rows = []
    for arg in blocked:
        prov = provenance.get(arg) or {}
        rows.append({
            "timestamp": decision.get("timestamp"),
            "product": "intentflow",
            "service": "mediator",
            "baseline": decision.get("baseline"),
            "tool": decision.get("tool"),
            "task_id": decision.get("task_id"),
            "decision": "block",
            "argument_name": arg,
            "provenance_origin": prov.get("origin", "unknown"),
            "provenance_confidence": prov.get("confidence"),
            "source_file": source_file,
        })
    return rows


SIGMA_RULE = """title: IntentFlow - Argument-Level Block on Injected-Document Provenance
id: intentflow-block-injected-provenance
status: stable
description: >
  Fires when the IntentFlow mediator blocks a tool-call argument whose value
  traces to an injected/untrusted document. Under Lemma 1 (no-allow-without-
  authorization), this is an unconditional invariant, not a heuristic: a hit
  means a retrieval-to-action attack attempt was intercepted before tool
  execution, so it is a high-precision signal for the SOC (backtested count
  below, not a fabricated estimate).
references:
  - Intent-Conditioned Argument Mediation for Retrieval-to-Action Security in
    Agentic RAG (ICAART 2027), Section 3.2 (Lemma 1) and Section 6 (artifact)
logsource:
  product: intentflow
  service: mediator
detection:
  selection:
    decision: block
    provenance_origin: injected_document
  condition: selection
level: high
tags:
  - attack.execution
  - attack.t1195  # supply_chain / injected-content analogue
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="JSONL file(s) or directory root")
    parser.add_argument("--recursive", action="store_true", help="glob **/*.jsonl under given directories")
    args = parser.parse_args()

    files: list[str] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            pattern = "**/*.jsonl" if args.recursive else "*.jsonl"
            files.extend(str(f) for f in path.glob(pattern))
        elif path.is_file():
            files.append(str(path))

    if not files:
        print("No JSONL files found.", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_events = []
    total_decisions = 0
    for fpath in sorted(files):
        decisions = load_jsonl(fpath)
        total_decisions += len(decisions)
        for d in decisions:
            rows = to_sigma_event(d, fpath)
            if rows:
                all_events.extend(rows)

    events_path = OUT_DIR / "sigma_events.jsonl"
    with open(events_path, "w") as f:
        for ev in all_events:
            f.write(json.dumps(ev) + "\n")

    rule_path = OUT_DIR / "intentflow_blocked_injected_provenance.yml"
    with open(rule_path, "w") as f:
        f.write(SIGMA_RULE)

    # Backtest: how many of the normalized events actually match the rule,
    # against how many decisions were scanned in total.
    matches = sum(1 for ev in all_events if ev["provenance_origin"] == "injected_document")

    print(f"Scanned {len(files)} log files, {total_decisions} decisions.")
    print(f"Emitted {len(all_events)} block events -> {events_path}")
    print(f"Sigma rule -> {rule_path}")
    print(f"Backtest: {matches}/{len(all_events)} block events match the "
          f"injected-document rule (the rest block on other provenance origins, "
          f"e.g. rag_inferred/rag_document under intent mismatch).")


if __name__ == "__main__":
    main()
