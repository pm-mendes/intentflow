"""
scripts/pseudonymize_logs.py

Post-processing pseudonymization for IntentFlow JSONL decision logs
(paper Section "Privacy and security risks of IntentFlow itself" / Compliance).

Decision logs record, per call, the raw recipient/path/endpoint/URL values that
were evaluated. This script replaces email addresses, URLs, and filesystem paths
found in the loggable fields of each MediationDecision record with a salted
SHA-256 digest, so the pseudonymized copy can be shared (e.g. in an artifact)
without exposing the underlying values. The salt is per-run: a fresh random
salt is generated unless --salt is given explicitly (e.g. to make two runs
comparable by the same hash, at the cost of making them dictionary-attackable
against each other — see the paper's Limitations for this tradeoff).

Usage:
    python scripts/pseudonymize_logs.py data/results/decisions_B6.jsonl
    python scripts/pseudonymize_logs.py data/results/ --recursive
"""

import argparse
import hashlib
import json
import re
import secrets
import sys
from pathlib import Path
from typing import Any, Callable

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
URL_RE = re.compile(r"https?://[^\s'\"]+")
PATH_RE = re.compile(r"(?:/[\w.\-]+){2,}")

# Fields of MediationDecision (see models/models.py) that may contain raw
# authority-bearing values and should be pseudonymized.
LOGGABLE_FIELDS = ("arguments", "user_message", "reasons", "provenance", "execution_result")


def make_hasher(salt: str) -> Callable[[str], str]:
    def _hash(value: str) -> str:
        digest = hashlib.sha256((salt + value.lower()).encode()).hexdigest()[:16]
        return f"<redacted:{digest}>"

    return _hash


def pseudonymize_text(text: str, hasher: Callable[[str], str]) -> str:
    text = EMAIL_RE.sub(lambda m: hasher(m.group(0)), text)
    text = URL_RE.sub(lambda m: hasher(m.group(0)), text)
    text = PATH_RE.sub(lambda m: hasher(m.group(0)), text)
    return text


def pseudonymize_value(value: Any, hasher: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return pseudonymize_text(value, hasher)
    if isinstance(value, list):
        return [pseudonymize_value(v, hasher) for v in value]
    if isinstance(value, dict):
        return {k: pseudonymize_value(v, hasher) for k, v in value.items()}
    return value


def pseudonymize_record(record: dict, hasher: Callable[[str], str]) -> dict:
    record = dict(record)
    for field in LOGGABLE_FIELDS:
        if field in record:
            record[field] = pseudonymize_value(record[field], hasher)
    return record


def pseudonymize_file(in_path: Path, out_path: Path, hasher: Callable[[str], str]) -> int:
    n = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            fout.write(json.dumps(pseudonymize_record(record, hasher)) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pseudonymize (SHA-256, per-run salt) IntentFlow JSONL decision logs."
    )
    parser.add_argument("path", help="A single .jsonl file, or a directory to scan")
    parser.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories when path is a directory"
    )
    parser.add_argument(
        "--salt",
        default=None,
        help="Explicit salt (default: a fresh random salt, printed to stderr, not persisted)",
    )
    parser.add_argument(
        "--suffix",
        default=".pseudonymized.jsonl",
        help="Suffix appended to each output file (default: .pseudonymized.jsonl)",
    )
    args = parser.parse_args()

    salt = args.salt or secrets.token_hex(16)
    if not args.salt:
        print(f"[pseudonymize_logs] per-run salt (not persisted): {salt}", file=sys.stderr)
    hasher = make_hasher(salt)

    src = Path(args.path)
    if src.is_file():
        targets = [src]
    else:
        pattern = "**/*.jsonl" if args.recursive else "*.jsonl"
        targets = [p for p in sorted(src.glob(pattern)) if not p.name.endswith(args.suffix)]

    if not targets:
        print(f"[pseudonymize_logs] no .jsonl files found under {src}", file=sys.stderr)
        return

    total = 0
    for in_path in targets:
        out_path = in_path.with_name(in_path.stem + args.suffix)
        n = pseudonymize_file(in_path, out_path, hasher)
        print(f"[pseudonymize_logs] {in_path} -> {out_path} ({n} records)")
        total += n

    print(f"[pseudonymize_logs] done: {len(targets)} file(s), {total} record(s) pseudonymized")


if __name__ == "__main__":
    main()
