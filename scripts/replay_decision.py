"""
artifact/scripts/replay_decision.py

Replays a single argument-level mediation decision from a JSONL audit log.
Allows reviewers to verify any decision independently.

Usage:
    python replay_decision.py --task dv2_attacked_001 --baseline B6
    python replay_decision.py --task dv2_attacked_001 --log path/to/decisions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def find_decision(log_path: str, task_id: str) -> list[dict]:
    """Find all argument decisions for a given task."""
    decisions = []
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("task_id") == task_id:
                    decisions.append(d)
    except FileNotFoundError:
        print(f"ERROR: Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    return decisions


def format_decision(d: dict) -> str:
    """Format a single decision for display."""
    lines = [
        f"  task_id     : {d.get('task_id', 'N/A')}",
        f"  baseline    : {d.get('baseline', 'N/A')}",
        f"  tool        : {d.get('tool', 'N/A')}",
        f"  is_benign   : {d.get('is_benign', 'N/A')}",
        f"  allowed     : {d.get('allowed', 'N/A')}",
        f"  executed    : {d.get('executed', 'N/A')}",
    ]
    if "blocked_arguments" in d and d["blocked_arguments"]:
        lines.append(f"  blocked_args: {d['blocked_arguments']}")
    if "attack_type" in d and d["attack_type"]:
        lines.append(f"  attack_type : {d['attack_type']}")
    if "latency_ms" in d:
        lines.append(f"  latency_ms  : {d['latency_ms']:.1f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Replay an IntentFlow argument-level mediation decision."
    )
    parser.add_argument("--task", required=True, help="Task ID to replay")
    parser.add_argument("--baseline", default="B6", help="Baseline (default: B6)")
    parser.add_argument(
        "--log", default=None,
        help="Path to JSONL log file (default: auto-detect from baseline)"
    )
    parser.add_argument(
        "--log-dir", default="data/results",
        help="Directory containing JSONL logs"
    )
    args = parser.parse_args()

    # Auto-detect log file
    if args.log:
        log_path = args.log
    else:
        candidates = [
            f"{args.log_dir}/decisions_{args.baseline}.jsonl",
            f"{args.log_dir}/benchmark_v2/{args.baseline}_decisions.jsonl",
            f"{args.log_dir}/disambiguation_v2/disambiguation_v2_{args.baseline}.jsonl",
            f"{args.log_dir}/b6_n100/decisions_{args.baseline}.jsonl",
        ]
        log_path = None
        for c in candidates:
            if Path(c).exists():
                log_path = c
                break
        if not log_path:
            # Search all JSONL files
            results_dir = Path(args.log_dir)
            all_logs = list(results_dir.rglob("*.jsonl"))
            for log in all_logs:
                decs = find_decision(str(log), args.task)
                if decs:
                    log_path = str(log)
                    break
        if not log_path:
            print(f"ERROR: No JSONL log found for task '{args.task}' in {args.log_dir}")
            print("Available logs:")
            for log in Path(args.log_dir).rglob("*.jsonl"):
                print(f"  {log}")
            sys.exit(1)

    decisions = find_decision(log_path, args.task)

    if not decisions:
        print(f"Task '{args.task}' not found in {log_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"REPLAY: task={args.task} | baseline={args.baseline}")
    print(f"Log: {log_path}")
    print(f"{'='*60}")

    for d in decisions:
        if d.get("baseline", args.baseline) == args.baseline:
            print(format_decision(d))
            print()

    # Also show raw JSON for verification
    print("Raw JSON (for diff verification):")
    for d in decisions:
        if d.get("baseline", args.baseline) == args.baseline:
            print(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()