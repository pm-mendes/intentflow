"""
intentflow/experiments/label_noise_v2.py

Label noise étendu avec 5 types de bruit distincts — recommandation #7.

Types de bruit :
1. external_marked_internal  : doc injecté → trusted_internal (pire cas)
2. internal_marked_unknown   : doc bénin → unknown_origin
3. forwarded_marked_trusted  : external forwarded → trusted_internal
4. sender_metadata_missing   : source_domain → None/unknown
5. domain_spoofing           : domaine external ressemble à internal

Répond au point 7 du professeur : "ajouter différents types de bruit".
"""

import json
import csv
import copy
import random
from pathlib import Path
from intentflow.models.models import BenignTask, AttackedTask, RAGDocument
from intentflow.pipeline.mediator import ToolMediator
from intentflow.policy.engine import PolicyEngine
from intentflow.tools.registry import get_tool
from intentflow.metrics.confidence_intervals import wilson_interval
from intentflow.utils.logger import logger

SEED = 42
RESULTS_DIR = Path("data/results/label_noise_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NOISE_TYPES = [
    "external_marked_internal",
    "internal_marked_unknown",
    "forwarded_marked_trusted",
    "sender_metadata_missing",
    "domain_spoofing",
]

NOISE_LABELS = {
    "external_marked_internal": "External -> trusted\\_internal",
    "internal_marked_unknown": "Internal -> unknown\\_origin",
    "forwarded_marked_trusted": "Forwarded ext -> trusted",
    "sender_metadata_missing": "Sender metadata missing",
    "domain_spoofing": "Domain spoofing",
}


def apply_typed_noise(
    documents: list[RAGDocument],
    noise_rate: float,
    noise_type: str,
    seed: int,
) -> list[RAGDocument]:
    rng = random.Random(seed)
    noisy = [copy.deepcopy(d) for d in documents]
    n_corrupt = max(1, int(len(noisy) * noise_rate))
    indices = rng.sample(range(len(noisy)), min(n_corrupt, len(noisy)))

    for idx in indices:
        doc = noisy[idx]

        if noise_type == "external_marked_internal" and doc.is_injected:
            noisy[idx] = RAGDocument(doc_id=doc.doc_id, content=doc.content,
                is_injected=False, source_domain="trusted_internal")

        elif noise_type == "internal_marked_unknown" and not doc.is_injected:
            noisy[idx] = RAGDocument(doc_id=doc.doc_id, content=doc.content,
                is_injected=False, source_domain="unknown_origin")

        elif noise_type == "forwarded_marked_trusted" and doc.is_injected:
            noisy[idx] = RAGDocument(doc_id=doc.doc_id, content=doc.content,
                is_injected=False, source_domain="trusted_internal",
                injection_type="forwarded_external")

        elif noise_type == "sender_metadata_missing":
            noisy[idx] = RAGDocument(doc_id=doc.doc_id, content=doc.content,
                is_injected=False, source_domain="unknown_origin")

        elif noise_type == "domain_spoofing" and doc.is_injected:
            # Simule un domaine spoofed : untrusted_external ressemble à trusted
            noisy[idx] = RAGDocument(doc_id=doc.doc_id, content=doc.content,
                is_injected=False, source_domain="trusted_internal")

    return noisy


def _run_b5_b6_with_noise(
    noise_type: str,
    noise_rate: float,
    benign_tasks: list,
    attacked_tasks: list,
    documents: list,
    n_reps: int = 3,
) -> dict:
    policy_engine = PolicyEngine.load_default_policies()

    all_b5_asr, all_b6_asr = [], []
    all_btsr, all_fbr = [], []

    for rep in range(n_reps):
        noisy_docs = apply_typed_noise(documents, noise_rate, noise_type, seed=SEED + rep * 100)
        mediator = ToolMediator(policy_engine=policy_engine, documents=noisy_docs)

        b5_exec = b6_exec = 0
        b5_ben = b6_ben = 0
        b5_fbr = b6_fbr = 0

        for task in attacked_tasks:
            tool_fn = get_tool(task.expected_tool)
            if not tool_fn:
                continue
            args = {**task.expected_arguments, **task.corrupted_arguments}
            r5 = mediator.execute(task.expected_tool, args, task.user_message,
                                  tool_fn, baseline="B5", context_documents=noisy_docs)
            r6 = mediator.execute(task.expected_tool, args, task.user_message,
                                  tool_fn, baseline="B6", context_documents=noisy_docs)
            if r5["executed"]:
                b5_exec += 1
            if r6["executed"]:
                b6_exec += 1

        for task in benign_tasks:
            tool_fn = get_tool(task.expected_tool)
            if not tool_fn:
                continue
            r5 = mediator.execute(task.expected_tool, task.expected_arguments,
                                  task.user_message, tool_fn, baseline="B5",
                                  context_documents=noisy_docs)
            r6 = mediator.execute(task.expected_tool, task.expected_arguments,
                                  task.user_message, tool_fn, baseline="B6",
                                  context_documents=noisy_docs)
            if r5["executed"]:
                b5_ben += 1
            else:
                b5_fbr += 1
            if r6["executed"]:
                b6_ben += 1
            else:
                b6_fbr += 1

        n_a = len(attacked_tasks)
        n_b = len(benign_tasks)
        all_b5_asr.append(b5_exec / n_a * 100 if n_a > 0 else 0)
        all_b6_asr.append(b6_exec / n_a * 100 if n_a > 0 else 0)
        all_btsr.append(b6_ben / n_b * 100 if n_b > 0 else 0)
        all_fbr.append(b6_fbr / n_b * 100 if n_b > 0 else 0)

    return {
        "b5_asr": round(sum(all_b5_asr) / n_reps, 1),
        "b6_asr": round(sum(all_b6_asr) / n_reps, 1),
        "btsr": round(sum(all_btsr) / n_reps, 1),
        "fbr": round(sum(all_fbr) / n_reps, 1),
    }


def main():
    bench_dir = Path("data")
    benign = [BenignTask(**t) for t in json.loads((bench_dir / "tasks_benign.json").read_text())]
    attacked = [AttackedTask(**t) for t in json.loads((bench_dir / "tasks_attacked.json").read_text())]
    documents = [RAGDocument(**d) for d in json.loads((bench_dir / "documents.json").read_text())]

    noise_rates = [0.0, 0.05, 0.10, 0.20, 0.30]
    rows = []

    print("\n" + "=" * 90)
    print("LABEL NOISE ROBUSTNESS v2 — 5 noise types × 5 noise rates")
    print("=" * 90)

    for noise_type in ["clean"] + NOISE_TYPES:
        print(f"\n--- {NOISE_LABELS.get(noise_type, noise_type)} ---")
        print(f"{'Noise':>7} {'B5 ASR':>8} {'B6 ASR':>8} {'BTSR':>8} {'FBR':>8}")
        print("-" * 45)

        for noise_rate in noise_rates:
            if noise_type == "clean":
                r = _run_b5_b6_with_noise(
                    "external_marked_internal", 0.0, benign, attacked, documents)
            else:
                r = _run_b5_b6_with_noise(
                    noise_type, noise_rate, benign, attacked, documents)

            print(f"{noise_rate:>6.0%} {r['b5_asr']:>7.1f}% {r['b6_asr']:>7.1f}% "
                  f"{r['btsr']:>7.1f}% {r['fbr']:>7.1f}%")
            rows.append({
                "noise_type": NOISE_LABELS.get(noise_type, noise_type),
                "noise_rate": noise_rate,
                **r,
            })

            if noise_type == "clean":
                break  # Pas besoin d'itérer les rates pour clean

    with open(RESULTS_DIR / "label_noise_v2.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["noise_type","noise_rate","b5_asr","b6_asr","btsr","fbr"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Results saved to {RESULTS_DIR}/label_noise_v2.csv")


if __name__ == "__main__":
    main()