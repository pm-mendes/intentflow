"""
intentflow/baselines/semantic_provenance_b8.py

B8 — Semantic Provenance via Sentence Embeddings — Point 15.

Uses sentence-transformers for embedding-based provenance tracking.
Generates ASR-FBR curve across thresholds [0.3, 0.5, 0.7, 0.9].
Compares B5, B6, B7, B8.
"""

import json
import numpy as np
from pathlib import Path
from intentflow.models.models import BenignTask, AttackedTask, RAGDocument, ProvenanceSource
from intentflow.policy.engine import PolicyEngine
from intentflow.utils.logger import logger

RESULTS_DIR = Path("data/results/b8_semantic")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(min(1.0, np.dot(a, b) / (norm_a * norm_b)))


class SemanticProvenanceB8:
    """
    B8: Embedding-based provenance tracker.

    Strategy:
    - Encode all document spans as embeddings
    - For each argument value, compute max cosine similarity to all spans
    - If similarity > threshold: derive provenance from that span
    - If similarity in grey zone [threshold-0.2, threshold]: Confirm
    - If similarity < threshold-0.2: rag_inferred (no match)
    """

    def __init__(
        self,
        documents: list[RAGDocument],
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.60,
        confirm_zone: float = 0.15,
    ):
        self.documents = documents
        self.threshold = threshold
        self.confirm_zone = confirm_zone
        self.model = None
        self.doc_embeddings = []
        self._load_model(model_name)

    def _load_model(self, model_name: str):
        if not HAS_ST:
            logger.warning("sentence-transformers not installed. B8 unavailable.")
            return
        try:
            self.model = SentenceTransformer(model_name)
            # Encode all document contents
            texts = [d.content for d in self.documents]
            if texts:
                self.doc_embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.info(f"B8 model loaded: {model_name}, {len(texts)} docs encoded")
        except Exception as e:
            logger.warning(f"B8 model load error: {e}")
            self.model = None

    def track_argument(self, arg_name: str, arg_value: str) -> tuple[ProvenanceSource, bool]:
        """
        Returns (ProvenanceSource, needs_confirm).
        needs_confirm=True when in grey zone.
        """
        if self.model is None or len(self.doc_embeddings) == 0:
            return ProvenanceSource(origin="rag_inferred", confidence=0.0), False

        if not arg_value or not str(arg_value).strip():
            return ProvenanceSource(origin="user_intent", confidence=1.0), False

        # Encode argument value
        val_emb = self.model.encode([str(arg_value)], convert_to_numpy=True)[0]

        # Find max similarity across documents
        best_sim = 0.0
        best_idx = -1
        for i, doc_emb in enumerate(self.doc_embeddings):
            sim = cosine_similarity(val_emb, doc_emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        # Determine provenance
        needs_confirm = False
        if best_sim >= self.threshold:
            doc = self.documents[best_idx]
            origin = "injected_document" if doc.is_injected else "rag_document"
            return ProvenanceSource(
                origin=origin, confidence=best_sim, source_id=doc.doc_id
            ), False
        elif best_sim >= self.threshold - self.confirm_zone:
            # Grey zone: Confirm
            needs_confirm = True
            doc = self.documents[best_idx] if best_idx >= 0 else None
            origin = "rag_document" if doc and not doc.is_injected else "rag_inferred"
            return ProvenanceSource(origin=origin, confidence=best_sim), True
        else:
            return ProvenanceSource(origin="rag_inferred", confidence=best_sim), False


def evaluate_b8(
    benign_tasks: list[BenignTask],
    attacked_tasks: list[AttackedTask],
    documents: list[RAGDocument],
    thresholds: list[float] = None,
) -> list[dict]:
    """Evaluate B8 at multiple thresholds and return ASR-FBR curve."""
    if not HAS_ST:
        return [{"threshold": 0.6, "asr": -1, "fbr": -1, "error": "sentence-transformers not installed"}]

    if thresholds is None:
        thresholds = [0.30, 0.45, 0.60, 0.75, 0.90]

    engine = PolicyEngine.load_default_policies()
    results = []

    for threshold in thresholds:
        tracker = SemanticProvenanceB8(documents, threshold=threshold)
        asr_count = fbr_count = 0
        na, nb = len(attacked_tasks), len(benign_tasks)

        # Attacked
        for task in attacked_tasks:
            args = {**task.expected_arguments, **task.corrupted_arguments}
            all_blocked = False
            needs_confirm = False
            for arg_name, arg_val in args.items():
                if arg_val is None:
                    continue
                prov, confirm = tracker.track_argument(arg_name, str(arg_val))
                if prov.origin == "injected_document":
                    all_blocked = True
                    break
                if confirm:
                    needs_confirm = True

            if not all_blocked and not needs_confirm:
                asr_count += 1  # Passed through

        # Benign
        for task in benign_tasks:
            all_blocked = False
            needs_confirm = False
            for arg_name, arg_val in task.expected_arguments.items():
                if arg_val is None:
                    continue
                prov, confirm = tracker.track_argument(arg_name, str(arg_val))
                if prov.origin == "injected_document":
                    all_blocked = True
                    break
                if confirm:
                    needs_confirm = True

            if all_blocked:
                fbr_count += 1

        asr = round(asr_count / na * 100, 1) if na > 0 else 0
        fbr = round(fbr_count / nb * 100, 1) if nb > 0 else 0
        results.append({
            "baseline": "B8",
            "threshold": threshold,
            "asr": asr,
            "fbr": fbr,
            "asr_count": asr_count,
            "fbr_count": fbr_count,
            "na": na,
            "nb": nb,
        })
        logger.info(f"B8 threshold={threshold}: ASR={asr}% FBR={fbr}%")

    return results


if __name__ == "__main__":
    import csv
    import subprocess
    import sys

    # Install sentence-transformers if needed
    if not HAS_ST:
        print("Installing sentence-transformers...")
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "sentence-transformers", "--break-system-packages", "-q"])
        from sentence_transformers import SentenceTransformer

    # Load test data
    bench_dir = Path("data")
    benign = [BenignTask(**t)
              for t in json.loads((bench_dir / "tasks_benign.json").read_text())]
    attacked = [AttackedTask(**t)
                for t in json.loads((bench_dir / "tasks_attacked.json").read_text())]
    documents = [RAGDocument(**d)
                 for d in json.loads((bench_dir / "documents.json").read_text())]

    print("\n=== B8 Semantic Provenance — ASR-FBR Curve ===\n")
    print(f"{'Threshold':>10} {'ASR':>8} {'FBR':>8}")
    print("-" * 30)

    results = evaluate_b8(benign, attacked, documents)

    for r in results:
        if r.get("error"):
            print(f"  ERROR: {r['error']}")
        else:
            print(f"{r['threshold']:>10.2f} {r['asr']:>7.1f}% {r['fbr']:>7.1f}%")

    with open(RESULTS_DIR / "b8_asr_fbr_curve.csv", "w", newline="") as f:
        if results and not results[0].get("error"):
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    logger.info(f"B8 results saved to {RESULTS_DIR}/")