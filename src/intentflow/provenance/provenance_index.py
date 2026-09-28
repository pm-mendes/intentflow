"""
intentflow/provenance/provenance_index.py

Inverted index for O(1) provenance tracking — Point 21.
Replaces naive O(n) substring search.

Uses Aho-Corasick automaton (ahocorasick library) for multi-pattern matching.
Falls back to dict-based index if ahocorasick not available.
"""

import time
import hashlib
import json
from collections import defaultdict
from intentflow.models.models import RAGDocument
from intentflow.normalization import normalize_unicode
from intentflow.utils.logger import logger

try:
    import ahocorasick

    HAS_AC = True
except ImportError:
    HAS_AC = False

HASH_ALGO = "sha256"
MIN_TOKEN_LEN = 4  # Ignore very short tokens


def hash_arguments(args: dict) -> str:
    """
    Tamper-evident hash of a tool call's arguments — canonical(tool, args) in
    Algorithm 1's terms. Standalone function (no index state needed) so
    `pipeline/mediator.py` can call it directly for TOCTOU re-verification
    without constructing a `ProvenanceIndex`. `ProvenanceIndex.get_argument_hash`
    delegates here for backward compatibility.
    """
    canonical = json.dumps(args, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def normalize_value(value: str) -> str:
    """
    Normalize a value before indexing/lookup.

    Delegates to `intentflow.normalization.normalize_unicode` (NFKC +
    zero-width removal + homoglyph mapping + lowercase) instead of a
    second, slightly weaker NFKC+lowercase-only implementation that used
    to live here — the two had drifted (this one didn't map homoglyphs,
    so an Aho-Corasick lookup could miss a Cyrillic-lookalike value that
    `normalize_unicode` would have caught). Single source of truth now.
    """
    if not value:
        return ""
    return normalize_unicode(value)


def tokenize_content(content: str, min_len: int = MIN_TOKEN_LEN) -> list[str]:
    """
    Extract meaningful tokens from document content.
    Includes: emails, URLs, paths, words >= min_len.
    """
    import re

    tokens = set()
    # Full content normalized
    tokens.add(normalize_value(content))
    # Email addresses
    for match in re.finditer(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", content):
        tokens.add(normalize_value(match.group()))
    # URLs
    for match in re.finditer(r'https?://[^\s\'"]+', content):
        tokens.add(normalize_value(match.group()))
    # File paths
    for match in re.finditer(r"/[a-zA-Z0-9._/-]{3,}", content):
        tokens.add(normalize_value(match.group()))
    # Words >= min_len
    for word in re.split(r"\s+", content):
        word_clean = normalize_value(word.strip(".,;:!?()[]{}"))
        if len(word_clean) >= min_len:
            tokens.add(word_clean)
    return [t for t in tokens if t]


class ProvenanceIndex:
    """
    Inverted index for fast argument-to-document provenance tracking.

    build(): O(sum(len(doc.content))) — one-time cost
    lookup(): O(len(value)) — amortized O(1)
    """

    def __init__(self):
        self._index: dict[str, list[str]] = defaultdict(list)  # token -> [doc_ids]
        self._docs: dict[str, RAGDocument] = {}
        self._ac_automaton = None
        self._doc_hashes: dict[str, str] = {}

    def build(self, documents: list[RAGDocument]) -> None:
        """Build the index from a document corpus."""
        t0 = time.perf_counter()
        self._index.clear()
        self._docs.clear()

        for doc in documents:
            self._docs[doc.doc_id] = doc
            self._doc_hashes[doc.doc_id] = hashlib.sha256(doc.content.encode()).hexdigest()[:16]
            for token in tokenize_content(doc.content):
                self._index[token].append(doc.doc_id)

        # Build Aho-Corasick automaton for multi-pattern matching
        if HAS_AC and self._index:
            self._ac_automaton = ahocorasick.Automaton()
            for i, token in enumerate(self._index.keys()):
                if len(token) >= MIN_TOKEN_LEN:
                    self._ac_automaton.add_word(token, (i, token))
            self._ac_automaton.make_automaton()

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            f"Index built: {len(documents)} docs, " f"{len(self._index)} tokens, {elapsed:.1f}ms"
        )

    def lookup(self, value: str) -> list[RAGDocument]:
        """
        Find all documents containing the given value.
        Returns list of matching documents.
        """
        if not value:
            return []

        norm_val = normalize_value(str(value))
        matching_doc_ids = set()

        if HAS_AC and self._ac_automaton is not None:
            # Aho-Corasick: scan value for any indexed token
            try:
                for _, (_, token) in self._ac_automaton.iter(norm_val):
                    matching_doc_ids.update(self._index.get(token, []))
                # Also scan the full value as a single token
                if norm_val in self._index:
                    matching_doc_ids.update(self._index[norm_val])
            except Exception:
                # Fallback to dict lookup
                if norm_val in self._index:
                    matching_doc_ids.update(self._index[norm_val])
        else:
            # Dict-based fallback
            if norm_val in self._index:
                matching_doc_ids.update(self._index[norm_val])
            # Check if any token is a substring of value
            for token, doc_ids in self._index.items():
                if len(token) >= MIN_TOKEN_LEN and token in norm_val:
                    matching_doc_ids.update(doc_ids)

        return [self._docs[doc_id] for doc_id in matching_doc_ids if doc_id in self._docs]

    def get_argument_hash(self, args: dict) -> str:
        """
        Compute a tamper-evident hash of validated arguments.
        Used for TOCTOU protection: hash before mediation,
        verify before execution.
        """
        return hash_arguments(args)

    @property
    def n_tokens(self) -> int:
        return len(self._index)

    @property
    def n_docs(self) -> int:
        return len(self._docs)


def benchmark_index(n_docs_list: list[int]) -> list[dict]:
    """Benchmark index lookup vs naive O(n) at different corpus sizes."""
    import random
    import re

    rng = random.Random(42)
    results = []

    for n_docs in n_docs_list:
        # Generate synthetic documents
        docs = []
        emails = [f"user{i}@company.com" for i in range(n_docs)]
        for i in range(n_docs):
            content = (
                f"Document {i}. Contact: {emails[i]}. "
                f"Path: /reports/doc_{i}.json. "
                f"API: https://internal-api.company.com/db. "
                f"{'Lorem ipsum dolor sit amet ' * 10}"
            )
            docs.append(
                RAGDocument(
                    doc_id=f"doc_{i}",
                    content=content,
                    is_injected=(i % 10 == 0),
                    source_domain="trusted_internal",
                )
            )

        # Build index
        idx = ProvenanceIndex()
        t_build = time.perf_counter()
        idx.build(docs)
        build_ms = (time.perf_counter() - t_build) * 1000

        # Benchmark lookup (50 queries)
        test_values = [emails[rng.randint(0, n_docs - 1)] for _ in range(50)]
        lookup_times = []
        for val in test_values:
            t = time.perf_counter()
            idx.lookup(val)
            lookup_times.append((time.perf_counter() - t) * 1000)

        lookup_times.sort()
        p50 = lookup_times[len(lookup_times) // 2]
        p95 = lookup_times[int(len(lookup_times) * 0.95)]

        # Naive O(n) baseline
        naive_times = []
        for val in test_values:
            t = time.perf_counter()
            norm_val = normalize_value(val)
            _ = [d for d in docs if norm_val in normalize_value(d.content)]
            naive_times.append((time.perf_counter() - t) * 1000)
        naive_times.sort()
        naive_p50 = naive_times[len(naive_times) // 2]
        naive_p95 = naive_times[int(len(naive_times) * 0.95)]

        speedup = naive_p50 / p50 if p50 > 0 else 999

        results.append(
            {
                "n_docs": n_docs,
                "build_ms": round(build_ms, 1),
                "n_tokens": idx.n_tokens,
                "index_p50_ms": round(p50, 2),
                "index_p95_ms": round(p95, 2),
                "naive_p50_ms": round(naive_p50, 2),
                "naive_p95_ms": round(naive_p95, 2),
                "speedup_p50": round(speedup, 1),
                "under_100ms": p50 < 100,
            }
        )

    return results


if __name__ == "__main__":
    import csv
    from pathlib import Path

    # Install ahocorasick if not available
    if not HAS_AC:
        import subprocess, sys

        print("Installing pyahocorasick...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "pyahocorasick",
                "--break-system-packages",
                "-q",
            ]
        )
        try:
            import ahocorasick

            print("pyahocorasick installed OK")
        except ImportError:
            print("Using dict-based fallback (no speedup)")

    results_dir = Path("data/results/provenance_index")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Provenance Index Benchmark ===")
    print(f"Mode: {'Aho-Corasick' if HAS_AC else 'dict-based fallback'}")
    print(
        f"\n{'N docs':>8} {'Index p50':>12} {'Index p95':>12} "
        f"{'Naive p50':>12} {'Speedup':>9} {'<100ms':>8}"
    )
    print("-" * 70)

    bench = benchmark_index([20, 200, 2000, 10000])
    for r in bench:
        flag = "YES" if r["under_100ms"] else "NO"
        print(
            f"{r['n_docs']:>8} {r['index_p50_ms']:>11.2f}ms {r['index_p95_ms']:>11.2f}ms "
            f"{r['naive_p50_ms']:>11.2f}ms {r['speedup_p50']:>8.1f}x {flag:>8}"
        )

    with open(results_dir / "index_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bench[0].keys())
        writer.writeheader()
        writer.writerows(bench)

    logger.info(f"Benchmark saved to {results_dir}/")
