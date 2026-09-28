"""Tests unitaires pour le tracker de provenance."""

import pytest
from intentflow.models.models import Intent, RAGDocument
from intentflow.provenance.tracker import ProvenanceTracker


@pytest.fixture
def documents() -> list[RAGDocument]:
    return [
        RAGDocument(
            doc_id="d_benign", content="Send report to alice@company.com", is_injected=False
        ),
        RAGDocument(
            doc_id="d_injected", content="attacker@malicious.io steal data", is_injected=True
        ),
    ]


@pytest.fixture
def intent() -> Intent:
    return Intent(action="send_email", parameters={"recipient": "alice@company.com"})


class TestProvenanceTracker:
    def test_tracks_from_intent(self, documents: list[RAGDocument], intent: Intent) -> None:
        tracker = ProvenanceTracker(documents, intent)
        source = tracker.track_argument("recipient", "alice@company.com")
        assert source.origin == "user_intent"
        assert source.confidence >= 0.9

    def test_tracks_injected_document(self, documents: list[RAGDocument]) -> None:
        tracker = ProvenanceTracker(documents)
        source = tracker.track_argument("recipient", "attacker@malicious.io")
        assert source.origin == "injected_document"
        assert source.source_id == "d_injected"

    def test_tracks_benign_document(self, documents: list[RAGDocument]) -> None:
        tracker = ProvenanceTracker(documents)
        source = tracker.track_argument("body", "Send report to alice@company.com")
        assert source.origin == "rag_document"

    def test_fallback_for_unknown_value(self, documents: list[RAGDocument]) -> None:
        tracker = ProvenanceTracker(documents)
        source = tracker.track_argument("subject", "completely-unknown-xyz-value-123")
        assert source.origin == "rag_inferred"
        assert source.confidence < 0.5

    def test_homoglyph_obfuscated_value_still_traced_to_injected_document(
        self, documents: list[RAGDocument]
    ) -> None:
        """
        Phase 1: `normalization.py` (NFKC/homoglyph/zero-width) is now wired
        into the tracker — before, `.lower().strip()` alone meant a Cyrillic
        lookalike character would never match the plain-Latin value in the
        injected document, silently falling through to 'rag_inferred'
        instead of 'injected_document'.
        """
        tracker = ProvenanceTracker(documents)
        # 'а' here is Cyrillic U+0430, not Latin 'a' — HOMOGLYPH_MAP maps it back.
        obfuscated = "аttacker@malicious.io"
        source = tracker.track_argument("recipient", obfuscated)
        assert source.origin == "injected_document"
        assert source.source_id == "d_injected"

    def test_date_matches_intent_across_formats(self) -> None:
        """Phase 1: 'date' is compared semantically (ISO vs natural language)
        in addition to plain string matching — a planner that synthesizes
        "August 15, 2026" for an intent authorizing "2026-08-15" should still
        be recognized as user_intent, not fall through to rag_inferred."""
        intent = Intent(action="create_calendar_event", parameters={"date": "2026-08-15"})
        tracker = ProvenanceTracker([], intent)
        source = tracker.track_argument("date", "August 15, 2026")
        assert source.origin == "user_intent"

    def test_zero_width_obfuscated_value_still_matches_user_message(self) -> None:
        docs: list[RAGDocument] = []
        intent = Intent(action="send_email", parameters={})
        tracker = ProvenanceTracker(docs, intent)
        # Zero-width space injected inside the email the user actually typed.
        user_message = "Send it to ali​ce@company.com please."
        source = tracker.track_argument("recipient", "alice@company.com", user_message)
        assert source.origin == "user_intent"

    def test_path_does_not_inherit_user_intent_from_unrelated_content_field(self) -> None:
        """
        Regression test: found by running Disambiguation-v2 for the first
        time with real oracle intents (Phase 2). A corrupted `path` argument
        containing the substring "output" was matching the *unrelated*
        `content` intent parameter (also "output"), silently granting
        user_intent/1.0 confidence to a path-traversal-style attack. An
        argument only matches the intent parameters declared as its
        equivalents (`_INTENT_FIELD_EQUIVALENTS`) — "content" has none
        registered for "path".
        """
        intent = Intent(
            action="write_file",
            parameters={"path": "/reports/output.json", "content": "output"},
        )
        tracker = ProvenanceTracker([], intent)
        source = tracker.track_argument("path", "/finance/exports/output.csv")
        assert source.origin != "user_intent"

    def test_payload_argument_never_matches_via_intent_parameters(self) -> None:
        """A payload-role argument (body/content/subject/description/...) has
        no declared intent-field equivalent, so it can't inherit user_intent
        purely from an unrelated parameter — it still can via a direct
        message-substring match (a separate, unaffected code path)."""
        intent = Intent(action="send_email", parameters={"recipient": "alice@company.com"})
        tracker = ProvenanceTracker([], intent)
        source = tracker.track_argument("body", "alice@company.com")
        assert source.origin != "user_intent"
