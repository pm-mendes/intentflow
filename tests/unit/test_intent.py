"""Tests unitaires pour l'extracteur d'intention."""

import pytest
from intentflow.intent.extractor import IntentExtractor
from intentflow.models.models import Intent


class TestIntentExtractor:
    """Tests sans API (mode heuristique uniquement)."""

    @pytest.fixture
    def extractor(self) -> IntentExtractor:
        return IntentExtractor(use_llm=False)

    def test_extract_send_email_heuristic(self, extractor: IntentExtractor) -> None:
        intent = extractor.extract("Send an email to alice", mode="heuristic")
        assert intent.action == "send_email"
        assert intent.extraction_mode == "heuristic"

    def test_extract_calendar_heuristic(self, extractor: IntentExtractor) -> None:
        intent = extractor.extract("Schedule a meeting with the team", mode="heuristic")
        assert intent.action == "create_calendar_event"

    def test_extract_oracle_mode(self, extractor: IntentExtractor) -> None:
        oracle = Intent(action="write_file", parameters={"path": "/reports/out.txt"})
        intent = extractor.extract("some message", mode="oracle", oracle_intent=oracle)
        assert intent.action == "write_file"
        assert intent.parameters["path"] == "/reports/out.txt"
        assert intent.extraction_mode == "oracle"

    def test_confidence_bounds(self, extractor: IntentExtractor) -> None:
        intent = extractor.extract("unknown request xyz", mode="heuristic")
        assert 0.0 <= intent.confidence <= 1.0

    def test_email_extraction_in_message(self, extractor: IntentExtractor) -> None:
        intent = extractor.extract("Send report to alice@company.com", mode="heuristic")
        assert "alice@company.com" in intent.parameters.get("recipient", "")

    def test_oracle_date_is_canonicalized_to_iso(self, extractor: IntentExtractor) -> None:
        """Phase 1: typed extraction canonicalizes a natural-language date to
        ISO-8601 so downstream provenance matching / field-F1 evaluation
        don't treat a format change as a disagreement."""
        oracle = Intent(action="create_calendar_event", parameters={"date": "August 15, 2026"})
        intent = extractor.extract("some message", mode="oracle", oracle_intent=oracle)
        assert intent.parameters["date"] == "2026-08-15"

    def test_oracle_unparseable_date_is_left_untouched(self, extractor: IntentExtractor) -> None:
        oracle = Intent(action="create_calendar_event", parameters={"date": "next sprint"})
        intent = extractor.extract("some message", mode="oracle", oracle_intent=oracle)
        assert intent.parameters["date"] == "next sprint"  # fail-closed: not guessed

    def test_oracle_participants_string_is_coerced_to_list(
        self, extractor: IntentExtractor
    ) -> None:
        oracle = Intent(
            action="create_calendar_event",
            parameters={"participants": "alice@company.com and bob@company.com"},
        )
        intent = extractor.extract("some message", mode="oracle", oracle_intent=oracle)
        assert intent.parameters["participants"] == ["alice@company.com", "bob@company.com"]

    def test_oracle_sql_selector_is_canonicalized(self, extractor: IntentExtractor) -> None:
        oracle = Intent(action="run_query", parameters={"query": "select   *   from reports;"})
        intent = extractor.extract("some message", mode="oracle", oracle_intent=oracle)
        assert intent.parameters["query"].upper().startswith("SELECT")
        assert "  " not in intent.parameters["query"]
