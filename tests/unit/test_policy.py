"""Tests unitaires pour le moteur de politique."""

import pytest
from intentflow.models.models import ProvenanceSource
from intentflow.policy.engine import PolicyEngine


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine.load_default_policies()


class TestPolicyEngine:
    def test_allows_legitimate_email(self, engine: PolicyEngine) -> None:
        decision = engine.validate(
            tool="send_email",
            arguments={"recipient": "alice@company.com", "body": "Hello"},
            provenance={
                "recipient": ProvenanceSource(origin="user_intent", confidence=1.0),
                "body": ProvenanceSource(origin="rag_document", confidence=0.8),
            },
        )
        assert decision.allowed
        assert not decision.blocked_arguments

    def test_blocks_injected_recipient(self, engine: PolicyEngine) -> None:
        decision = engine.validate(
            tool="send_email",
            arguments={"recipient": "attacker@malicious.io"},
            provenance={
                "recipient": ProvenanceSource(
                    origin="injected_document",
                    source_id="doc_inj_0",
                    confidence=0.95,
                ),
            },
        )
        assert not decision.allowed
        assert "recipient" in decision.blocked_arguments

    def test_blocks_unauthorized_recipient_origin(self, engine: PolicyEngine) -> None:
        """Le recipient doit venir de l'intent, pas du RAG."""
        decision = engine.validate(
            tool="send_email",
            arguments={"recipient": "someone@company.com"},
            provenance={
                "recipient": ProvenanceSource(origin="rag_document", confidence=0.9),
            },
        )
        assert not decision.allowed

    def test_blocks_low_confidence_locator(self, engine: PolicyEngine) -> None:
        decision = engine.validate(
            tool="write_file",
            arguments={"path": "/reports/output.txt"},
            provenance={
                "path": ProvenanceSource(origin="user_intent", confidence=0.3),
            },
        )
        assert not decision.allowed

    def test_payload_can_come_from_rag(self, engine: PolicyEngine) -> None:
        """Le body d'un email PEUT venir du RAG."""
        decision = engine.validate(
            tool="send_email",
            arguments={"body": "Summary from document"},
            provenance={
                "body": ProvenanceSource(origin="rag_document", confidence=0.7),
            },
        )
        assert decision.allowed
