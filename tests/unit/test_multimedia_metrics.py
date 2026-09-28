"""Tests unitaires pour DER/AHR@k/CASR (Phase 4)."""

from intentflow.metrics.multimedia_metrics import (
    adversarial_value_present,
    character_similarity,
    compute_ahr_at_k,
    compute_casr,
    compute_der,
)


class TestCharacterSimilarity:
    def test_identical_text_is_100(self) -> None:
        assert character_similarity("hello world", "hello world") == 100.0

    def test_completely_different_text_is_low(self) -> None:
        assert character_similarity("hello world", "xyz123") < 30.0

    def test_case_and_whitespace_insensitive(self) -> None:
        assert character_similarity("Hello World", "hello   world") > 95.0

    def test_both_empty_is_100(self) -> None:
        assert character_similarity("", "") == 100.0

    def test_minor_ocr_noise_still_scores_high(self) -> None:
        # Typical single-character OCR substitution (l -> 1, O -> 0).
        assert character_similarity("Send to alice@company.com", "Send t0 a1ice@company.com") > 85.0


class TestComputeDER:
    def test_averages_across_documents(self) -> None:
        pairs = [("hello", "hello"), ("world", "xxxxx")]
        result = compute_der(pairs)
        assert result["n"] == 2
        assert result["der_min"] < result["der_mean"] < 100.0

    def test_empty_pairs_do_not_crash(self) -> None:
        result = compute_der([])
        assert result["n"] == 0
        assert result["der_mean"] == 0.0


class TestAdversarialValuePresent:
    def test_exact_match(self) -> None:
        assert adversarial_value_present("attacker@evil.io", "Please contact attacker@evil.io now.")

    def test_absent_value(self) -> None:
        assert not adversarial_value_present("attacker@evil.io", "Please contact the team.")

    def test_minor_ocr_typo_still_detected(self) -> None:
        # A single-character OCR misread of the adversarial email.
        assert adversarial_value_present("attacker@evil.io", "contact attacker@evil.i0 immediately")

    def test_empty_value_never_matches(self) -> None:
        assert not adversarial_value_present("", "any text at all")


class TestComputeAhrAtK:
    def test_rate_computed_correctly(self) -> None:
        result = compute_ahr_at_k([True, True, False, True])
        assert result["n"] == 4
        assert result["n_hit"] == 3
        assert result["ahr"] == 75.0

    def test_empty_hits(self) -> None:
        result = compute_ahr_at_k([])
        assert result["ahr"] == 0.0


class TestComputeCasr:
    def test_matches_pmcr_semantics(self) -> None:
        decisions = [
            {"is_attacked": True, "planner_failed": False, "argument_corrupted": True},
            {"is_attacked": True, "planner_failed": False, "argument_corrupted": False},
            {
                "is_attacked": False,
                "planner_failed": False,
                "argument_corrupted": True,
            },  # ignored: benign
            {
                "is_attacked": True,
                "planner_failed": True,
                "argument_corrupted": True,
            },  # ignored: planner failed
        ]
        result = compute_casr(decisions)
        assert result["n"] == 2
        assert result["n_corrupted"] == 1
        assert result["casr"] == 50.0
