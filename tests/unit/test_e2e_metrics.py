"""Tests unitaires pour PMCR/CME/UER (Phase 3, point 2.2)."""

from intentflow.metrics.e2e_metrics import compute_e2e_metrics


def _attacked(corrupted: bool, blocked: bool) -> dict:
    return {
        "is_attacked": True,
        "planner_failed": False,
        "argument_corrupted": corrupted,
        "mediation_blocked": blocked,
    }


def _benign(tool_correct: bool, blocked: bool) -> dict:
    return {
        "is_attacked": False,
        "planner_failed": False,
        "tool_correct": tool_correct,
        "mediation_blocked": blocked,
    }


class TestComputeE2EMetrics:
    def test_low_exposure_does_not_inflate_mediator_credit(self) -> None:
        """The exact failure mode this module exists to prevent: with only
        1/10 attacks reaching the mediator corrupted, UER looks great (10%)
        but that must not be read as "the mediator is 90% effective" — CME
        (the mediator's own number) must be computed on the exposed subset
        only.
        """
        decisions = [_attacked(corrupted=False, blocked=False) for _ in range(9)]
        decisions.append(_attacked(corrupted=True, blocked=True))  # the one exposed case, blocked
        m = compute_e2e_metrics(decisions)
        assert m["na"] == 10
        assert m["pmcr_n"] == 1
        assert m["pmcr"] == 10.0
        assert m["cme_n"] == 1
        assert m["cme_denominator"] == 1
        assert m["cme"] == 100.0  # mediator caught 100% of what it actually saw
        assert m["uer_n"] == 0
        assert m["uer"] == 0.0

    def test_cme_is_none_when_nothing_reaches_the_mediator(self) -> None:
        decisions = [_attacked(corrupted=False, blocked=False) for _ in range(5)]
        m = compute_e2e_metrics(decisions)
        assert m["pmcr_n"] == 0
        assert m["cme"] is None
        assert m["uer"] == 0.0

    def test_uer_counts_only_corrupted_and_unblocked(self) -> None:
        decisions = [
            _attacked(corrupted=True, blocked=True),  # caught
            _attacked(corrupted=True, blocked=False),  # slipped through
            _attacked(corrupted=False, blocked=False),  # planner never corrupted it
        ]
        m = compute_e2e_metrics(decisions)
        assert m["na"] == 3
        assert m["pmcr_n"] == 2
        assert m["uer_n"] == 1
        assert m["cme_n"] == 1
        assert m["cme"] == 50.0

    def test_planner_failed_tasks_are_excluded_from_denominators(self) -> None:
        decisions = [
            _attacked(corrupted=True, blocked=True),
            {"is_attacked": True, "planner_failed": True},
        ]
        m = compute_e2e_metrics(decisions)
        assert m["na"] == 1  # the failed one doesn't count

    def test_benign_side_btsr_fbr(self) -> None:
        decisions = [
            _benign(tool_correct=True, blocked=False),
            _benign(tool_correct=True, blocked=False),
            _benign(tool_correct=True, blocked=True),  # false block
        ]
        m = compute_e2e_metrics(decisions)
        assert m["nb"] == 3
        assert m["btsr_n"] == 2
        assert m["fbr_n"] == 1

    def test_empty_decisions_do_not_crash(self) -> None:
        m = compute_e2e_metrics([])
        assert m["na"] == 0
        assert m["nb"] == 0
        assert m["pmcr"] == 0.0
        assert m["cme"] is None
        assert m["uer"] == 0.0
