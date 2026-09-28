"""Tests unitaires pour la comparaison sémantique par champ (Phase 1)."""

from intentflow.metrics.field_comparison import (
    dates_equal,
    participants_equal,
    sql_equal,
    values_equal,
)


class TestDatesEqual:
    def test_iso_vs_natural_language(self) -> None:
        assert dates_equal("2026-08-15", "August 15, 2026")

    def test_iso_vs_slash_format(self) -> None:
        assert dates_equal("2026-08-15", "15/08/2026")

    def test_different_dates_do_not_match(self) -> None:
        assert not dates_equal("2026-08-15", "2026-08-16")

    def test_unparseable_value_fails_closed(self) -> None:
        assert not dates_equal("2026-08-15", "next Monday-ish whenever")

    def test_none_never_matches(self) -> None:
        assert not dates_equal(None, "2026-08-15")


class TestParticipantsEqual:
    def test_same_set_different_order(self) -> None:
        assert participants_equal(
            ["alice@company.com", "bob@company.com"],
            ["bob@company.com", "alice@company.com"],
        )

    def test_string_vs_list(self) -> None:
        assert participants_equal(
            ["alice@company.com", "bob@company.com"],
            "alice@company.com and bob@company.com",
        )

    def test_partial_coverage_is_not_equal(self) -> None:
        assert not participants_equal(
            ["alice@company.com", "bob@company.com"], ["alice@company.com"]
        )

    def test_extra_participant_is_not_equal(self) -> None:
        assert not participants_equal(
            ["alice@company.com"], ["alice@company.com", "bob@company.com"]
        )


class TestSqlEqual:
    def test_case_and_whitespace_insensitive(self) -> None:
        assert sql_equal(
            "SELECT * FROM reports WHERE month='2026-07'",
            "select   *   from reports where month = '2026-07'",
        )

    def test_different_predicate_is_not_equal(self) -> None:
        assert not sql_equal(
            "SELECT * FROM reports WHERE month='2026-07'",
            "SELECT * FROM reports WHERE month='2026-08'",
        )

    def test_trailing_semicolon_ignored(self) -> None:
        assert sql_equal("SELECT * FROM x", "SELECT * FROM x;")


class TestValuesEqualDispatch:
    def test_strict_field_is_case_insensitive_exact_match(self) -> None:
        assert values_equal("recipient", "Alice@Company.com", "alice@company.com")

    def test_strict_field_rejects_different_value(self) -> None:
        assert not values_equal("recipient", "alice@company.com", "bob@company.com")

    def test_date_field_uses_semantic_comparison(self) -> None:
        assert values_equal("date", "2026-08-15", "August 15, 2026")

    def test_selector_field_uses_sql_comparison(self) -> None:
        assert values_equal("selector", "SELECT * FROM x", "select * from x")
