"""Numeric grounding — spec §5.1, the check that catches F1."""

import pytest

from moc.evals.deterministic import check_numeric_grounding


def test_flags_orphan_figure():
    result = check_numeric_grounding(
        reply="الرسوم ١٢٥٠ جنيه للساعة المعتمدة",
        retrieved_passages=["رسوم الساعة المعتمدة لكلية الهندسة 1400 جنيه"],
        script_constants=[],
    )
    assert result.passed is False
    assert 1250 in result.orphan_numbers


def test_passes_when_figure_is_grounded():
    result = check_numeric_grounding(
        reply="الرسوم 1400 جنيه للساعة",
        retrieved_passages=["رسوم الساعة المعتمدة 1400 جنيه"],
        script_constants=[],
    )
    assert result.passed is True
    assert result.orphan_numbers == []


def test_matches_across_scripts():
    """Reply in Arabic-Indic, source in Latin digits — must still match."""
    result = check_numeric_grounding(
        reply="الرسوم ١٤٠٠ جنيه",
        retrieved_passages=["fee is 1400 EGP"],
        script_constants=[],
    )
    assert result.passed is True


def test_rounded_figure_is_a_failure():
    """A plausible-looking rounded number is still a hallucination."""
    result = check_numeric_grounding(
        reply="حوالي 1500 جنيه",
        retrieved_passages=["1400 جنيه"],
        script_constants=[],
    )
    assert result.passed is False
    assert 1500 in result.orphan_numbers


def test_approximation_marker_trips_the_check_even_when_the_figure_matches():
    """§19.3: markers are configurable, that an approximation fails is not.

    This is the case the orphan check alone cannot catch — the figure is
    correct, but hedging it invites the customer to treat it as negotiable.
    """
    result = check_numeric_grounding(
        reply="حوالي 1400 جنيه",
        retrieved_passages=["1400 جنيه"],
        script_constants=[],
    )
    assert result.passed is False
    assert result.orphan_numbers == []
    assert result.approximations != []


def test_script_constant_grounds_a_figure():
    result = check_numeric_grounding(
        reply="المقدم ١٠٪",
        retrieved_passages=[],
        script_constants=[10],
    )
    assert result.passed is True


def test_reply_without_figures_passes():
    result = check_numeric_grounding(
        reply="ممكن تحددلي الكلية؟",
        retrieved_passages=[],
        script_constants=[],
    )
    assert result.passed is True
    assert result.orphan_numbers == []


def test_year_in_reply_is_not_treated_as_a_figure():
    """The academic year is a qualifier, not a claim needing a source figure."""
    result = check_numeric_grounding(
        reply="رسوم 1400 جنيه لعام 2026",
        retrieved_passages=["1400 جنيه"],
        script_constants=[],
    )
    assert result.passed is True


def test_unit_scaled_figure_matches_a_plain_source_figure():
    """Reply says ٦ مليون; the inventory row says 6000000."""
    result = check_numeric_grounding(
        reply="سعر الوحدة ٦ مليون",
        retrieved_passages=["price 6000000 EGP"],
        script_constants=[],
    )
    assert result.passed is True


def test_reports_every_orphan_not_just_the_first():
    result = check_numeric_grounding(
        reply="الرسوم 1250 جنيه و المقدم 300 جنيه",
        retrieved_passages=["1400 جنيه"],
        script_constants=[],
    )
    assert sorted(result.orphan_numbers) == [300, 1250]


@pytest.mark.parametrize("constants", [[1400], ["1400"], [1400.0]])
def test_script_constants_accept_numbers_or_strings(constants):
    result = check_numeric_grounding(
        reply="الرسوم ١٤٠٠ جنيه", retrieved_passages=[], script_constants=constants
    )
    assert result.passed is True
