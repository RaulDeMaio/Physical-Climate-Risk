import pandas as pd

from app.education import (
    build_hazard_catalog,
    compute_quiz_score,
    filter_intensity_levels,
    get_hazard_type_explanation,
)


def test_build_hazard_catalog_returns_sorted_unique_values():
    df = pd.DataFrame({"hazard": ["flood", "storm", "flood", None, " "]})
    assert build_hazard_catalog(df) == ["flood", "storm"]


def test_filter_intensity_levels_keeps_expected_order():
    levels = ["extreme", "moderate", "very_extreme"]
    assert filter_intensity_levels(levels) == ["moderate", "extreme", "very_extreme"]


def test_compute_quiz_score_handles_zero_questions():
    assert compute_quiz_score(3, 0) == 0.0


def test_compute_quiz_score_bounds_negative_correct_answers():
    assert compute_quiz_score(-2, 4) == 0.0


def test_get_hazard_type_explanation_returns_specific_text_for_known_hazard():
    explanation = get_hazard_type_explanation("hydrological")
    assert "Hydrological" in explanation


def test_get_hazard_type_explanation_falls_back_for_unknown_hazard():
    explanation = get_hazard_type_explanation("volcanic")
    assert "No hazard-type education text" in explanation
