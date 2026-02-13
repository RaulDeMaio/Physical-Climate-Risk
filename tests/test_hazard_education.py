import pandas as pd

from app.education import (
    build_hazard_catalog,
    calculate_progress,
    compute_quiz_score,
    filter_intensity_levels,
    persist_education_progress,
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


def test_calculate_progress_clamps_between_zero_and_one():
    assert calculate_progress(5, 4) == 1.0
    assert calculate_progress(-1, 4) == 0.0


def test_persist_education_progress_deduplicates_steps():
    state = {}
    persist_education_progress(state, "hazard_education_steps", "opened_education")
    persist_education_progress(state, "hazard_education_steps", "opened_education")
    assert state["hazard_education_steps"] == ["opened_education"]
