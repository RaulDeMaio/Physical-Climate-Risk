from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class EducationTier:
    key: str
    title: str
    description_md: str


@dataclass(frozen=True)
class HazardEducationContent:
    intensity_tiers: Dict[str, EducationTier]
    hazard_type_explanations_md: Dict[str, str]
    calibration_explanation_md: str
    fallback_hazard_description_md: str


EDUCATION_CONTENT = HazardEducationContent(
    intensity_tiers={
        "moderate": EducationTier(
            key="moderate",
            title="Moderate",
            description_md="Frequent but contained disruption. Expect local production slowdowns with limited cross-border propagation.",
        ),
        "severe": EducationTier(
            key="severe",
            title="Severe",
            description_md="Material disruption with visible macro-sector impacts. Supply chain effects typically spread to major trade partners.",
        ),
        "extreme": EducationTier(
            key="extreme",
            title="Extreme",
            description_md="High-intensity disruption likely to trigger broad upstream/downstream bottlenecks and substantial output losses.",
        ),
        "very_extreme": EducationTier(
            key="very_extreme",
            title="Very Extreme",
            description_md="Tail-risk event calibrated from the highest historical loss percentiles; systemic impacts may dominate local impacts.",
        ),
    },
    hazard_type_explanations_md={
        "hydrological": "**Hydrological** hazards are driven by water-system extremes (e.g., river flooding) that can disrupt transport, assets, and production continuity.",
        "meteorological": "**Meteorological** hazards include short-duration atmospheric events (e.g., storms) that create direct and indirect supply-chain interruptions.",
        "climatological": "**Climatological** hazards are longer-duration climate anomalies (e.g., drought, wildfire-conducive conditions) that can reduce sectoral output over extended periods.",
    },
    calibration_explanation_md=(
        "Calibration maps **historical hazard losses** to country-level output and computes an intensity scalar (φ). "
        "Each tier (moderate → very_extreme) corresponds to progressively higher historical percentiles in the calibration table."
    ),
    fallback_hazard_description_md="No hazard-type education text is available yet for this hazard.",
)


def build_hazard_catalog(pct_table: Optional[pd.DataFrame]) -> List[str]:
    if pct_table is None or pct_table.empty or "hazard" not in pct_table:
        return []
    return sorted([h for h in pct_table["hazard"].dropna().unique() if str(h).strip()])


def filter_intensity_levels(available_levels: List[str]) -> List[str]:
    ordered = ["moderate", "severe", "extreme", "very_extreme"]
    return [level for level in ordered if level in available_levels]


def compute_quiz_score(correct_answers: int, total_questions: int) -> float:
    if total_questions <= 0:
        return 0.0
    return round((max(0, correct_answers) / total_questions) * 100.0, 2)


def get_hazard_type_explanation(selected_hazard: str) -> str:
    if not selected_hazard:
        return EDUCATION_CONTENT.fallback_hazard_description_md
    return EDUCATION_CONTENT.hazard_type_explanations_md.get(
        selected_hazard.strip().lower(),
        EDUCATION_CONTENT.fallback_hazard_description_md,
    )


def render_education_panel(selected_level: str, selected_hazard: str):
    with st.sidebar.expander("ℹ️ Hazard education", expanded=False):
        tier = EDUCATION_CONTENT.intensity_tiers.get(selected_level)
        if tier:
            st.markdown(f"**{tier.title} intensity**  ")
            st.markdown(tier.description_md)
        else:
            st.warning("Selected intensity definition is unavailable.")

        st.markdown("**Hazard type**")
        st.markdown(get_hazard_type_explanation(selected_hazard))

        st.markdown("**Calibration logic**")
        st.markdown(EDUCATION_CONTENT.calibration_explanation_md)
        st.caption(f"Selected hazard: `{selected_hazard}`")
