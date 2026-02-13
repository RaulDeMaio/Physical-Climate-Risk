from __future__ import annotations

from typing import Mapping

import pandas as pd
import streamlit as st

LIGHT_TABLE_FALLBACK = {
    "header_bg": "var(--primary-color)",
    "header_text": "var(--background-color)",
    "row_odd_bg": "var(--secondary-background-color)",
    "row_even_bg": "var(--background-color)",
    "border": "var(--secondary-background-color)",
}

DARK_TABLE_FALLBACK = {
    "header_bg": "var(--secondary-background-color)",
    "header_text": "var(--text-color)",
    "row_odd_bg": "var(--background-color)",
    "row_even_bg": "var(--secondary-background-color)",
    "border": "var(--secondary-background-color)",
}


def resolve_table_theme(tokens: Mapping[str, object] | None, mode: str = "light") -> dict:
    """Resolve table theme tokens with light/dark fallbacks."""
    fallback = DARK_TABLE_FALLBACK if mode == "dark" else LIGHT_TABLE_FALLBACK
    table_tokens = (tokens or {}).get("tables", {}) if isinstance(tokens, Mapping) else {}
    return {key: table_tokens.get(key, fallback[key]) for key in fallback}


def build_alignment_map(df: pd.DataFrame) -> dict[str, str]:
    """Numeric columns right-aligned, textual columns left-aligned."""
    alignments: dict[str, str] = {}
    for col in df.columns:
        alignments[col] = "right" if pd.api.types.is_numeric_dtype(df[col]) else "left"
    return alignments


def build_global_table_css(theme: Mapping[str, str]) -> str:
    """Create CSS for branded Streamlit dataframe rendering."""
    return f"""
    <style>
      [data-testid="stDataFrame"] thead tr th {{
          background: {theme['header_bg']} !important;
          color: {theme['header_text']} !important;
          border: 1px solid {theme['border']} !important;
      }}
      [data-testid="stDataFrame"] tbody tr:nth-child(odd) td {{
          background: {theme['row_odd_bg']} !important;
      }}
      [data-testid="stDataFrame"] tbody tr:nth-child(even) td {{
          background: {theme['row_even_bg']} !important;
      }}
      [data-testid="stDataFrame"] tbody td {{
          border: 1px solid {theme['border']} !important;
      }}
      [data-testid="stDataFrame"] tbody td.key-metric {{
          font-weight: 700 !important;
      }}
    </style>
    """


def build_branded_styler(
    df: pd.DataFrame,
    key_metric_columns: list[str] | None = None,
) -> pd.io.formats.style.Styler:
    """Build a styler with alignment and key metric emphasis, without mutating values."""
    key_metric_columns = key_metric_columns or []
    alignments = build_alignment_map(df)

    styler = df.style
    for col, alignment in alignments.items():
        styler = styler.set_properties(subset=[col], **{"text-align": alignment})

    if key_metric_columns:
        existing = [col for col in key_metric_columns if col in df.columns]
        if existing:
            styler = styler.set_properties(subset=existing, **{"font-weight": "700"})

    return styler


def apply_table_branding(tokens: Mapping[str, object] | None, mode: str = "light") -> None:
    """Inject global CSS so branded table styles are consistent across views."""
    theme = resolve_table_theme(tokens, mode=mode)
    st.markdown(build_global_table_css(theme), unsafe_allow_html=True)
