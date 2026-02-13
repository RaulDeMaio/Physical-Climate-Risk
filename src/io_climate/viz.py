# src/io_climate/viz.py
"""
Visualization helpers for the Physical Climate Risk Propagation Model.

This module standardizes Plotly visual outputs (maps, bar charts, linkage charts)
so that notebooks and future webapps (e.g., Streamlit) can reuse the same code.

Key features
------------
- Country codes (ISO-2) can be displayed as full country names in charts.
- Choropleth map shows only countries present in the result data.
- Long sector names are wrapped for readable horizontal bar charts.
- Linkage diagnostics display both absolute ΔA (percentage points) and relative change,
  while ranking continues to rely on absolute ΔA.

Assumptions
-----------
- df_country contains ISO-2 codes in a column named 'country'.
- df_sector contains a human-readable column 'sector_name' (fallback to 'sector').
- Linkage tables contain 'delta' (absolute) and optionally 'delta_rel' (relative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.io_climate.config import labels
from src.io_climate.supply_chain_heatmap import build_heatmap_frame

# ---------------------------------------------------------------------
# Country name decoding (ISO-2 -> ISO-3 + full name)
# ---------------------------------------------------------------------

# Minimal, robust EU/EEA-centric mapping (extend as needed).
# ISO-2 -> (ISO-3, Country Name)
ISO2_TO_ISO3_NAME: Dict[str, tuple[str, str]] = {
    "AT": ("AUT", "Austria"),
    "BE": ("BEL", "Belgium"),
    "BG": ("BGR", "Bulgaria"),
    "HR": ("HRV", "Croatia"),
    "CY": ("CYP", "Cyprus"),
    "CZ": ("CZE", "Czechia"),
    "DK": ("DNK", "Denmark"),
    "EE": ("EST", "Estonia"),
    "FI": ("FIN", "Finland"),
    "FR": ("FRA", "France"),
    "DE": ("DEU", "Germany"),
    "GR": ("GRC", "Greece"),
    "HU": ("HUN", "Hungary"),
    "IE": ("IRL", "Ireland"),
    "IT": ("ITA", "Italy"),
    "LV": ("LVA", "Latvia"),
    "LT": ("LTU", "Lithuania"),
    "LU": ("LUX", "Luxembourg"),
    "MT": ("MLT", "Malta"),
    "NL": ("NLD", "Netherlands"),
    "PL": ("POL", "Poland"),
    "PT": ("PRT", "Portugal"),
    "RO": ("ROU", "Romania"),
    "SK": ("SVK", "Slovakia"),
    "SI": ("SVN", "Slovenia"),
    "ES": ("ESP", "Spain"),
    "SE": ("SWE", "Sweden"),
    # Common extras in SAM / world rest:
    "GB": ("GBR", "United Kingdom"),
    "UK": ("GBR", "United Kingdom"),
    "NO": ("NOR", "Norway"),
    "CH": ("CHE", "Switzerland"),
    "IS": ("ISL", "Iceland"),
    "US": ("USA", "United States"),
    "CN": ("CHN", "China"),
    "JP": ("JPN", "Japan"),
    "KR": ("KOR", "South Korea"),
    "CA": ("CAN", "Canada"),
    "AU": ("AUS", "Australia"),
}


def iso2_to_iso3(code: str) -> str:
    """Best-effort ISO-2 to ISO-3 conversion (falls back to input if unknown)."""
    if not isinstance(code, str):
        return str(code)
    code = code.strip().upper()
    return ISO2_TO_ISO3_NAME.get(code, (code, code))[0]


def iso2_to_name(code: str) -> str:
    """Best-effort ISO-2 to country full name conversion (falls back to code if unknown)."""
    if not isinstance(code, str):
        return str(code)
    code = code.strip().upper()
    return ISO2_TO_ISO3_NAME.get(code, (code, code))[1]


# ---------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------


def wrap_label(s: str, width: int = 32) -> str:
    """Wrap long labels with <br> for Plotly."""
    s = "" if s is None else str(s)
    return "<br>".join(textwrap.wrap(s, width=width)) if len(s) > width else s


# ---------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------


def plot_country_map(
    df_country: pd.DataFrame,
    metric: str = "loss_pct",
    *,
    title: Optional[str] = None,
    use_country_names: bool = True,
) -> Any:
    """
    Choropleth map for country impacts.

    Parameters
    ----------
    df_country : DataFrame
        Must contain 'country' and `metric`.
        'country' is expected to be ISO-2.
    metric : str
        Column to visualize (e.g., 'loss_pct', 'loss_abs', 'VA_loss_pct', 'VA_loss_abs').
    title : str, optional
    use_country_names : bool
        If True, hover labels use full names.

    Returns
    -------
    plotly Figure
    """
    if "country" not in df_country.columns:
        raise ValueError("df_country must contain a 'country' column (ISO-2 codes).")
    if metric not in df_country.columns:
        raise ValueError(f"df_country does not contain metric column '{metric}'.")

    d = df_country.copy()
    d["iso3"] = d["country"].map(iso2_to_iso3)
    d["country_name"] = d["country"].map(iso2_to_name)

    hover_name = "country_name" if use_country_names else "country"

    if title is None:
        title = f"Country impact ({labels[metric]})"

    hover_data = {
        "country": True,
        "iso3": False,
        "country_name": False,
    }

    # Add metric with proper formatting
    is_pct = metric.endswith("_pct")
    if is_pct:
        # Values are already in 0-100 scale (from postprocess_results)
        # We don't use Plotly's ":" formatter with "%" because it multiplies by 100.
        hover_data[metric] = ":.3f"
    else:
        hover_data[metric] = ":.4f" if d[metric].dtype.kind in "fc" else True

    fig = px.choropleth(
        d,
        locations="iso3",
        locationmode="ISO-3",
        color=metric,
        hover_name=hover_name,
        hover_data=hover_data,
        title=title,
    )

    # Phase 2: Static Labels (Scattergeo overlay)
    # We only label countries that HAVE data in our d dataframe
    label_df = d.dropna(subset=[metric])

    # Optional: threshold to avoid clutter (only show labels for loss_pct > 0.1% or if absolute loss is large)
    # For now, we show all impacted countries.

    label_text = (
        [f"{v:.3f}%" for v in label_df[metric]]
        if is_pct
        else [f"{v:,.1f}" for v in label_df[metric]]
    )

    fig.add_trace(
        go.Scattergeo(
            locations=label_df["iso3"],
            locationmode="ISO-3",
            text=label_text,
            mode="text",
            textfont=dict(color="black", size=12, weight="bold"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    fig.update_geos(
        showcountries=True,
        countrycolor="LightGrey",
        showland=True,
        landcolor="white",
    )
    return fig


def plot_top_countries(
    df_country: pd.DataFrame,
    metric: str = "loss_abs",
    *,
    top_k: int = 20,
    title: Optional[str] = None,
    use_country_names: bool = True,
) -> Any:
    """Horizontal bar chart for top country impacts."""
    if "country" not in df_country.columns:
        raise ValueError("df_country must contain a 'country' column (ISO-2 codes).")
    if metric not in df_country.columns:
        raise ValueError(f"df_country does not contain metric column '{metric}'.")

    d = df_country.copy()
    d["country_label"] = (
        d["country"].map(iso2_to_name) if use_country_names else d["country"]
    )
    d = d.sort_values(metric, ascending=False).head(int(top_k))

    if title is None:
        title = f"Top countries by {metric}"

    fig = px.bar(
        d.sort_values(metric),
        x=metric,
        y="country_label",
        orientation="h",
        title=title,
        text=metric,
    )
    fig.update_traces(
        textposition="outside",
        texttemplate="%{text:,.1f}" if d[metric].dtype.kind in "fc" else "%{text}",
        textfont=dict(weight="bold"),
    )
    fig.update_layout(
        height=max(420, 22 * len(d) + 140),
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title="",
    )
    return fig


def plot_top_sectors(
    df_sector: pd.DataFrame,
    metric: str = "loss_abs",
    *,
    top_k: int = 20,
    title: Optional[str] = None,
    wrap_width: int = 40,
) -> Any:
    """Horizontal bar chart for top sector impacts, with wrapped labels."""
    if metric not in df_sector.columns:
        raise ValueError(f"df_sector does not contain metric column '{metric}'.")

    d = df_sector.copy()
    label_col = (
        "sector_name"
        if "sector_name" in d.columns
        else ("sector" if "sector" in d.columns else None)
    )
    if label_col is None:
        raise ValueError("df_sector must contain 'sector_name' or 'sector'.")

    d = d.sort_values(metric, ascending=False).head(int(top_k))
    d["sector_label"] = d[label_col].apply(lambda s: wrap_label(s, width=wrap_width))

    if title is None:
        title = f"Top sectors by {metric}"

    fig = px.bar(
        d.sort_values(metric),
        x=metric,
        y="sector_label",
        orientation="h",
        title=title,
        text=metric,
    )
    fig.update_traces(
        textposition="outside",
        texttemplate="%{text:,.1f}" if d[metric].dtype.kind in "fc" else "%{text}",
        textfont=dict(weight="bold"),
    )
    fig.update_layout(
        height=max(520, 26 * len(d) + 160),
        margin=dict(l=260, r=40, t=60, b=20),  # Increased right margin for labels
        yaxis_title="",
    )
    return fig


def plot_linkage_changes(
    df_links_strengthened: pd.DataFrame,
    df_links_weakened: pd.DataFrame,
    *,
    top_k: int = 20,
    title_strengthened: str = "Most strengthened linkages (ΔA)",
    title_weakened: str = "Most weakened linkages (ΔA)",
) -> Dict[str, Any]:
    """
    Build bar charts for strengthened/weakened linkages.

    Expected columns:
    - 'i_label', 'j_label' (preferred) or 'origin_node','dest_node'
    - 'delta' (absolute ΔA), and optionally 'delta_rel'

    Returns
    -------
    dict with two plotly figures: {'strengthened': fig1, 'weakened': fig2}
    """

    def _label_cols(df: pd.DataFrame) -> tuple[str, str]:
        if {"i_label", "j_label"}.issubset(df.columns):
            return "i_label", "j_label"
        if {"origin_node", "dest_node"}.issubset(df.columns):
            return "origin_node", "dest_node"
        raise ValueError(
            "Linkage df must have ('i_label','j_label') or ('origin_node','dest_node')."
        )

    def _build(df: pd.DataFrame, title: str) -> Any:
        i_col, j_col = _label_cols(df)
        if "delta" not in df.columns:
            raise ValueError("Linkage df must contain column 'delta' (absolute ΔA).")

        d = df.copy().head(int(top_k))
        d["edge"] = d[i_col].astype(str) + " → " + d[j_col].astype(str)

        d["delta_pp"] = 100.0 * d["delta"]  # percentage points
        if "delta_rel" in d.columns:
            d["delta_rel_pct"] = 100.0 * d["delta_rel"]
        else:
            d["delta_rel_pct"] = None

        # Keep original ordering (already ranked by absolute ΔA in postprocess)
        fig = px.bar(
            d.sort_values("delta_pp"),
            x="delta_pp",
            y="edge",
            orientation="h",
            title=title,
            hover_data={"delta_pp": ":.3f", "delta_rel_pct": ":.1f"},
            text="delta_pp",
        )
        fig.update_traces(
            textposition="outside",
            texttemplate="%{text:.3f}",
            textfont=dict(weight="bold"),
        )
        fig.update_layout(
            height=max(520, 26 * len(d) + 160),
            margin=dict(l=320, r=40, t=60, b=20),  # Increased right margin for labels
            yaxis_title="",
        )
        fig.update_xaxes(title_text="ΔA (percentage points)")
        return fig

    return {
        "strengthened": _build(df_links_strengthened, title_strengthened),
        "weakened": _build(df_links_weakened, title_weakened),
    }


def plot_supply_chain_heatmap(
    df_links_all: pd.DataFrame,
    *,
    perspective: str = "absolute",
    aggregation: str = "sector",
) -> Any:
    """Render a source-target supply-chain heatmap for linkage deltas."""
    matrix, norm_matrix, clip_limit = build_heatmap_frame(
        df_links_all, perspective=perspective, aggregation=aggregation
    )

    if matrix.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No supply-chain linkage data available for heatmap.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(height=520, xaxis_visible=False, yaxis_visible=False)
        return fig

    value_label = "ΔA" if perspective == "absolute" else "ΔA (%)"
    scale_name = "RdBu_r"

    z_values = matrix.to_numpy()
    if clip_limit > 0.0:
        z_values = z_values.clip(-clip_limit, clip_limit)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale=scale_name,
            zmid=0.0,
            zmin=-clip_limit if clip_limit > 0 else None,
            zmax=clip_limit if clip_limit > 0 else None,
            customdata=np.dstack([matrix.to_numpy(), norm_matrix.to_numpy()]),
            colorbar=dict(title=value_label),
            hovertemplate=(
                "Source: %{y}<br>Target: %{x}<br>"
                + "Raw value: %{customdata[0]:.4f}<br>"
                + f"Displayed ({value_label}): %{{z:.4f}}<br>"
                + "Intensity: %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Supply Chain Heatmap ({perspective.title()} · {aggregation.title()} level)",
        xaxis_title=f"Target {aggregation}",
        yaxis_title=f"Source {aggregation}",
        height=620,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    show_ticks = len(matrix.columns) <= 60
    fig.update_xaxes(tickangle=45, showticklabels=show_ticks)
    fig.update_yaxes(showticklabels=len(matrix.index) <= 60)
    return fig


# ---------------------------------------------------------------------
# Bundle builder (dashboard-ready)
# ---------------------------------------------------------------------


@dataclass
class DashboardBundle:
    """A thin container of tables + figures that can feed notebooks or apps."""

    tables: Dict[str, pd.DataFrame]
    figures: Dict[str, Any]
    meta: Dict[str, Any]


def build_dashboard_bundle(
    pp: Any,
    *,
    country_metric_for_map: str = "loss_pct",
    top_k_countries: int = 20,
    top_k_sectors: int = 20,
    top_k_links: int = 20,
    use_country_names: bool = True,
) -> DashboardBundle:
    """
    Build a consistent set of figures and tables from a PostprocessResults-like object.

    The object `pp` is expected to expose:
      - pp.df_country, pp.df_sector, pp.df_nodes
      - pp.df_links_strengthened, pp.df_links_weakened
      - pp.meta (dict)
    """
    tables = {
        "nodes": pp.df_nodes,
        "country": pp.df_country,
        "sector": pp.df_sector,
        "links_strengthened": pp.df_links_strengthened,
        "links_weakened": pp.df_links_weakened,
        "links_all": getattr(pp, "df_links_all", pd.DataFrame()),
    }

    figures: Dict[str, Any] = {
        "country_map": plot_country_map(
            pp.df_country,
            metric=country_metric_for_map,
            use_country_names=use_country_names,
        ),
        "top_countries": plot_top_countries(
            pp.df_country,
            metric="loss_abs",
            top_k=top_k_countries,
            use_country_names=use_country_names,
        ),
        "top_sectors": plot_top_sectors(
            pp.df_sector, metric="loss_abs", top_k=top_k_sectors
        ),
    }

    if (
        getattr(pp, "df_links_strengthened", None) is not None
        and len(pp.df_links_strengthened) > 0
    ):
        link_figs = plot_linkage_changes(
            pp.df_links_strengthened,
            pp.df_links_weakened,
            top_k=top_k_links,
        )
        figures.update(
            {
                "links_strengthened": link_figs["strengthened"],
                "links_weakened": link_figs["weakened"],
            }
        )

    meta = dict(getattr(pp, "meta", {}))
    return DashboardBundle(tables=tables, figures=figures, meta=meta)
