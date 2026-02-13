from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd

AggregationLevel = Literal["node", "country", "sector"]


def normalize_intensity(values: pd.Series, quantile_cap: float = 0.95) -> pd.Series:
    """Normalize magnitudes into [0, 1] with deterministic quantile capping."""
    if values is None or len(values) == 0:
        return pd.Series(dtype=float)

    clean = pd.to_numeric(values, errors="coerce").fillna(0.0).abs()
    if clean.empty:
        return pd.Series(dtype=float)

    cap = (
        float(clean.quantile(quantile_cap, interpolation="nearest"))
        if clean.notna().any()
        else 0.0
    )
    if cap <= 0.0:
        return pd.Series(np.zeros(len(clean)), index=clean.index, dtype=float)

    clipped = clean.clip(upper=cap)
    return (clipped / cap).clip(lower=0.0, upper=1.0)


def symmetric_clip_limit(values: pd.Series, quantile_cap: float = 0.98) -> float:
    """Return a symmetric +/- clipping threshold from absolute quantile."""
    clean = pd.to_numeric(values, errors="coerce").fillna(0.0).abs()
    if clean.empty:
        return 0.0
    return float(clean.quantile(quantile_cap, interpolation="nearest"))


def map_coordinates_to_viewport(
    points: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    bounds: Tuple[float, float, float, float],
    mode: Literal["clip", "drop"] = "clip",
) -> pd.DataFrame:
    """Map logical coordinates into viewport bounds and mark in/out state."""
    xmin, xmax, ymin, ymax = bounds
    mapped = points.copy()

    x = pd.to_numeric(mapped[x_col], errors="coerce")
    y = pd.to_numeric(mapped[y_col], errors="coerce")
    in_bounds = x.between(xmin, xmax) & y.between(ymin, ymax)

    mapped["in_bounds"] = in_bounds
    mapped["x_mapped"] = x
    mapped["y_mapped"] = y

    if mode == "clip":
        mapped["x_mapped"] = mapped["x_mapped"].clip(lower=xmin, upper=xmax)
        mapped["y_mapped"] = mapped["y_mapped"].clip(lower=ymin, upper=ymax)
    elif mode == "drop":
        mapped = mapped[mapped["in_bounds"]].copy()
    else:
        raise ValueError("mode must be either 'clip' or 'drop'")

    return mapped


def _split_node_label(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = series.astype(str).str.split("::", n=1, expand=True)
    if parts.shape[1] == 1:
        return parts[0], parts[0]
    return parts[0], parts[1]


def _resolve_axes(
    working: pd.DataFrame,
    aggregation: AggregationLevel,
) -> tuple[pd.Series, pd.Series]:
    if {"i_label", "j_label"}.issubset(working.columns):
        src = working["i_label"]
        dst = working["j_label"]
    elif {"origin_node", "dest_node"}.issubset(working.columns):
        src = working["origin_node"]
        dst = working["dest_node"]
    elif {"source", "target"}.issubset(working.columns):
        src = working["source"]
        dst = working["target"]
    else:
        raise ValueError("links_df must contain one of (i_label,j_label), (origin_node,dest_node), (source,target)")

    if aggregation == "node":
        return src, dst

    src_country, src_sector = _split_node_label(src)
    dst_country, dst_sector = _split_node_label(dst)

    if aggregation == "country":
        return src_country, dst_country
    if aggregation == "sector":
        return src_sector, dst_sector

    raise ValueError("aggregation must be one of: node, country, sector")


def build_heatmap_frame(
    links_df: pd.DataFrame,
    *,
    perspective: Literal["absolute", "percentage"] = "absolute",
    aggregation: AggregationLevel = "sector",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Build value/intensity matrices and clipping range for heatmap rendering."""
    if links_df is None or links_df.empty:
        return pd.DataFrame(), pd.DataFrame(), 0.0

    value_col = "delta" if perspective == "absolute" else "delta_rel"
    if value_col not in links_df.columns:
        raise ValueError(f"links_df must contain '{value_col}' for {perspective} view")

    working = links_df.copy()
    src_axis, dst_axis = _resolve_axes(working, aggregation)

    working[value_col] = pd.to_numeric(working[value_col], errors="coerce").fillna(0.0)
    working["_src_axis"] = src_axis.astype(str)
    working["_dst_axis"] = dst_axis.astype(str)

    matrix = (
        working.groupby(["_src_axis", "_dst_axis"], observed=True)[value_col]
        .sum()
        .unstack(fill_value=0.0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    flat = matrix.stack(future_stack=True)
    norm_long = normalize_intensity(flat, quantile_cap=0.95)
    norm_matrix = norm_long.unstack(fill_value=0.0).reindex_like(matrix).fillna(0.0)
    clip_limit = symmetric_clip_limit(flat, quantile_cap=0.98)

    return matrix, norm_matrix, clip_limit
