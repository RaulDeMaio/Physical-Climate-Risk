# src/io_climate/postprocess.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PostProcessOutputs:
    """
    Container for model post-processing outputs.

    Attributes
    ----------
    df_nodes : pd.DataFrame
        Node-level impacts (country-sector nodes).
    df_country : pd.DataFrame
        Country aggregates.
    df_sector : pd.DataFrame
        Sector aggregates.
    df_links_weakened : pd.DataFrame
        Top weakened linkages (by delta technical coefficient or delta flow).
    df_links_strengthened : pd.DataFrame
        Top strengthened linkages.
    meta : dict
        Additional metadata and scalar KPIs.
    """
    df_nodes: pd.DataFrame
    df_country: pd.DataFrame
    df_sector: pd.DataFrame
    df_links_weakened: pd.DataFrame
    df_links_strengthened: pd.DataFrame
    meta: Dict[str, float]


def postprocess_results(
    *,
    node_labels: Sequence[str],
    Z0: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    X1: np.ndarray,
    FD_post: Optional[np.ndarray] = None,
    # Optional satellites
    va0: Optional[np.ndarray] = None,
    emp0: Optional[np.ndarray] = None,
    emp_per_output: Optional[np.ndarray] = None,
    # Optional decoding maps
    sector_name_map: Optional[Dict[str, str]] = None,
    country_name_map: Optional[Dict[str, str]] = None,
    # Linkage analysis controls
    linkage_metric: str = "A",   # "A" or "Z"
    top_k_links: int = 25,
) -> PostProcessOutputs:
    """
    Compute a standardized suite of outputs from a model run.

    Parameters
    ----------
    node_labels
        Length-n list of node labels formatted as "CC::P_...".
    Z0, X0
        Baseline intermediate matrix and gross output (n x n) and (n,).
        Z is producer i -> user j.
    Z1, X1
        Final intermediate matrix and feasible gross output.
    FD_post
        Final post-shock effective demand vector (optional; used for KPIs only).
    va0
        Baseline value-added vector (n,) if available from the SAM GDP block.
        If None, VA will be computed as IO residual: VA = X - colsum(Z).
    emp0
        Baseline employment vector (n,) if available.
    emp_per_output
        Alternative to emp0: employment intensity coefficients (jobs per unit output).
        If provided and emp0 is None, emp0 := emp_per_output * X0.
    sector_name_map
        Optional dict mapping sector codes like "P_C10-12" -> description.
    country_name_map
        Optional dict mapping country codes like "IT" -> country name (optional).
    linkage_metric
        "A": compare technical coefficients A = Z / X (per column).
        "Z": compare absolute intermediate flows.
    top_k_links
        Number of strongest positive / negative linkage changes to report.

    Returns
    -------
    PostProcessOutputs
        Standardized outputs: node/country/sector frames + linkage deltas + meta KPIs.
    """
    # ---- validate ----
    Z0 = np.asarray(Z0, dtype=float)
    Z1 = np.asarray(Z1, dtype=float)
    X0 = np.asarray(X0, dtype=float).reshape(-1)
    X1 = np.asarray(X1, dtype=float).reshape(-1)

    if Z0.shape != Z1.shape or Z0.shape[0] != Z0.shape[1]:
        raise ValueError("Z0 and Z1 must be square matrices of the same shape.")
    n = Z0.shape[0]
    if len(node_labels) != n or X0.shape[0] != n or X1.shape[0] != n:
        raise ValueError("node_labels, X0, X1 must all have length n = Z.shape[0].")

    # ---- parse labels ----
    countries = [lbl.split("::")[0] for lbl in node_labels]
    sectors = [lbl.split("::")[1] for lbl in node_labels]

    # ---- output impacts ----
    df_nodes = pd.DataFrame(
        {
            "node": list(node_labels),
            "country": countries,
            "sector": sectors,
            "X_baseline": X0,
            "X_final": X1,
        }
    )
    df_nodes["loss_abs"] = df_nodes["X_baseline"] - df_nodes["X_final"]
    df_nodes["loss_pct"] = np.where(
        df_nodes["X_baseline"] > 0.0,
        100.0 * df_nodes["loss_abs"] / df_nodes["X_baseline"],
        0.0,
    )

    # Optional decoded names
    if sector_name_map is not None:
        df_nodes["sector_name"] = df_nodes["sector"].map(sector_name_map).fillna(df_nodes["sector"])
    if country_name_map is not None:
        df_nodes["country_name"] = df_nodes["country"].map(country_name_map).fillna(df_nodes["country"])

    # ---- value added ----
    # IO residual VA definition: VA_j = X_j - sum_i Z_{i,j} (intermediate inputs, column sum)
    if va0 is None:
        va0 = X0 - Z0.sum(axis=0)
    else:
        va0 = np.asarray(va0, dtype=float).reshape(-1)
        if va0.shape[0] != n:
            raise ValueError("va0 must have length n.")

    va1 = X1 - Z1.sum(axis=0)
    va1 = np.maximum(va1, 0.0)

    df_nodes["VA_baseline"] = va0
    df_nodes["VA_final"] = va1
    df_nodes["VA_loss_abs"] = df_nodes["VA_baseline"] - df_nodes["VA_final"]
    df_nodes["VA_loss_pct"] = np.where(
        df_nodes["VA_baseline"] > 0.0,
        100.0 * df_nodes["VA_loss_abs"] / df_nodes["VA_baseline"],
        0.0,
    )

    # ---- employment ----
    if emp0 is None and emp_per_output is not None:
        emp_per_output = np.asarray(emp_per_output, dtype=float).reshape(-1)
        if emp_per_output.shape[0] != n:
            raise ValueError("emp_per_output must have length n.")
        emp0 = emp_per_output * X0

    if emp0 is not None:
        emp0 = np.asarray(emp0, dtype=float).reshape(-1)
        if emp0.shape[0] != n:
            raise ValueError("emp0 must have length n.")
        # Scale employment with output (first-order approximation)
        scale = np.divide(X1, X0, out=np.zeros_like(X1), where=X0 > 0.0)
        emp1 = emp0 * scale

        df_nodes["EMP_baseline"] = emp0
        df_nodes["EMP_final"] = emp1
        df_nodes["EMP_loss_abs"] = df_nodes["EMP_baseline"] - df_nodes["EMP_final"]
        df_nodes["EMP_loss_pct"] = np.where(
            df_nodes["EMP_baseline"] > 0.0,
            100.0 * df_nodes["EMP_loss_abs"] / df_nodes["EMP_baseline"],
            0.0,
        )

    # ---- aggregates ----
    df_country = (
        df_nodes.groupby("country", as_index=False)
        .agg(
            X_baseline=("X_baseline", "sum"),
            X_final=("X_final", "sum"),
            loss_abs=("loss_abs", "sum"),
            VA_baseline=("VA_baseline", "sum"),
            VA_final=("VA_final", "sum"),
            VA_loss_abs=("VA_loss_abs", "sum"),
        )
    )
    df_country["loss_pct"] = np.where(df_country["X_baseline"] > 0, 100.0 * df_country["loss_abs"] / df_country["X_baseline"], 0.0)
    df_country["VA_loss_pct"] = np.where(df_country["VA_baseline"] > 0, 100.0 * df_country["VA_loss_abs"] / df_country["VA_baseline"], 0.0)

    if "EMP_baseline" in df_nodes.columns:
        emp_country = (
            df_nodes.groupby("country", as_index=False)
            .agg(EMP_baseline=("EMP_baseline", "sum"), EMP_final=("EMP_final", "sum"), EMP_loss_abs=("EMP_loss_abs", "sum"))
        )
        emp_country["EMP_loss_pct"] = np.where(emp_country["EMP_baseline"] > 0, 100.0 * emp_country["EMP_loss_abs"] / emp_country["EMP_baseline"], 0.0)
        df_country = df_country.merge(emp_country, on="country", how="left")

    sector_group_cols = ["sector"]
    if "sector_name" in df_nodes.columns:
        sector_group_cols.append("sector_name")

    df_sector = (
        df_nodes.groupby(sector_group_cols, as_index=False)
        .agg(
            X_baseline=("X_baseline", "sum"),
            X_final=("X_final", "sum"),
            loss_abs=("loss_abs", "sum"),
            VA_baseline=("VA_baseline", "sum"),
            VA_final=("VA_final", "sum"),
            VA_loss_abs=("VA_loss_abs", "sum"),
        )
    )
    df_sector["loss_pct"] = np.where(df_sector["X_baseline"] > 0, 100.0 * df_sector["loss_abs"] / df_sector["X_baseline"], 0.0)
    df_sector["VA_loss_pct"] = np.where(df_sector["VA_baseline"] > 0, 100.0 * df_sector["VA_loss_abs"] / df_sector["VA_baseline"], 0.0)

    if "EMP_baseline" in df_nodes.columns:
        emp_sector = (
            df_nodes.groupby(sector_group_cols, as_index=False)
            .agg(EMP_baseline=("EMP_baseline", "sum"), EMP_final=("EMP_final", "sum"), EMP_loss_abs=("EMP_loss_abs", "sum"))
        )
        emp_sector["EMP_loss_pct"] = np.where(emp_sector["EMP_baseline"] > 0, 100.0 * emp_sector["EMP_loss_abs"] / emp_sector["EMP_baseline"], 0.0)
        df_sector = df_sector.merge(emp_sector, on=sector_group_cols, how="left")

    # ---- linkage deltas ----
    df_links_weakened, df_links_strengthened = top_linkage_changes(
        node_labels=node_labels,
        Z0=Z0,
        X0=X0,
        Z1=Z1,
        X1=X1,
        metric=linkage_metric,
        top_k=top_k_links,
    )

    # ---- meta KPIs ----
    meta: Dict[str, float] = {
        "X_baseline_total": float(np.sum(X0)),
        "X_final_total": float(np.sum(X1)),
        "X_loss_abs_total": float(np.sum(X0) - np.sum(X1)),
        "X_loss_pct_total": float(100.0 * (np.sum(X0) - np.sum(X1)) / (np.sum(X0) + 1e-12)),
        "VA_baseline_total": float(np.sum(va0)),
        "VA_final_total": float(np.sum(va1)),
        "VA_loss_abs_total": float(np.sum(va0) - np.sum(va1)),
        "VA_loss_pct_total": float(100.0 * (np.sum(va0) - np.sum(va1)) / (np.sum(va0) + 1e-12)),
    }
    if FD_post is not None:
        FD_post = np.asarray(FD_post, dtype=float).reshape(-1)
        if FD_post.shape[0] == n:
            meta["FD_post_total"] = float(np.sum(FD_post))

    if emp0 is not None:
        meta["EMP_baseline_total"] = float(np.sum(emp0))
        meta["EMP_final_total"] = float(np.sum(df_nodes["EMP_final"]))
        meta["EMP_loss_abs_total"] = float(np.sum(df_nodes["EMP_loss_abs"]))
        meta["EMP_loss_pct_total"] = float(100.0 * np.sum(df_nodes["EMP_loss_abs"]) / (np.sum(emp0) + 1e-12))

    return PostProcessOutputs(
        df_nodes=df_nodes,
        df_country=df_country.sort_values("loss_abs", ascending=False),
        df_sector=df_sector.sort_values("loss_abs", ascending=False),
        df_links_weakened=df_links_weakened,
        df_links_strengthened=df_links_strengthened,
        meta=meta,
    )


def top_linkage_changes(
    *,
    node_labels: Sequence[str],
    Z0: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    X1: np.ndarray,
    metric: str = "A",
    top_k: int = 25,
    min_baseline: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify the strongest weakened/strengthened linkages between nodes.

    Parameters
    ----------
    metric
        "A" for technical coefficients deltas (recommended for structural change),
        "Z" for absolute flow deltas.
    top_k
        Number of links to return on each side.
    min_baseline
        If metric="Z", ignore baseline flows below this threshold.

    Returns
    -------
    weakened, strengthened : pd.DataFrame, pd.DataFrame
        Each has columns: i_node, j_node, i_label, j_label, baseline, final, delta
    """
    Z0 = np.asarray(Z0, dtype=float)
    Z1 = np.asarray(Z1, dtype=float)
    X0 = np.asarray(X0, dtype=float).reshape(-1)
    X1 = np.asarray(X1, dtype=float).reshape(-1)

    n = Z0.shape[0]
    if len(node_labels) != n:
        raise ValueError("node_labels length must match Z shape.")

    if metric.upper() == "A":
        # A[i,j] = Z[i,j] / X[j] (column scaling)
        denom0 = X0.copy()
        denom1 = X1.copy()
        denom0[denom0 == 0.0] = np.nan
        denom1[denom1 == 0.0] = np.nan

        A0 = Z0 / denom0[None, :]
        A1 = Z1 / denom1[None, :]
        A0 = np.nan_to_num(A0, nan=0.0)
        A1 = np.nan_to_num(A1, nan=0.0)

        base = A0
        final = A1
    elif metric.upper() == "Z":
        base = Z0
        final = Z1
    else:
        raise ValueError("metric must be either 'A' or 'Z'.")

    delta = final - base

    # Optional pruning for Z deltas (ignore tiny baseline flows)
    if metric.upper() == "Z" and min_baseline > 0.0:
        mask = base >= min_baseline
        delta = np.where(mask, delta, 0.0)
        base = np.where(mask, base, 0.0)
        final = np.where(mask, final, 0.0)

    # Flatten and take top-K negative and positive
    flat = delta.ravel()
    if flat.size == 0:
        empty = pd.DataFrame(columns=["i_node", "j_node", "i_label", "j_label", "baseline", "final", "delta"])
        return empty, empty

    # strongest strengthened: largest positive deltas
    pos_idx = np.argpartition(flat, -top_k)[-top_k:]
    pos_idx = pos_idx[np.argsort(flat[pos_idx])[::-1]]

    # strongest weakened: most negative deltas
    neg_idx = np.argpartition(flat, top_k)[:top_k]
    neg_idx = neg_idx[np.argsort(flat[neg_idx])]

    def _build_df(idxs):
        i = idxs // n
        j = idxs % n
        return pd.DataFrame(
            {
                "i_node": i,
                "j_node": j,
                "i_label": [node_labels[ii] for ii in i],
                "j_label": [node_labels[jj] for jj in j],
                "baseline": base[i, j],
                "final": final[i, j],
                "delta": delta[i, j],
            }
        )

    strengthened = _build_df(pos_idx)
    weakened = _build_df(neg_idx)

    # Clean ordering: strengthened high->low, weakened low->high
    strengthened = strengthened.sort_values("delta", ascending=False).reset_index(drop=True)
    weakened = weakened.sort_values("delta", ascending=True).reset_index(drop=True)

    return weakened, strengthened
