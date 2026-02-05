"""
Eurostat national accounts output loader (nama_10_a64, na_item=P1).

Downloads gross output (P1) and converts it to constant 2024 prices,
in million EUR, using a data-internal implicit deflator:

- CP_MEUR: output at current prices, million EUR
- CLV20_MEUR: chain-linked volumes (reference year 2020), million EUR

For each country (geo), compute:
    price_level_2024 = CP_MEUR(2024) / CLV20_MEUR(2024)

Then:
    OUT_2024_MEUR(t) = CLV20_MEUR(t) * price_level_2024

Why this approach?
- Avoids guessing Eurostat price-index conventions.
- Internally consistent across nominal and volume series.

API formats
-----------
Eurostat endpoints can return JSON-stat (dimension/value) or SDMX JSON (structure/dataSets).
This loader supports both.

Returned units: million EUR (MEUR).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import requests


@dataclass(frozen=True)
class EurostatOutputConfig:
    api_url: str
    cache_path: Optional[str] = None  # parquet file
    timeout_s: int = 60


# ------------------------------ Parsers ------------------------------------- #
def _parse_sdmx3_json_to_long(js: Dict[str, Any]) -> pd.DataFrame:
    """Parse SDMX 3.0 JSON response to long DataFrame."""
    structure = js["structure"]
    obs_dims = structure["dimensions"]["observation"]
    dim_names = [d["id"] for d in obs_dims]
    dim_values = [[v["id"] for v in d["values"]] for d in obs_dims]

    obs = js["dataSets"][0]["observations"]

    rows: List[Dict[str, Any]] = []
    for key, val in obs.items():
        idx = [int(x) for x in key.split(":")]
        rec = {dim_names[i]: dim_values[i][idx[i]] for i in range(len(dim_names))}
        rec["value"] = float(val[0]) if val and val[0] is not None else None
        rows.append(rec)

    return pd.DataFrame(rows)


def _parse_jsonstat_to_long(js: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse JSON-stat response to long DataFrame.

    Supports sparse 'value' dict encoding (Eurostat commonly uses this).
    """
    if "dimension" not in js or "value" not in js:
        raise ValueError("Not a JSON-stat response (missing 'dimension'/'value').")

    # Dimension order
    dim_order = js["id"] if isinstance(js.get("id"), list) else list(js["dimension"].keys())

    # Authoritative sizes from JSON-stat
    if not (isinstance(js.get("size"), list) and len(js["size"]) == len(dim_order)):
        raise ValueError("JSON-stat missing consistent 'size' array.")
    sizes = [int(x) for x in js["size"]]

    # Dimension code lists in correct order
    dim_values: List[List[str]] = []
    for d in dim_order:
        cat = js["dimension"][d]["category"]
        idx = cat.get("index")

        if isinstance(idx, dict):
            # codes keyed by position
            codes_sorted = [k for k, _ in sorted(idx.items(), key=lambda kv: kv[1])]
            dim_values.append(codes_sorted)
        elif isinstance(idx, list):
            dim_values.append(idx)
        else:
            lab = cat.get("label", {})
            if isinstance(lab, dict) and lab:
                dim_values.append(list(lab.keys()))
            else:
                raise ValueError(f"Cannot parse JSON-stat dimension '{d}'.")

    # Build dense value vector of length N = prod(sizes)
    N = int(np.prod(sizes))
    raw_val = js["value"]
    arr = np.full(N, np.nan, dtype=float)

    if isinstance(raw_val, dict):
        # Sparse dict: keys are positions in dense vector
        for k, v in raw_val.items():
            if v is None:
                continue
            idx = int(k)
            if 0 <= idx < N:
                arr[idx] = float(v)
    elif isinstance(raw_val, list):
        # Dense list
        L = min(len(raw_val), N)
        arr[:L] = [np.nan if v is None else float(v) for v in raw_val[:L]]
    else:
        raise ValueError(f"Unsupported JSON-stat 'value' type: {type(raw_val)}")

    # Expand cartesian product indices
    grids = np.indices(sizes).reshape(len(sizes), -1).T  # shape (N, ndim)

    rows: List[Dict[str, Any]] = []
    for pos, val in zip(grids, arr):
        rec = {dim_order[i]: dim_values[i][pos[i]] for i in range(len(dim_order))}
        rec["value"] = val
        rows.append(rec)

    return pd.DataFrame(rows)




def _to_long(js: Dict[str, Any]) -> pd.DataFrame:
    """Auto-detect Eurostat response format."""
    if "structure" in js and "dataSets" in js:
        return _parse_sdmx3_json_to_long(js)
    if "dimension" in js and "value" in js:
        return _parse_jsonstat_to_long(js)
    if "error" in js:
        raise ValueError(f"Eurostat API error: {js['error']}")
    raise ValueError(f"Unrecognized Eurostat response format. Keys: {list(js.keys())[:25]}")


# ------------------------------ Public API ---------------------------------- #
def fetch_output_2024_prices_mn_eur(
    *,
    config: EurostatOutputConfig,
    geos: Optional[Iterable[str]] = None,
    years: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Download P1 output and convert to constant 2024 prices (million EUR).

    Returns columns:
      - geo
      - year
      - output_2024_mn_eur
      - cp_mn_eur
      - clv20_mn_eur
      - price_level_2024
    """
    cache_path = Path(config.cache_path) if config.cache_path else None
    if cache_path and cache_path.exists():
        return pd.read_parquet(cache_path)

    resp = requests.get(config.api_url, timeout=config.timeout_s)
    resp.raise_for_status()

    try:
        js = resp.json()
    except Exception as e:
        raise ValueError(f"Eurostat response is not JSON. First 300 chars:\n{resp.text[:300]}") from e

    df_long = _to_long(js)

    # Identify time column
    time_col = None
    for cand in ("TIME_PERIOD", "time", "TIME", "year"):
        if cand in df_long.columns:
            time_col = cand
            break
    if time_col is None:
        raise ValueError(f"Cannot find time dimension. Columns: {df_long.columns.tolist()}")

    df_long["year"] = df_long[time_col].astype(int)

    # Identify geo/unit columns (JSON-stat sometimes capitalizes)
    geo_col = "geo" if "geo" in df_long.columns else ("GEO" if "GEO" in df_long.columns else None)
    unit_col = "unit" if "unit" in df_long.columns else ("UNIT" if "UNIT" in df_long.columns else None)
    if geo_col is None:
        raise ValueError(f"Cannot find geo dimension. Columns: {df_long.columns.tolist()}")
    if unit_col is None:
        raise ValueError(f"Cannot find unit dimension. Columns: {df_long.columns.tolist()}")

    df_long = df_long.rename(columns={geo_col: "geo", unit_col: "unit"})

    # Filter to the 2 needed units
    df_long = df_long[df_long["unit"].isin(["CP_MEUR", "CLV20_MEUR"])].copy()

    if geos is not None:
        geos = set(geos)
        df_long = df_long[df_long["geo"].isin(geos)].copy()

    if years is not None:
        years = set(int(y) for y in years)
        df_long = df_long[df_long["year"].isin(years)].copy()

    # Pivot to CP and CLV20
    df_piv = (
        df_long.pivot_table(index=["geo", "year"], columns="unit", values="value", aggfunc="first")
        .reset_index()
        .rename(columns={"CP_MEUR": "cp_mn_eur", "CLV20_MEUR": "clv20_mn_eur"})
    )
   
    # --- Preferred method: country-specific CLV scaled to 2024 price level --------
    base = df_piv[df_piv["year"] == 2024][["geo", "cp_mn_eur", "clv20_mn_eur"]].copy()
    if base.empty:
        raise ValueError("No year=2024 observations found in API response (needed for 2024 scaling).")

    base["price_level_2024"] = base["cp_mn_eur"] / base["clv20_mn_eur"]
    base = base[["geo", "price_level_2024"]]

    df_out = df_piv.merge(base, on="geo", how="left")
    df_out["output_2024_mn_eur"] = df_out["clv20_mn_eur"] * df_out["price_level_2024"]

    # --- Fallback: EU-wide deflator computed from country sums --------------------
    # Compute dEU_t = sum(CP_t) / sum(CLV_t) over countries with both values in year t.
    eu = df_piv[["year", "cp_mn_eur", "clv20_mn_eur"]].copy()
    eu = eu.dropna(subset=["cp_mn_eur", "clv20_mn_eur"])

    eu_sum = (
        eu.groupby("year", as_index=False)
        .agg(cp_sum=("cp_mn_eur", "sum"), clv_sum=("clv20_mn_eur", "sum"))
    )

    # Guard: need a valid 2024 deflator
    eu_sum["d_eu"] = eu_sum["cp_sum"] / eu_sum["clv_sum"]
    row_2024 = eu_sum.loc[eu_sum["year"] == 2024, "d_eu"]

    if not row_2024.empty:
        d_eu_2024 = float(row_2024.iloc[0])

        df_out = df_out.merge(eu_sum[["year", "d_eu"]], on="year", how="left")

        # Apply fallback ONLY where preferred output_2024_mn_eur is missing
        missing = df_out["output_2024_mn_eur"].isna()

        # OUT_2024 = CP_t * (dEU_2024 / dEU_t)
        df_out.loc[missing, "output_2024_mn_eur"] = (
            df_out.loc[missing, "cp_mn_eur"] * (d_eu_2024 / df_out.loc[missing, "d_eu"])
        )

        # Optional: tag fallback usage for diagnostics
        df_out.loc[missing, "used_eu_deflator_fallback"] = True
    else:
        # Optional: keep column for consistent schema
        df_out["used_eu_deflator_fallback"] = False


    # Final cleaning: drop rows still missing (no CP or no EU deflator, etc.)
    df_out = df_out.dropna(subset=["output_2024_mn_eur"])

    if "used_eu_deflator_fallback" in df_out.columns:
        print("Rows using EU-deflator fallback:", int(df_out["used_eu_deflator_fallback"].fillna(False).sum()))
        print("Countries touched:", sorted(df_out.loc[df_out["used_eu_deflator_fallback"].fillna(False), "geo"].unique().tolist())[:50])


    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(cache_path, index=False)

    return df_out
