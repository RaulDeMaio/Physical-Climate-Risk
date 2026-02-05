"""
Hazard calibration pipeline (losses -> country-hazard intensity percentiles -> supply shock scalars).

Implements the methodology you described:

1) For each country c, compute hazard shares ω_{c,h} from cumulative losses (1980–2024) by hazard.
2) Allocate annual total losses L_{c,t} across hazards:
       L̂_{c,h,t} = ω_{c,h} * L_{c,t}
3) Normalize by total output (in constant 2024 prices, million EUR):
       I_{c,h,t} = L̂_{c,h,t} / OUT_{c,t}
4) Compute percentiles of I_{c,h,t} for scenario intensity levels.

Assumptions (v1)
----------------
- Uniform within-country sector vulnerability: sp is flat across all sectors of the shocked country.
- No exogenous demand shock (sd=0); demand reduction arises endogenously from the IO model.

Data inputs
-----------
- input_data.xlsx with:
  - sheet "Coutry_type": ISO2, hazard cumulative losses columns
  - sheet "Country_year": ISO2, yearly total losses (1980..2024)
  Values are in million EUR at 2024 price level.

- output series from Eurostat (nama_10_a64, na_item=P1) converted to constant 2024 prices,
  also in million EUR.

Public API
----------
- build_intensity_panel(...)
- build_percentile_table(...)
- shock_scalar(...)
- build_supply_shock_vector(...)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# --- Canonical hazard names (internal) ---------------------------------------
HAZARD_COL_MAP: Dict[str, str] = {
    "geotechnical": "geotechnical",
    "meteorological": "meteorological",
    "hydrological": "hydrological",
    "Climatological (other)": "climatological_other",
    "climatological (heatwaves)": "climatological_heatwaves",
}


INTENSITY_LEVELS_DEFAULT: Dict[str, float] = {
    "moderate": 0.50,
    "severe": 0.75,
    "extreme": 0.90,
    "very_extreme": 0.95,
}


@dataclass(frozen=True)
class CalibrationPaths:
    losses_xlsx_path: str


def load_losses_from_excel(paths: CalibrationPaths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_type: ISO2, Name, hazard cumulative losses (mn EUR, 2024 prices)
      df_year: ISO2, Name, year columns (mn EUR, 2024 prices)
    """
    df_type = pd.read_excel(paths.losses_xlsx_path, sheet_name="Coutry_type")
    df_year = pd.read_excel(paths.losses_xlsx_path, sheet_name="Country_year")
    return df_type, df_year


def _compute_hazard_shares(df_type: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ω_{c,h} hazard shares from cumulative losses.

    Returns long table with columns:
      ISO2, hazard, omega
    """
    # Select hazard columns and rename to canonical internal names
    hazard_cols = [c for c in df_type.columns if c in HAZARD_COL_MAP]
    df = df_type[["ISO2"] + hazard_cols].copy()
    df = df.rename(columns=HAZARD_COL_MAP)

    # Total cumulative losses per country
    haz_names = [HAZARD_COL_MAP[c] for c in hazard_cols]
    df["total"] = df[haz_names].sum(axis=1)

    # Avoid division by zero: if total==0 set all shares to 0
    for h in haz_names:
        df[h] = np.where(df["total"] > 0, df[h] / df["total"], 0.0)

    df_long = df.melt(id_vars=["ISO2"], value_vars=haz_names, var_name="hazard", value_name="omega")
    return df_long


def _annual_total_losses_long(df_year: pd.DataFrame) -> pd.DataFrame:
    """
    Melt annual losses to long:
      ISO2, year, loss_total_mn_eur_2024
    """
    year_cols = [c for c in df_year.columns if isinstance(c, (int, np.integer))]
    df = df_year[["ISO2"] + year_cols].copy()
    df_long = df.melt(id_vars=["ISO2"], value_vars=year_cols, var_name="year", value_name="loss_total_mn_eur_2024")
    df_long["year"] = df_long["year"].astype(int)
    df_long["loss_total_mn_eur_2024"] = df_long["loss_total_mn_eur_2024"].astype(float)
    return df_long


def build_intensity_panel(
    *,
    df_type: pd.DataFrame,
    df_year: pd.DataFrame,
    df_output: pd.DataFrame,
    # df_output columns expected: geo (ISO2), year (int), output_2024_mn_eur
    geo_col_output: str = "geo",
    years: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Construct intensity panel I_{c,h,t}.

    Returns long table:
      ISO2, year, hazard, loss_hazard_mn_eur_2024, output_2024_mn_eur, intensity
    """
    shares = _compute_hazard_shares(df_type)  # ISO2, hazard, omega
    annual = _annual_total_losses_long(df_year)  # ISO2, year, total loss

    # Join annual total with hazard shares -> hazard-specific annual loss
    df = annual.merge(shares, on="ISO2", how="left")
    df["loss_hazard_mn_eur_2024"] = df["loss_total_mn_eur_2024"] * df["omega"]

    # Join output (Eurostat) - align ISO2 code column name
    out = df_output.rename(columns={geo_col_output: "ISO2"})
    out = out[["ISO2", "year", "output_2024_mn_eur"]].copy()

    df = df.merge(out, on=["ISO2", "year"], how="inner")

    # Intensity as share of output (dimensionless)
    # If output==0 (rare), set intensity=0 to avoid blow-ups.
    df["intensity"] = np.where(df["output_2024_mn_eur"] > 0, df["loss_hazard_mn_eur_2024"] / df["output_2024_mn_eur"], 0.0)

    if years is not None:
        years = set(int(y) for y in years)
        df = df[df["year"].isin(years)].copy()

    return df[["ISO2", "year", "hazard", "loss_hazard_mn_eur_2024", "output_2024_mn_eur", "intensity"]]


def build_percentile_table(
    intensity_panel: pd.DataFrame,
    intensity_levels: Optional[Dict[str, float]] = None,
    min_years: int = 10,
) -> pd.DataFrame:
    """
    Compute intensity percentiles per (country, hazard) for scenario levels.

    Returns wide table:
      ISO2, hazard, <level_name>...
    """
    if intensity_levels is None:
        intensity_levels = INTENSITY_LEVELS_DEFAULT

    # Filter out countries/hazards with too few observations
    counts = intensity_panel.groupby(["ISO2", "hazard"])["intensity"].count().reset_index(name="n")
    valid = counts[counts["n"] >= min_years][["ISO2", "hazard"]]
    df = intensity_panel.merge(valid, on=["ISO2", "hazard"], how="inner")

    q = (
        df.groupby(["ISO2", "hazard"])["intensity"]
        .quantile(list(intensity_levels.values()))
        .reset_index()
        .rename(columns={"level_2": "q", "intensity": "value"})
    )

    # Map quantile values back to level names
    inv = {v: k for k, v in intensity_levels.items()}
    q["level"] = q["q"].map(inv)

    wide = q.pivot_table(index=["ISO2", "hazard"], columns="level", values="value", aggfunc="first").reset_index()
    # Ensure all columns exist
    for lvl in intensity_levels.keys():
        if lvl not in wide.columns:
            wide[lvl] = np.nan
    return wide


def shock_scalar(
    percentile_table: pd.DataFrame,
    *,
    country_iso2: str,
    hazard: str,
    intensity_level: str,
    clamp_max: float = 0.50,
) -> float:
    """
    Retrieve a shock scalar φ for a (country, hazard, intensity_level).

    - If not available, returns 0.
    - Clamps to [0, clamp_max] to avoid unrealistic shocks.
    """
    sel = percentile_table[(percentile_table["ISO2"] == country_iso2) & (percentile_table["hazard"] == hazard)]
    if sel.empty or intensity_level not in sel.columns:
        return 0.0
    val = float(sel.iloc[0][intensity_level])
    if np.isnan(val) or val < 0:
        return 0.0
    return float(np.clip(val, 0.0, clamp_max))


def build_supply_shock_vector(
    node_labels: Sequence[str],
    *,
    country_iso2: str,
    shock_scalar: float,
) -> np.ndarray:
    """
    Flat supply shock across all sectors within a country.

    node_labels are "CC::P_..." strings.
    Returns sp vector length n with sp[i]=shock_scalar if country matches, else 0.
    """
    n = len(node_labels)
    sp = np.zeros(n, dtype=float)
    for i, lab in enumerate(node_labels):
        c, _sec = lab.split("::", 1)
        if c == country_iso2:
            sp[i] = shock_scalar
    return sp
