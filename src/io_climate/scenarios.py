# src/io_climate/scenarios.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Set

import numpy as np


Label = str
CodeOrCodes = Union[str, Sequence[str], None]


@dataclass(frozen=True)
class NodeKey:
    """Parsed representation of a node label 'CC::SECTOR'."""
    country: str
    sector: str


def _as_set(x: CodeOrCodes) -> Set[str]:
    """
    Normalize a code input to a set of strings.

    - None -> empty set (interpreted as 'no filter')
    - "IT" -> {"IT"}
    - ["IT","DE"] -> {"IT","DE"}
    """
    if x is None:
        return set()
    if isinstance(x, str):
        x = [x]
    return {str(v) for v in x}


def _validate_pct(name: str, pct: float) -> None:
    if not np.isfinite(pct):
        raise ValueError(f"{name} must be a finite number (got {pct}).")
    if pct < 0.0 or pct > 100.0:
        raise ValueError(f"{name} must be in [0, 100] (got {pct}).")


def _parse_node_label(label: str) -> NodeKey:
    """
    Parse 'CC::SECTOR' node label into NodeKey.

    Raises
    ------
    ValueError if the label does not match the expected format.
    """
    if "::" not in label:
        raise ValueError(
            f"Invalid node label '{label}'. Expected format 'CC::SECTOR'."
        )
    parts = label.split("::")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid node label '{label}'. Expected exactly one '::' separator."
        )
    country, sector = parts[0].strip(), parts[1].strip()
    if not country or not sector:
        raise ValueError(
            f"Invalid node label '{label}'. Country/sector cannot be empty."
        )
    return NodeKey(country=country, sector=sector)


def make_shock_vectors(
    node_labels: Sequence[Label],
    country_codes: CodeOrCodes = None,
    sector_codes: CodeOrCodes = None,
    supply_shock_pct: float = 0.0,
    demand_shock_pct: float = 0.0,
    *,
    # Optional: separate targeting for supply vs demand (future-proof).
    # If provided, these override the shared country_codes/sector_codes for that shock type.
    supply_country_codes: CodeOrCodes = None,
    supply_sector_codes: CodeOrCodes = None,
    demand_country_codes: CodeOrCodes = None,
    demand_sector_codes: CodeOrCodes = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build demand (sd) and supply (sp) shock vectors for IOClimateModel.

    This utility maps user-friendly shock targeting (countries/sectors + % sizes)
    into full-length vectors aligned with `node_labels`.

    Parameters
    ----------
    node_labels
        Sequence of node labels formatted as "CC::P_XXX" (e.g., "IT::P_C10-12").
        Vector indices correspond 1:1 with this ordering.
    country_codes, sector_codes
        Shared filters used for both supply and demand shocks unless overridden
        by supply_* or demand_* filters.
        - None means "no filter" (selects all countries/sectors).
        - String means one code (e.g., "IT").
        - Sequence means multiple codes (e.g., ["IT", "DE"]).
    supply_shock_pct
        Capacity reduction in percent (0–100). Example: 5 means 5% capacity loss.
    demand_shock_pct
        Final demand reduction in percent (0–100). Example: 10 means 10% demand loss.

    supply_country_codes, supply_sector_codes
        Optional overrides for the supply shock target set.
    demand_country_codes, demand_sector_codes
        Optional overrides for the demand shock target set.

    Returns
    -------
    sd, sp
        (n,) numpy arrays of floats in [0,1], where:
        - sd[j] is the fraction of final demand lost at node j
        - sp[i] is the fraction of capacity lost at node i

    Notes
    -----
    - Shocks are applied elementwise to nodes that match BOTH country and sector filters.
    - If the relevant filter set is empty, it is interpreted as "match all".
    """
    if node_labels is None or len(node_labels) == 0:
        raise ValueError("node_labels must be a non-empty sequence.")

    _validate_pct("supply_shock_pct", float(supply_shock_pct))
    _validate_pct("demand_shock_pct", float(demand_shock_pct))

    # Convert to fractions
    sp_frac = float(supply_shock_pct) / 100.0
    sd_frac = float(demand_shock_pct) / 100.0

    # Shared filters (fallback)
    shared_countries = _as_set(country_codes)
    shared_sectors = _as_set(sector_codes)

    # Per-shock filters (override if provided)
    sp_countries = _as_set(supply_country_codes) if supply_country_codes is not None else shared_countries
    sp_sectors = _as_set(supply_sector_codes) if supply_sector_codes is not None else shared_sectors

    sd_countries = _as_set(demand_country_codes) if demand_country_codes is not None else shared_countries
    sd_sectors = _as_set(demand_sector_codes) if demand_sector_codes is not None else shared_sectors

    n = len(node_labels)
    sp = np.zeros(n, dtype=float)
    sd = np.zeros(n, dtype=float)

    # Pre-parse labels once (faster and validates early)
    parsed: List[NodeKey] = [_parse_node_label(lbl) for lbl in node_labels]

    def _match(key: NodeKey, countries: Set[str], sectors: Set[str]) -> bool:
        country_match = (len(countries) == 0) or (key.country in countries)
        sector_match = (len(sectors) == 0) or (key.sector in sectors)
        return country_match and sector_match

    # Apply supply shocks
    if sp_frac > 0.0:
        for i, key in enumerate(parsed):
            if _match(key, sp_countries, sp_sectors):
                sp[i] = sp_frac

    # Apply demand shocks
    if sd_frac > 0.0:
        for i, key in enumerate(parsed):
            if _match(key, sd_countries, sd_sectors):
                sd[i] = sd_frac

    return sd, sp

