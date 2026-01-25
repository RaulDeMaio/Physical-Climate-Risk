# src/io_climate/propagation.py

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def propagate_once(
    Z: np.ndarray,
    A: np.ndarray,
    globsec_of: np.ndarray,
    X_dem: np.ndarray,
    X_cap: np.ndarray,
    FD_post: np.ndarray,
    sp: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Execute one propagation + within-global-sector reallocation step.

    Conceptual overview
    -------------------
    Inputs represent a baseline production network and an exogenous post-shock
    environment:

    - Z (n x n): baseline intermediate flows (producer i -> user j)
    - A (n x n): baseline technical coefficients (producer i per unit of output of j)
    - FD_post (n,): post-shock final demand used in this outer demand-iteration step
    - X_dem (n,): "demand-only" gross output requirement (computed upstream as L0 @ FD_post)
    - X_cap (n,): fixed capacity constraint after supply shock (X0 * (1-sp))
    - sp (n,): supply shock fractions in [0,1] (used to cap deliverability)
    - gamma: in [0,1], fraction of feasible substitution / reallocation applied
    - globsec_of (n,): global sector id (same for all country-nodes of a sector)

    Algorithm structure (paper-consistent interpretation)
    -----------------------------------------------------
    1) Compute rationing factors at producer level:
           r_i = min(1, X_cap_i / X_dem_i)

    2) Compute bottleneck constraints per using sector j:
           s_j = min_{i: A_ij > 0} r_i

       NOTE: s is defined per-using-sector (columns), but we apply it row-wise as an
       approximation consistent with your original implementation.

    3) Build constrained intermediate flows:
           row_factor_i = min(s_i, 1 - sp_i)      (approximation)
           Z_con[i,:] = Z[i,:] * row_factor_i

    4) Build needed intermediate flows to satisfy the demand-only output requirement:
           Z_need[i,j] = A[i,j] * X_dem[j]

    5) Base allocation:
           Z_base = min(Z_con, Z_need)

       Interpretation: flows cannot exceed what is feasible under constraints, nor
       exceed what is strictly required by demand-only production needs.

    6) Extra demand for inputs relative to constrained economy:
           E = max(Z_need - Z_con, 0)

    7) Inventories at producer i:
           inv_i = max(X_cap_i - FD_post_i - sum_j Z_con[i,j], 0)

    8) Aggregate inventories and extra demand by global sector and compute a
       substitution ratio:
           sub_s = min(1, Inv_sec_s / Extra_sec_s)

    9) Apply within-global-sector reallocation:
           frac_s = gamma * sub_s
           delivered_s_j = frac_s * Extra_sec_j[s, j]      (across users j)

       Delivered flows are distributed to producers i within the same global sector
       proportionally to their inventories (inv_share).

       Z_new = Z_base + delta_Z (reallocation increment)

    10) Local supply-side accounting output:
           X_supply_local = FD_post + row_sum(Z_new)

       (Global feasibility constraint is applied upstream in the model class.)

    Returns
    -------
    Z_new : (n, n) ndarray
        Intermediate matrix after within-global-sector reallocation.
    X_supply_local : (n,) ndarray
        Gross output implied by FD_post and delivered intermediate sales.
    aux : dict
        Diagnostics useful for debugging and scientific validation.
    """
    # ---- Basic validation / shaping ----
    Z = np.asarray(Z, dtype=float)
    A = np.asarray(A, dtype=float)
    globsec_of = np.asarray(globsec_of, dtype=int).reshape(-1)

    X_dem = np.asarray(X_dem, dtype=float).reshape(-1)
    X_cap = np.asarray(X_cap, dtype=float).reshape(-1)
    FD_post = np.asarray(FD_post, dtype=float).reshape(-1)
    sp = np.asarray(sp, dtype=float).reshape(-1)

    if Z.ndim != 2 or Z.shape[0] != Z.shape[1]:
        raise ValueError("Z must be a square (n x n) matrix.")
    n = Z.shape[0]

    if A.shape != (n, n):
        raise ValueError("A must have the same shape as Z (n x n).")
    if globsec_of.shape[0] != n:
        raise ValueError("globsec_of must have length n.")
    if X_dem.shape[0] != n or X_cap.shape[0] != n or FD_post.shape[0] != n or sp.shape[0] != n:
        raise ValueError("X_dem, X_cap, FD_post, sp must all have length n.")

    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0,1].")
    if (sp < 0).any() or (sp > 1).any():
        raise ValueError("sp must be within [0,1].")

    S = int(globsec_of.max()) + 1

    # ------------------------------------------------------------------
    # 1) Rationing factors r_i = min(1, X_cap_i / X_dem_i)
    # ------------------------------------------------------------------
    r = np.ones(n, dtype=float)
    mask_dem = X_dem > 0.0
    r[mask_dem] = np.minimum(1.0, X_cap[mask_dem] / X_dem[mask_dem])

    # ------------------------------------------------------------------
    # 2) Bottleneck constraints per using sector j:
    #    s_j = min_{i: A_ij > 0} r_i
    # ------------------------------------------------------------------
    s = np.ones(n, dtype=float)
    for j in range(n):
        suppliers = A[:, j] > 0.0
        if suppliers.any():
            s[j] = r[suppliers].min()
        else:
            s[j] = 1.0

    # ------------------------------------------------------------------
    # 3) Constrained intermediate flows Z_con (row-wise scaling)
    #    Z_con[i,:] = Z[i,:] * min(s_i, 1 - sp_i)
    # ------------------------------------------------------------------
    # NOTE: s is defined per-using-sector (columns). We keep your existing
    # approximation by applying s as if it were row-wise, then combining with (1-sp).
    row_factor = np.minimum(s, 1.0 - sp)
    Z_con = row_factor[:, None] * Z

    # ------------------------------------------------------------------
    # 4) Needed flows Z_need to produce X_dem under baseline coefficients
    #    Z_need[i,j] = A[i,j] * X_dem[j]
    # ------------------------------------------------------------------
    Z_need = A * X_dem[None, :]

    # ------------------------------------------------------------------
    # 5) Base allocation: min between constrained and needed
    # ------------------------------------------------------------------
    Z_base = np.minimum(Z_con, Z_need)

    # ------------------------------------------------------------------
    # 6) Extra demand (input shortages) relative to constrained economy
    # ------------------------------------------------------------------
    E = np.maximum(Z_need - Z_con, 0.0)

    # ------------------------------------------------------------------
    # 7) Inventories at producer i
    #    inv_i = max(X_cap_i - FD_post_i - sum_j Z_con[i,j], 0)
    # ------------------------------------------------------------------
    interm_sales_con = Z_con.sum(axis=1)
    inv = np.maximum(X_cap - FD_post - interm_sales_con, 0.0)

    # ------------------------------------------------------------------
    # 8) Aggregate inventories and extra demand by global sector
    # ------------------------------------------------------------------
    Inv_sec = np.zeros(S, dtype=float)
    Extra_sec = np.zeros(S, dtype=float)
    Extra_sec_j = np.zeros((S, n), dtype=float)  # per global sector s, across users j

    for i in range(n):
        s_id = globsec_of[i]
        Inv_sec[s_id] += inv[i]
        row_E = E[i, :]
        Extra_sec[s_id] += row_E.sum()
        Extra_sec_j[s_id, :] += row_E

    # ------------------------------------------------------------------
    # 9) Substitution ratios sub_s = min(1, Inv_sec / Extra_sec)
    # ------------------------------------------------------------------
    sub = np.zeros(S, dtype=float)
    mask_sec = Extra_sec > 0.0
    sub[mask_sec] = np.minimum(1.0, Inv_sec[mask_sec] / Extra_sec[mask_sec])

    # ------------------------------------------------------------------
    # 10) γ-based inventories reallocation within each global sector
    # ------------------------------------------------------------------
    Z_new = Z_base.copy()

    for s_id in range(S):
        if Extra_sec[s_id] <= 0.0 or Inv_sec[s_id] <= 0.0:
            continue

        frac = gamma * sub[s_id]
        if frac <= 0.0:
            continue

        # producers i belonging to global sector s_id
        i_idx = np.where(globsec_of == s_id)[0]
        inv_i = inv[i_idx]
        total_inv_i = inv_i.sum()
        if total_inv_i <= 0.0:
            continue

        inv_share = inv_i / total_inv_i  # distributes delivered flows across producers

        # total extra demand by users j for this global sector
        delivered_s_j = frac * Extra_sec_j[s_id, :]  # (n,)

        # allocate deliveries to producers proportionally to inventories
        delta_Z_block = np.outer(inv_share, delivered_s_j)  # (len(i_idx), n)
        Z_new[i_idx, :] += delta_Z_block

    # ------------------------------------------------------------------
    # 11) Local supply-side accounting output
    # ------------------------------------------------------------------
    X_supply_local = FD_post + Z_new.sum(axis=1)

    aux: Dict[str, Any] = {
        "X_dem": X_dem,
        "X_cap": X_cap,
        "FD_post": FD_post,
        "r": r,
        "s": s,
        "row_factor": row_factor,
        "Z_con": Z_con,
        "Z_need": Z_need,
        "Z_base": Z_base,
        "E": E,
        "inv": inv,
        "Inv_sec": Inv_sec,
        "Extra_sec": Extra_sec,
        "sub": sub,
        # A few helpful scalars for quick debugging
        "tot_Z_con": float(Z_con.sum()),
        "tot_Z_need": float(Z_need.sum()),
        "tot_Z_new": float(Z_new.sum()),
        "tot_inv": float(inv.sum()),
        "tot_extra": float(E.sum()),
    }

    return Z_new, X_supply_local, aux
