# src/io_climate/propagation.py

from typing import Dict, Any, Tuple
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
    Single propagation + reallocation step.

    AGREED LOGIC
    ------------
    - Build Z_con from bottlenecks and (1-sp).
    - Build Z_need from A and X_dem.
    - Base allocation is min(Z_con, Z_need).
    - Add gamma reallocation flows to this base.
    - Do NOT force-cap Z_new <= Z_need after reallocation (paper-consistent interpretation).

    Returns:
      Z_new, X_supply_local, aux
    """
    Z = np.asarray(Z, dtype=float)
    A = np.asarray(A, dtype=float)
    X_dem = np.asarray(X_dem, dtype=float).reshape(-1)
    X_cap = np.asarray(X_cap, dtype=float).reshape(-1)
    FD_post = np.asarray(FD_post, dtype=float).reshape(-1)
    sp = np.asarray(sp, dtype=float).reshape(-1)
    globsec_of = np.asarray(globsec_of, dtype=int).reshape(-1)

    n = Z.shape[0]
    if Z.shape != (n, n):
        raise ValueError("Z must be (n x n).")
    if A.shape != (n, n):
        raise ValueError("A must be (n x n).")
    if (
        X_dem.shape[0] != n
        or X_cap.shape[0] != n
        or FD_post.shape[0] != n
        or sp.shape[0] != n
        or globsec_of.shape[0] != n
    ):
        raise ValueError("All vectors must have length n.")

    S = int(globsec_of.max()) + 1

    # 1) Rationing factors r_i = min(1, X_cap_i / X_dem_i)
    r = np.ones(n, dtype=float)
    mask_dem = X_dem > 0.0
    r[mask_dem] = np.minimum(1.0, X_cap[mask_dem] / X_dem[mask_dem])

    # 2) Bottleneck constraints per using sector j:
    #    s_j = min_{i: A_ij>0} r_i
    s = np.ones(n, dtype=float)
    for j in range(n):
        suppliers = A[:, j] > 0.0
        if suppliers.any():
            s[j] = r[suppliers].min()
        else:
            s[j] = 1.0

    # 3) Constrained intermediate flows Z_con (row-wise scaling)
    #    Z_con[i,:] = Z[i,:] * min(s_i, 1 - sp_i)
    # NOTE: s is defined per-using-sector j; we keep your approximation and apply s
    # as a row-wise factor, combined with (1 - sp) as requested.
    row_factor = np.minimum(s, 1.0 - sp)
    Z_con = row_factor[:, None] * Z

    # 4) Needed flows Z_need given desired output X_dem (no bottlenecks)
    Z_need = A * X_dem[None, :]

    # 5) Base allocation: min between constrained and needed
    Z_base = np.minimum(Z_con, Z_need)

    # 6) Extra demand for inputs relative to constrained state
    E = np.maximum(Z_need - Z_con, 0.0)

    # 7) Inventories per producer:
    #    inv_i = max(X_cap_i - FD_post_i - sum_j Z_con[i,j], 0)
    interm_sales_con = Z_con.sum(axis=1)
    inv = np.maximum(X_cap - FD_post - interm_sales_con, 0.0)

    # 8) Aggregate by global sector: inventories & extra demand
    Inv_sec = np.zeros(S, dtype=float)
    Extra_sec = np.zeros(S, dtype=float)
    Extra_sec_j = np.zeros((S, n), dtype=float)

    for i in range(n):
        s_id = globsec_of[i]
        Inv_sec[s_id] += inv[i]
        row_E = E[i, :]
        Extra_sec[s_id] += row_E.sum()
        Extra_sec_j[s_id, :] += row_E

    # 9) Substitution ratios sub_s = min(1, Inv_sec / Extra_sec)
    sub = np.zeros(S, dtype=float)
    mask_sec = Extra_sec > 0.0
    sub[mask_sec] = np.minimum(1.0, Inv_sec[mask_sec] / Extra_sec[mask_sec])

    # 10) γ-based inventories reallocation within each global sector
    Z_new = Z_base.copy()

    for s_id in range(S):
        if Extra_sec[s_id] <= 0.0 or Inv_sec[s_id] <= 0.0:
            continue

        frac = gamma * sub[s_id]
        if frac <= 0.0:
            continue

        i_idx = np.where(globsec_of == s_id)[0]
        inv_i = inv[i_idx]
        total_inv_i = inv_i.sum()
        if total_inv_i <= 0.0:
            continue

        inv_share = inv_i / total_inv_i  # shape (len(i_idx),)

        extra_s_j = Extra_sec_j[s_id, :]     # shape (n,)
        delivered_s_j = frac * extra_s_j     # shape (n,)

        delta_Z_block = np.outer(inv_share, delivered_s_j)  # (len(i_idx), n)
        Z_new[i_idx, :] += delta_Z_block

    # 11) Local supply-side gross output from accounting
    X_supply_local = FD_post + Z_new.sum(axis=1)

    aux: Dict[str, Any] = {
        "X_dem": X_dem,
        "X_cap": X_cap,
        "r": r,
        "s": s,
        "Z_con": Z_con,
        "Z_need": Z_need,
        "Z_base": Z_base,
        "E": E,
        "inv": inv,
        "Inv_sec": Inv_sec,
        "Extra_sec": Extra_sec,
        "sub": sub,
    }

    return Z_new, X_supply_local, aux
