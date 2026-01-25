# src/io_climate/model.py

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .propagation import propagate_once
from .scenarios import make_shock_vectors


class IOClimateModel:
    """
    Multi-country, multi-sector physical risk propagation model.

    Core node-level objects (n = #country–sectors):
        Z0 : (n, n) intermediate-use matrix (producer i -> user j)
        FD0: (n,)   final demand vector
        X0 : (n,)   gross output vector
        A0 : (n, n) technical coefficients matrix
        L0 : (n, n) Leontief inverse
        globsec_of : (n,) global sector id for each node i
        node_labels: length-n list of "CC::P_..." labels

    Global-technology discipline (baseline, fixed across iterations):
        ZG0 : (S, n) row-aggregated intermediate matrix (S global sectors)
        A_G : (S, n) aggregated technical coefficients A_G[s,j] = ZG0[s,j] / X0[j]

    AGREED ITERATION LOGIC (OUTER LOOP ON DEMAND ONLY)
    --------------------------------------------------
    - Supply shock sp fixes capacity: X_cap0 = X0 * (1 - sp)
    - Initial post-shock demand is:   FD_post0 = FD0 * (1 - sd)

    For k = 1..max_iter:
        1) Demand-only required output (baseline IO propagation):
               X_dem^k = L0 @ FD_post^k

        2) Propagate + reallocate (from baseline Z0,A0, fixed X_cap0):
               (Z_new^k, X_supply_local^k) = propagate_once(Z0, A0, X_dem^k, X_cap0, FD_post^k, sp, gamma)

        3) Impose global feasibility (fixed A_G):
               ZG_new^k = aggregate_rows_by_global_sector(Z_new^k)
               X_supply_global^k[j] = min_s ZG_new^k[s,j] / A_G[s,j]   over A_G[s,j] > 0
               X_supply^k = min(X_supply_local^k, X_supply_global^k)

        4) Implied final demand (accounting identity with Z_new^k and feasible output):
               FD_implied^k = max(X_supply^k - row_sum(Z_new^k), 0)

        5) Elementwise monotone update (never increase demand):
               FD_post^{k+1} = min(FD_post^k, FD_implied^k)

        Convergence check (demand-only):
               ||FD_post^{k+1} - FD_post^k||_1 / ||FD_post^k||_1 < tol
    """

    def __init__(
        self,
        Z: np.ndarray,
        FD: np.ndarray,
        X: np.ndarray,
        globsec_of: np.ndarray,
        A: Optional[np.ndarray] = None,
        L: Optional[np.ndarray] = None,
        node_labels: Optional[Sequence[str]] = None,
    ) -> None:
        # ---- Validate and store baseline objects ----
        Z = np.asarray(Z, dtype=float)
        FD = np.asarray(FD, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float).reshape(-1)
        globsec_of = np.asarray(globsec_of, dtype=int).reshape(-1)

        if Z.ndim != 2 or Z.shape[0] != Z.shape[1]:
            raise ValueError("Z must be a square (n x n) matrix.")
        n = Z.shape[0]

        if FD.shape[0] != n or X.shape[0] != n or globsec_of.shape[0] != n:
            raise ValueError("FD, X, globsec_of must all have length n = Z.shape[0].")

        self.n = n
        self.globsec_of = globsec_of
        self.S_glob = int(globsec_of.max()) + 1

        # Labels (must align with node ordering used to build Z/FD/X)
        if node_labels is None:
            self.node_labels = [f"node_{i}" for i in range(n)]
        else:
            if len(node_labels) != n:
                raise ValueError("node_labels must have length n.")
            self.node_labels = list(node_labels)

        # Baseline objects (never mutated by run)
        self.Z0 = Z
        self.FD0 = FD
        self.X0 = X

        # Baseline A0 and L0
        if A is None:
            A0 = self._compute_technical_coefficients(Z, X)
        else:
            A0 = np.asarray(A, dtype=float)
            if A0.shape != (n, n):
                raise ValueError("A must have the same shape as Z (n x n).")
        self.A0 = A0

        if L is None:
            L0 = self._compute_leontief_inverse(A0)
        else:
            L0 = np.asarray(L, dtype=float)
            if L0.shape != (n, n):
                raise ValueError("L must have the same shape as Z (n x n).")
        self.L0 = L0

        # Fixed global technology A_G (from baseline)
        ZG0 = self._aggregate_to_global(self.Z0)  # (S, n)
        self.A_G = self._compute_technical_coefficients(ZG0, self.X0)  # (S, n)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        sd: Optional[np.ndarray] = None,
        sp: Optional[np.ndarray] = None,
        *,
        # Shared targeting (applies to both if per-shock overrides not provided)
        country_codes=None,
        sector_codes=None,
        # Per-shock targeting overrides
        supply_country_codes=None,
        supply_sector_codes=None,
        demand_country_codes=None,
        demand_sector_codes=None,
        # Shock sizes (%)
        supply_shock_pct: float = 0.0,
        demand_shock_pct: float = 0.0,
        # Propagation control
        gamma: float = 0.5,
        max_iter: int = 50,
        tol: float = 1e-6,
        return_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the model.

        Two usage modes
        --------------
        1) Vector mode:
            Provide sd and sp explicitly (length n each).
        2) Scenario mode:
            Leave sd and sp as None and specify targeting + shock sizes.

        Parameters
        ----------
        sd, sp
            Demand and supply shock vectors in [0,1], both length n.
            If None, they are built via make_shock_vectors.
        country_codes, sector_codes
            Shared targeting filters (used for both shocks unless overridden).
        supply_country_codes, supply_sector_codes, demand_country_codes, demand_sector_codes
            Optional per-shock targeting overrides.
        supply_shock_pct, demand_shock_pct
            Shock sizes in percent (0–100).
        gamma
            Reallocation strength in [0,1].
        max_iter, tol
            Outer-loop parameters for demand-only iteration.
        return_history
            If True, returns FD_post, FD_implied, and X_supply histories.

        Returns
        -------
        results : dict
            Keys:
                converged, iterations,
                Z_final, X_supply_final, X_supply_local_last, X_supply_global_last,
                FD_post_final, FD_implied_final,
                sd, sp,
                aux_last,
                and optionally histories.
        """
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0,1].")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")

        # ---- Build or validate shock vectors ----
        if sd is None and sp is None:
            sd, sp = make_shock_vectors(
                node_labels=self.node_labels,
                # shared filters
                country_codes=country_codes,
                sector_codes=sector_codes,
                # per-shock overrides
                supply_country_codes=supply_country_codes,
                supply_sector_codes=supply_sector_codes,
                demand_country_codes=demand_country_codes,
                demand_sector_codes=demand_sector_codes,
                # sizes
                supply_shock_pct=supply_shock_pct,
                demand_shock_pct=demand_shock_pct,
            )
        elif (sd is None) != (sp is None):
            raise ValueError("Provide either both sd and sp, or neither.")

        sd = np.asarray(sd, dtype=float).reshape(-1)
        sp = np.asarray(sp, dtype=float).reshape(-1)

        if sd.shape[0] != self.n or sp.shape[0] != self.n:
            raise ValueError(f"sd and sp must both have length n = {self.n}.")
        if (sd < 0).any() or (sd > 1).any():
            raise ValueError("sd must be within [0,1].")
        if (sp < 0).any() or (sp > 1).any():
            raise ValueError("sp must be within [0,1].")

        # ---- Fixed baseline / fixed capacity after supply shock ----
        X_cap0 = self.X0 * (1.0 - sp)

        # Initial post-shock demand
        FD_post = self.FD0 * (1.0 - sd)

        eps = 1e-12
        converged = False

        # Histories (optional)
        FD_post_hist = []
        FD_implied_hist = []
        X_supply_hist = []

        # Last-iteration outputs
        Z_new = np.zeros_like(self.Z0)
        X_supply = np.zeros(self.n, dtype=float)
        X_supply_local = np.zeros(self.n, dtype=float)
        X_supply_global = np.zeros(self.n, dtype=float)
        FD_implied = np.zeros(self.n, dtype=float)
        aux_last: Dict[str, Any] = {}

        for it in range(1, max_iter + 1):
            # 1) Demand-only required output (baseline Leontief)
            X_dem = self.L0 @ FD_post

            # 2) Propagate + reallocate from baseline state (Z0,A0), fixed X_cap0
            Z_new, X_supply_local, aux_last = propagate_once(
                Z=self.Z0,
                A=self.A0,
                globsec_of=self.globsec_of,
                X_dem=X_dem,
                X_cap=X_cap0,
                FD_post=FD_post,
                sp=sp,
                gamma=gamma,
            )

            # 3) Global feasibility constraint (fixed A_G)
            ZG_new = self._aggregate_to_global(Z_new)  # (S, n)

            for j in range(self.n):
                a_col = self.A_G[:, j]
                z_col = ZG_new[:, j]
                mask = a_col > 0.0
                if mask.any():
                    caps = z_col[mask] / a_col[mask]
                    X_supply_global[j] = float(np.min(caps))
                else:
                    # If no global inputs are required for this column (degenerate),
                    # fall back to local supply.
                    X_supply_global[j] = X_supply_local[j]

            X_supply = np.minimum(X_supply_local, X_supply_global)

            # 4) Implied final demand
            FD_implied = X_supply - Z_new.sum(axis=1)
            FD_implied = np.maximum(FD_implied, 0.0)

            if return_history:
                FD_post_hist.append(FD_post.copy())
                FD_implied_hist.append(FD_implied.copy())
                X_supply_hist.append(X_supply.copy())

            # 5) Elementwise monotone update: never increase demand
            FD_post_next = np.minimum(FD_post, FD_implied)

            # Convergence on demand only
            denom = np.linalg.norm(FD_post, ord=1) + eps
            demand_update_gap = np.linalg.norm(FD_post_next - FD_post, ord=1) / denom

            if demand_update_gap < tol:
                converged = True
                FD_post = FD_post_next
                break

            FD_post = FD_post_next

        results: Dict[str, Any] = {
            "converged": converged,
            "iterations": it,
            "Z_final": Z_new,
            "X_supply_final": X_supply,
            "X_supply_local_last": X_supply_local,
            "X_supply_global_last": X_supply_global,
            "FD_post_final": FD_post,
            "FD_implied_final": FD_implied,
            "sd": sd,
            "sp": sp,
            "aux_last": aux_last,
        }

        if return_history:
            results["FD_post_history"] = FD_post_hist
            results["FD_implied_history"] = FD_implied_hist
            results["X_supply_history"] = X_supply_hist

        return results

    # ------------------------------------------------------------------ #
    # Internal helper methods
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_technical_coefficients(Z: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute technical coefficients A[i,j] = Z[i,j] / X[j], safely handling X[j]=0.
        Shapes:
            Z: (m, n), X: (n,)  -> A: (m, n)
        """
        Z = np.asarray(Z, dtype=float)
        X = np.asarray(X, dtype=float).reshape(-1)

        if Z.shape[1] != X.shape[0]:
            raise ValueError("X length must match Z.shape[1] (columns).")

        denom = X.copy()
        denom[denom == 0.0] = np.nan
        A = Z / denom[None, :]
        return np.nan_to_num(A, nan=0.0)

    @staticmethod
    def _compute_leontief_inverse(A: np.ndarray) -> np.ndarray:
        """Compute Leontief inverse L = (I - A)^(-1)."""
        A = np.asarray(A, dtype=float)
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square to compute Leontief inverse.")
        I = np.eye(n)
        try:
            return np.linalg.inv(I - A)
        except np.linalg.LinAlgError as err:
            raise ValueError("Leontief inverse (I - A)^(-1) is not invertible.") from err

    def _aggregate_to_global(self, Z: np.ndarray) -> np.ndarray:
        """
        Aggregate Z by summing rows that belong to the same global sector id.

        Input:
            Z: (n, n)
        Output:
            ZG: (S_glob, n)
        """
        Z = np.asarray(Z, dtype=float)
        if Z.shape != (self.n, self.n):
            raise ValueError("Z must be (n x n) for global aggregation.")

        ZG = np.zeros((self.S_glob, self.n), dtype=float)
        for i in range(self.n):
            ZG[self.globsec_of[i], :] += Z[i, :]
        return ZG
