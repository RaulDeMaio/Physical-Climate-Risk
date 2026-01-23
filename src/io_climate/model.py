# src/io_climate/model.py

from __future__ import annotations

from typing import Dict, Any, Optional, Sequence
import numpy as np

from .propagation import propagate_once
from .scenarios import make_shock_vectors


class IOClimateModel:
    """
    Global input–output climate-risk propagation model.

    Core (node-level) objects (n = #country–sectors):
        Z0 : (n, n) intermediate-use matrix (producer i -> user j)
        FD0: (n,)   final demand vector
        X0 : (n,)   gross output vector
        A0 : (n, n) technical coefficients matrix
        L0 : (n, n) Leontief inverse
        globsec_of : (n,) global sector id for each node i
        node_labels : length-n list of "CC::P_..." labels

    Fixed global-technology object (baseline):
        A_G : (S, n) aggregated technical coefficients by global sector s,
              where S = #global sectors and columns remain the n nodes.

    AGREED ITERATION LOGIC
    ----------------------
    The model iterates ONLY on post-shock demand (elementwise, monotone),
    while each outer iteration recomputes the economy from the SAME baseline
    state (Z0, A0, L0, X_cap0):

        FD_post^{k+1} = min(FD_post^{k}, FD_implied^{k})   (elementwise)

    Demand is never increased: if FD_implied > FD_post for a node, FD_post is kept.

    Convergence is checked on demand only, via relative L1 change in FD_post.
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
        Z = np.asarray(Z, dtype=float)
        FD = np.asarray(FD, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float).reshape(-1)
        globsec_of = np.asarray(globsec_of, dtype=int).reshape(-1)

        if Z.shape[0] != Z.shape[1]:
            raise ValueError("Z must be a square (n x n) matrix.")
        n = Z.shape[0]

        if FD.shape[0] != n or X.shape[0] != n or globsec_of.shape[0] != n:
            raise ValueError("FD, X, globsec_of must all have length n = Z.shape[0].")

        self.n = n
        self.globsec_of = globsec_of

        # Baseline objects (kept fixed across runs/iterations)
        self.Z0 = Z
        self.FD0 = FD
        self.X0 = X

        # Node labels
        if node_labels is not None:
            if len(node_labels) != n:
                raise ValueError("node_labels must have length n.")
            self.node_labels = list(node_labels)
        else:
            self.node_labels = [f"node_{i}" for i in range(n)]

        # Baseline A and L
        if A is None:
            A = self._compute_technical_coefficients(Z, X)
        else:
            A = np.asarray(A, dtype=float)
            if A.shape != Z.shape:
                raise ValueError("A must have the same shape as Z.")
        self.A0 = A

        if L is None:
            L = self._compute_leontief_inverse(A)
        else:
            L = np.asarray(L, dtype=float)
            if L.shape != Z.shape:
                raise ValueError("L must have the same shape as Z.")
        self.L0 = L

        # Global sector aggregation and fixed global technology A_G
        self.S_glob = int(globsec_of.max()) + 1
        ZG0 = self._aggregate_to_global(Z)
        self.A_G = self._compute_technical_coefficients(ZG0, X)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        sd: Optional[np.ndarray] = None,
        sp: Optional[np.ndarray] = None,
        *,
        country_codes=None,
        sector_codes=None,
        supply_shock_pct: float = 0.0,
        demand_shock_pct: float = 0.0,
        gamma: float = 0.5,
        max_iter: int = 50,
        tol: float = 1e-6,
        return_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the model with outer iteration on demand only.

        Two usage modes
        ----------------
        1) Vector mode:
            Provide sd and sp explicitly (length n).
        2) Scenario mode:
            Leave sd and sp as None and specify country_codes, sector_codes,
            supply_shock_pct, demand_shock_pct.

        Outer iteration (k = 1..max_iter)
        --------------------------------
        Fixed:
            Z_base = Z0,  A_base = A0,  L_base = L0
            X_cap0 = X0 * (1 - sp)   (fixed supply capacity after shock)

        Iterate:
            X_dem^k = L0 @ FD_post^k

            (Z_new^k, X_supply_local^k) = propagate_once(Z0, A0, X_dem^k, X_cap0, FD_post^k, sp, gamma)

            Apply global feasibility using fixed A_G:
                X_supply_global^k[j] = min_s ZG_new^k[s,j] / A_G[s,j]
                X_supply^k = min(X_supply_local^k, X_supply_global^k)

            FD_implied^k = max( X_supply^k - row_sum(Z_new^k), 0 )

            FD_post^{k+1} = min(FD_post^k, FD_implied^k)  (elementwise; no increases)

        Convergence:
            ||FD_post^{k+1} - FD_post^k||_1 / ||FD_post^k||_1 < tol
        """

        # Build shocks
        if sd is None and sp is None:
            sd, sp = make_shock_vectors(
                node_labels=self.node_labels,
                country_codes=country_codes,
                sector_codes=sector_codes,
                supply_shock_pct=supply_shock_pct,
                demand_shock_pct=demand_shock_pct,
            )
        elif (sd is None) != (sp is None):
            raise ValueError("Provide either both sd and sp, or neither.")

        sd = np.asarray(sd, dtype=float).reshape(-1)
        sp = np.asarray(sp, dtype=float).reshape(-1)

        if sd.shape[0] != self.n or sp.shape[0] != self.n:
            raise ValueError(f"sd and sp must both have length n = {self.n}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0,1].")

        # Fixed baseline objects
        Z_base = self.Z0
        A_base = self.A0
        L_base = self.L0
        X_cap0 = self.X0 * (1.0 - sp)

        # Initial post-shock demand
        FD_post = self.FD0 * (1.0 - sd)

        eps = 1e-12
        converged = False

        # History
        FD_post_hist = []
        FD_implied_hist = []
        X_supply_hist = []

        # Last-iteration outputs
        Z_new = np.zeros_like(Z_base)
        X_supply = np.zeros(self.n, dtype=float)
        FD_implied = np.zeros(self.n, dtype=float)
        aux_last: Dict[str, Any] = {}

        for it in range(1, max_iter + 1):
            # 1) Demand-only output requirement (baseline IO propagation)
            X_dem = L_base @ FD_post

            # 2) Propagation and reallocation from baseline state
            Z_new, X_supply_local, aux_last = propagate_once(
                Z=Z_base,
                A=A_base,
                globsec_of=self.globsec_of,
                X_dem=X_dem,
                X_cap=X_cap0,
                FD_post=FD_post,
                sp=sp,
                gamma=gamma,
            )

            # 3) Global feasibility constraint using fixed A_G
            ZG_new = self._aggregate_to_global(Z_new)  # (S_glob, n)

            X_supply_global = np.zeros(self.n, dtype=float)
            for j in range(self.n):
                a_col = self.A_G[:, j]
                z_col = ZG_new[:, j]
                mask = a_col > 0.0
                if mask.any():
                    caps = z_col[mask] / a_col[mask]
                    X_supply_global[j] = caps.min()
                else:
                    X_supply_global[j] = X_supply_local[j]

            X_supply = np.minimum(X_supply_local, X_supply_global)

            # 4) Implied final demand
            FD_implied = X_supply - Z_new.sum(axis=1)
            FD_implied = np.maximum(FD_implied, 0.0)

            if return_history:
                FD_post_hist.append(FD_post.copy())
                FD_implied_hist.append(FD_implied.copy())
                X_supply_hist.append(X_supply.copy())

            # 5) Elementwise monotone update (do not increase demand)
            FD_post_next = np.minimum(FD_post, FD_implied)

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
            "X_supply_global_last": X_supply_global,
            "X_supply_local_last": X_supply_local,
            "FD_post_final": FD_post,
            "FD_implied_final": FD_implied,
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
        """Compute A[i,j] = Z[i,j] / X[j], handling zero outputs safely."""
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
        I = np.eye(n)
        try:
            return np.linalg.inv(I - A)
        except np.linalg.LinAlgError as err:
            raise ValueError("Leontief inverse (I - A)^(-1) is not invertible.") from err

    def _aggregate_to_global(self, Z: np.ndarray) -> np.ndarray:
        """Aggregate row-wise Z (n x n) to global sectors: Z_G (S_glob x n)."""
        Z = np.asarray(Z, dtype=float)
        if Z.shape != (self.n, self.n):
            raise ValueError("Z must be (n x n) to aggregate.")
        ZG = np.zeros((self.S_glob, self.n), dtype=float)
        for i in range(self.n):
            ZG[self.globsec_of[i], :] += Z[i, :]
        return ZG
