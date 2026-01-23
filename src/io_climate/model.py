# src/io_climate/model.py

from typing import Dict, Any, Optional, Tuple, Sequence
import numpy as np

from .propagation import propagate_once
from .scenarios import make_shock_vectors


class IOClimateModel:
    """
    Global input–output climate-risk propagation model with
    simultaneous demand & capacity shocks and within-sector
    trade reallocation (γ).

    Core objects (n = #country–sectors):
        Z : (n, n) intermediate-use matrix (producer i -> user j)
        FD: (n,) final demand vector
        X : (n,) gross output vector
        A : (n, n) technical coefficients matrix
        L : (n, n) Leontief inverse
        globsec_of : (n,) global sector id for each row i
        node_labels : list/sequence of "CC::P_XXX" labels, length n

    Additional global-technology object:
        A_G : (S, n) global-sector technical coefficients (fixed),
              where S = #global sectors.
              A_G[s, j] = Z_G[s, j] / X[j]
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
        """
        Parameters
        ----------
        Z : np.ndarray
            Intermediate-use matrix (n x n).
        FD : np.ndarray
            Final demand vector (n,).
        X : np.ndarray
            Gross output vector (n,).
        globsec_of : np.ndarray
            Length-n vector with global-sector id for each country–sector.
        A : np.ndarray, optional
            Technical coefficients matrix A[i,j] = Z[i,j] / X[j].
            If None, it is computed.
        L : np.ndarray, optional
            Leontief inverse (I - A)^(-1). If None, it is computed.
        """
        Z = np.asarray(Z, dtype=float)
        FD = np.asarray(FD, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float).reshape(-1)
        globsec_of = np.asarray(globsec_of, dtype=int).reshape(-1)

        if Z.shape[0] != Z.shape[1]:
            raise ValueError("Z must be a square (n x n) matrix.")
        n = Z.shape[0]

        if FD.shape[0] != n or X.shape[0] != n or globsec_of.shape[0] != n:
            raise ValueError("FD, X, globsec_of must all have length n = Z.shape[0].")

        self.Z = Z
        self.FD = FD
        self.X = X
        self.globsec_of = globsec_of
        self.n = n

        # node labels
        if node_labels is not None:
            if len(node_labels) != n:
                raise ValueError("node_labels must have length n.")
            self.node_labels = list(node_labels)
        else:
            self.node_labels = [f"node_{i}" for i in range(n)]

        # Node-level technical coefficients
        if A is None:
            A = self._compute_technical_coefficients(Z, X)
        else:
            A = np.asarray(A, dtype=float)
            if A.shape != Z.shape:
                raise ValueError("A must have the same shape as Z.")
        self.A = A

        # Baseline Leontief inverse (only needed for first-iteration X_dem)
        if L is None:
            L = self._compute_leontief_inverse(A)
        else:
            L = np.asarray(L, dtype=float)
            if L.shape != Z.shape:
                raise ValueError("L must have the same shape as Z.")
        self.L = L

        # Global sectors
        self.S_glob = int(globsec_of.max()) + 1

        # Baseline global-sector aggregation Z_G (S_glob x n)
        ZG0 = self._aggregate_to_global(Z)

        # Fixed global-sector technology A_G from baseline Z_G and baseline X
        # A_G[s, j] = ZG0[s, j] / X[j]
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
        max_iter: int = 100,
        tol: float = 1e-6,
        return_history: bool = False,
    ) -> Dict[str, Any]:
        
        """
        Run the IO climate-risk propagation model with iteration.

        Two usage modes
        ----------------
        1) Vector mode:
            - Provide sd and sp explicitly (length n).
        2) Scenario mode:
            - Leave sd and sp as None.
            - Provide country_codes, sector_codes, supply_shock_pct, demand_shock_pct.

        Iteration logic
        ----------------
        Inputs:
            FD_post = FD * (1 - sd)    # desired post-shock final demand (fixed)
            X_cap^0 = X * (1 - sp)     # initial capacity after supply shock
            Z_curr^0 = Z               # baseline flows
            A_curr^0 = A               # baseline technology

        For t = 1:
            X_dem^1 = L @ FD_post      # demand-only IO propagation
            sp_used = sp

        For t >= 2:
            X_dem^t = FD_post + row_sum(Z_curr^{t-1})
            sp_used = 0                # no further supply shocks

        One-step propagation:
            Z_new^t, X_supply_local^t = propagate_once(
                Z_curr^t-1, A_curr^t-1, globsec_of, X_dem^t, X_cap^t-1, FD_post, sp_used, γ
            )

        Global feasibility:
            ZG_new^t  = aggregate_rows(Z_new^t)
            X_glob^t_j = min_s ZG_new^t[s,j] / A_G[s,j]
            X_s^t      = min(X_supply_local^t, X_glob^t)  (element-wise)

        Implied demand:
            FD_implied^t = max( X_s^t - row_sum(Z_new^t), 0 )

        Convergence:
            - demand_gap = ||FD_post - FD_implied^t|| / ||FD_post||
            - output_gap = ||X_s^t - (FD_post + row_sum(Z_new^t))|| / ||FD_post + row_sum(Z_new^t)||

            Stop if max(demand_gap, output_gap) < tol.

        Next iteration:
            Z_curr^t   = Z_new^t
            A_curr^t   = Z_new^t / X_s^t
            X_cap^t    = X_s^t
        """
        # Decide how to obtain sd, sp
        if sd is None and sp is None:
            # Scenario mode
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

       # Post-shock demand (fixed desired final demand)
        FD_post = self.FD * (1.0 - sd)

        # Initial capacity: supply-shocked output
        X_cap = self.X * (1.0 - sp)

        # Initial flows and node-level technology
        Z_curr = self.Z.copy()
        A_curr = self.A.copy()

        X_supply_history = []
        FD_implied_history = []

        converged = False
        demand_gap = np.inf
        output_gap = np.inf
        aux: Dict[str, Any] = {}

        eps = 1e-12

        for it in range(1, max_iter + 1):
            # 1) Desired output X_dem
            if it == 1:
                # First iteration: standard IO propagation of FD_post
                X_dem = self.L @ FD_post
                sp_used = sp
            else:
                # Later iterations: desired output = post-shock FD + current intermediates
                X_dem = FD_post + Z_curr.sum(axis=1)
                sp_used = np.zeros_like(sp)

            # 2) One-step propagation (local bottlenecks + γ)
            Z_new, X_supply_local, aux = propagate_once(
                Z=Z_curr,
                A=A_curr,
                globsec_of=self.globsec_of,
                X_dem=X_dem,
                X_cap=X_cap,
                FD_post=FD_post,
                sp=sp_used,
                gamma=gamma,
            )


            if return_history:
                X_supply_history.append(X_supply_local.copy())


            # 3) Global aggregation and feasibility
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

            # Realized feasible supply this iteration
            X_s = np.minimum(X_supply_local, X_supply_global)

            # TEMPORARY: bypass global feasibility, use only local propagation output
            # X_s = X_supply_local


            # 4) Implied final demand from flows and X_s
            FD_implied = X_s - Z_new.sum(axis=1)
            FD_implied = np.maximum(FD_implied, 0.0)

            if return_history:
                FD_implied_history.append(FD_implied.copy())

            # 5) Convergence on demand and output
            FD_norm = np.linalg.norm(FD_post, ord=1) + eps
            demand_gap = np.linalg.norm(FD_post - FD_implied, ord=1) / FD_norm

            X_target = FD_post + Z_new.sum(axis=1)
            X_target_norm = np.linalg.norm(X_target, ord=1) + eps
            output_gap = np.linalg.norm(X_s - X_target, ord=1) / X_target_norm

            if max(demand_gap, output_gap) < tol:
                converged = True
                Z_curr = Z_new
                A_curr = self._compute_technical_coefficients(Z_new, X_s)
                X_cap = X_s
                break

            # 6) Prepare next iteration
            Z_curr = Z_new
            A_curr = self._compute_technical_coefficients(Z_new, X_s)
            X_cap = X_s

         # After loop: update model internal state
        self.Z = Z_curr
        self.A = A_curr
        self.L = self._compute_leontief_inverse(A_curr)
        self.X = X_cap  # last feasible X_s

        results: Dict[str, Any] = {
            "X_supply_final": self.X,
            "FD_implied_final": FD_implied,
            "FD_post": FD_post,
            "Z_final": self.Z,
            "A_final": self.A,
            "L_final": self.L,
            "iterations": it,
            "converged": converged,
            "demand_gap_last": demand_gap,
            "output_gap_last": output_gap,
            "aux_last": aux,
        }

        if return_history:
            results["X_supply_history"] = X_supply_history
            results["FD_implied_history"] = FD_implied_history

        return results

    # ------------------------------------------------------------------ #
    # Internal helper methods
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_technical_coefficients(
        Z: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute A[i,j] = Z[i,j] / X[j], handling zero outputs safely.
        """
        Z = np.asarray(Z, dtype=float)
        X = np.asarray(X, dtype=float).reshape(-1)

        A = np.zeros_like(Z, dtype=float)
        denom = X.copy()
        denom[denom == 0.0] = np.nan
        A = Z / denom[None, :]
        A = np.nan_to_num(A, nan=0.0)
        return A

    @staticmethod
    def _compute_leontief_inverse(A: np.ndarray) -> np.ndarray:
        """
        Compute the Leontief inverse L = (I - A)^(-1).
        """
        A = np.asarray(A, dtype=float)
        n = A.shape[0]
        I = np.eye(n)
        try:
            L = np.linalg.inv(I - A)
        except np.linalg.LinAlgError as err:
            raise ValueError("Leontief inverse (I - A)^(-1) is not invertible.") from err
        return L

    def _aggregate_to_global(self, Z: np.ndarray) -> np.ndarray:
        """
        Aggregate row-wise intermediate flows Z (n x n)
        to global sectors: Z_G (S_glob x n).
        """
        Z = np.asarray(Z, dtype=float)
        if Z.shape[0] != self.n or Z.shape[1] != self.n:
            raise ValueError("Z must be (n x n) to aggregate to global sectors.")

        ZG = np.zeros((self.S_glob, self.n), dtype=float)
        for i in range(self.n):
            s_id = self.globsec_of[i]
            ZG[s_id, :] += Z[i, :]
        return ZG
