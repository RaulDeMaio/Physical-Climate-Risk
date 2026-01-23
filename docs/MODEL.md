# Model documentation

This document describes the algorithm implemented in `src/io_climate`.

## Notation

Let **nodes** index country–sector production accounts:

- *i*: producing node (row of **Z**)
- *j*: using node (column of **Z**)
- *n*: number of nodes (countries × production sectors)

Core node-level objects:

- **Z** ∈ R^(n×n): intermediate-use matrix (producer *i* → user *j*)
- **FD** ∈ R^n: final demand vector
- **X** ∈ R^n: gross output vector
- **A** ∈ R^(n×n): technical coefficients, A[i,j] = Z[i,j] / X[j]
- **L** ∈ R^(n×n): Leontief inverse, L = (I - A)^(-1)

Global-sector aggregation:

- *s*: global sector id (same `P_*` across countries)
- *S*: number of global sectors
- `globsec_of[i]`: maps node *i* to its global sector *s*
- **Z_G** ∈ R^(S×n): global-sector aggregation of rows of **Z**
- **A_G** ∈ R^(S×n): fixed global technology coefficients computed from baseline **Z_G0**

Shocks:

- **sp** ∈ [0,1]^n: capacity shock (fraction of capacity lost)
- **sd** ∈ [0,1]^n: final demand shock (fraction of final demand lost)

## Inputs and extracted blocks

The model consumes IO-style blocks extracted from a Eurostat SAM:

- Production nodes are those whose account codes start with `P_`.
- The intermediate block uses only `P_* → P_*` flows.
- The final demand block is aggregated from `P_* → {HH, GOV, CF, WRL_REST}` flows.
- Technical coefficients **A** are taken from the SAM `share` column.

See `src/data_io/eurostat_sam.py`.

## Algorithm overview

The model has two nested components:

1. **Single-step propagation and reallocation** (`propagate_once`):
   given a baseline economy and a desired output vector, builds:
   - a constrained intermediate matrix Z_con under bottlenecks and capacity
   - a needed intermediate matrix Z_need under desired output
   - a reallocated matrix Z_new that interpolates between the two using inventories

2. **Outer iteration on demand only** (`IOClimateModel.run`):
   if the implied final demand consistent with Z_new and feasible supply is below
   the immediate post-shock demand, demand is reduced monotonically until stable.

### 1) Propagation and reallocation (`propagate_once`)

Given:

- baseline Z and A
- desired output X_dem
- capacity X_cap = X0 * (1 - sp)
- post-shock demand FD_post
- reallocation intensity gamma ∈ [0,1]

Steps:

1. **Rationing**: r_i = min(1, X_cap_i / X_dem_i)
2. **Bottlenecks**: for each using sector j, s_j = min_{i: A_ij>0} r_i
3. **Constrained allocation**: Z_con = Z scaled by min(s, 1 - sp)
4. **Needed allocation**: Z_need = A * X_dem (broadcast over rows)
5. **Base allocation**: Z_base = min(Z_con, Z_need)
6. **Extra demand**: E = max(Z_need - Z_con, 0)
7. **Inventories**: inv_i = max(X_cap_i - FD_post_i - sum_j Z_con[i,j], 0)
8. Aggregate inv and E by global sector, compute substitution ratio sub_s
9. **Reallocation**: distribute a fraction gamma * sub_s of extra demand across
   producers in the same global sector proportionally to inventories.

Output:

- Z_new: reallocated intermediate matrix
- X_supply_local: local gross output by accounting, FD_post + row_sum(Z_new)

### 2) Outer iteration on demand only (`IOClimateModel.run`)

Key design choice: the outer loop updates only **post-shock demand** and always
recomputes the economy from the same baseline state.

1. Build initial post-shock demand:

   FD_post^0 = FD0 * (1 - sd)

2. Build fixed capacity from supply shock:

   X_cap = X0 * (1 - sp)

3. For k = 0..K:

   a) Compute desired demand-driven output (baseline propagation):

      X_dem^k = L0 @ FD_post^k

   b) Run `propagate_once(Z0, A0, X_dem^k, X_cap, FD_post^k, sp, gamma)`
      → Z_new^k, X_supply_local^k

   c) Impose global feasibility using fixed A_G:

      X_supply_global^k[j] = min_s Z_G_new^k[s,j] / A_G[s,j]

      X_supply^k = min(X_supply_local^k, X_supply_global^k)

   d) Compute implied final demand from accounting:

      FD_impl^k = max(X_supply^k - row_sum(Z_new^k), 0)

   e) Monotone demand update (no increases):

      FD_post^{k+1} = min(FD_post^k, FD_impl^k)  (elementwise)

   f) Convergence check (relative L1 change in FD_post):

      ||FD_post^{k+1} - FD_post^k||_1 / ||FD_post^k||_1 < tol

## Diagnostics and outputs

The `run()` method returns:

- Z_final, X_supply_final
- FD_post_final, FD_implied_final
- global vs local supply decomposition (last iteration)
- histories (optional)

See `main.ipynb` for standard diagnostics: country maps, bar charts by country
and sector, and top impacted country–sectors.
