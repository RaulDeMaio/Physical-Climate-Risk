# Physical Climate Risk Propagation Model (MRIO)

This repository contains a **multi-regional input–output (MRIO) physical risk propagation model**
implemented in Python.

The model simulates the propagation of **simultaneous capacity (supply) shocks** and **final demand shocks**
through EU-27 country–sector production networks, including a **within-sector trade reallocation mechanism**
parameterised by \(\gamma\).

The codebase is structured as a small scientific software package under `src/`, plus:

- `main.ipynb`: reference notebook for running scenarios and producing diagnostics.
- `app.py`: optional Streamlit demo application (Databricks Connect required).

## High-level method

At a high level the workflow is:

1. Load a Eurostat Social Accounting Matrix (SAM) in long format from a Databricks catalog table.
2. Extract the IO blocks required by the model:
   - Production-to-production intermediate matrix \(Z\) (only `P_*` accounts).
   - Final demand vector \(FD\) (accounts `HH`, `GOV`, `CF`, `WRL_REST`).
   - Gross output \(X\) as \(X = \sum_j Z_{ij} + FD_i\).
   - Technical coefficients \(A\) from the SAM precomputed `share` column.
   - A mapping from each node to its global sector (same `P_*` across countries).
3. Define scenario shocks and run the model.
4. Aggregate and visualise output impacts (country, sector, and country×sector).

## Model iteration logic (key point)

The model performs **outer iterations on demand only**.

- Supply capacity is shocked once: \(X_{cap} = X_0 (1 - sp)\).
- A baseline Leontief inverse \(L_0\) is used to compute a *demand-driven desired output*:
  \(X_{dem} = L_0 \, FD_{post}\).
- A single propagation step constructs a reallocated intermediate matrix \(Z_{new}\).
- A global feasibility constraint (fixed global technology \(A_G\)) produces a feasible output \(X_s\).
- An implied final demand \(FD_{impl}\) is computed from accounting:
  \(FD_{impl} = \max(X_s - \sum_j Z_{new,ij}, 0)\).
- If \(FD_{impl} < FD_{post}\) for some nodes, demand is reduced **elementwise, monotonically**:
  \(FD_{post}^{k+1} = \min(FD_{post}^{k}, FD_{impl}^{k})\).

Convergence is checked on the relative change of \(FD_{post}\).

This design prevents recursive “collapse” dynamics caused by re-feeding the constrained economy
into subsequent iterations while still capturing the endogenous demand contraction implied by
binding supply constraints.

## Repository structure

```
.
├─ src/
│  ├─ data_io/
│  │  └─ eurostat_sam.py        # load SAM & extract model blocks
│  └─ io_climate/
│     ├─ model.py               # IOClimateModel
│     ├─ propagation.py         # propagate_once (allocation + reallocation)
│     └─ scenarios.py           # scenario helpers (shock vectors)
├─ main.ipynb                   # reference notebook for scenarios + diagnostics
├─ app.py                       # Streamlit demo (optional)
└─ physical risks through input-output linkages.pdf
```

## Quickstart (notebook)

1. Ensure Databricks Connect is configured in your VS Code environment.
2. Open and run `main.ipynb`.

The notebook is organised to:

- Load the latest SAM year.
- Build \(Z, FD, X, A,\) and labels.
- Define a scenario (country/sector selection + shock sizes).
- Run the model and create maps and bar charts for impacts.

## Quickstart (Streamlit app)

If you have Databricks Connect configured:

```bash
streamlit run app.py
```

## Data assumptions

- The SAM table is expected to contain at least:
  `c_orig`, `ind_ava`, `c_dest`, `ind_use`, `value`, `share`, `time_period`.
- The `share` column must represent \(Z_{ij} / X_j\) for production-to-production flows.

## License / internal usage

This repository is intended for internal analytical use.
