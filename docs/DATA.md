# Data notes

## Source tables

The reference pipeline expects a Eurostat Social Accounting Matrix (SAM) stored
in a Databricks catalog table (long format).

Default table (configurable in code):

- `openeconomics.gold.couind_eurostat_sam_y`

The model code assumes the table provides at least:

- `time_period` (year)
- `c_orig`, `ind_ava` (origin country and available account)
- `c_dest`, `ind_use` (destination country and use account)
- `value` (flow value)
- `share` (precomputed technical coefficient: value / output of the destination column)

## Account conventions

- **Production nodes** are account codes starting with `P_` (e.g. `P_C10-12`).
- **Intermediate block**: `P_* → P_*` flows are used to build the intermediate matrix **Z**.
- **Final demand accounts**: `HH`, `GOV`, `CF`, `WRL_REST`.
- **GDP accounts** (typically present in the SAM but excluded from **Z** and **FD**): `LAB`, `CAP`, `TAX`.

These conventions are implemented in:

- `src/data_io/eurostat_sam.py`

## Ordering and reproducibility

The extraction function builds a stable node ordering:

1. Create the set of production nodes observed either as origin or destination.
2. Sort nodes by `(country, sector)`.

This ordering is used to:

- Reindex pivot tables to a full n×n grid.
- Build `node_labels` and `globsec_of`.

If you change ordering rules, ensure you update all downstream assumptions
(e.g., scenario selection by `node_labels`).

## Practical validation checks

After extraction, the notebook runs sanity checks:

- shapes: `Z (n,n)`, `A (n,n)`, `FD (n,)`, `X (n,)`
- non-negativity: all blocks >= 0
- accounting: `X ≈ row_sum(Z) + FD` (by construction)

The model assumes the `share` column corresponds to `Z[i,j] / X[j]` for the
production-to-production block. If the SAM generation changes, validate this
assumption before interpreting results.
