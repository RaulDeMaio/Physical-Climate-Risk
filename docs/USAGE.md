# Usage guide

This guide outlines how to run the model from the notebook and from the
Streamlit app.

## Notebook workflow (recommended)

Open `main.ipynb` and run the cells in order.

Key steps:

1. **Load SAM and extract blocks**
   - `load_sam_latest_year(spark)` loads the latest available year.
   - `extract_model_inputs_from_sam(...)` returns:
     - `Z, FD, X, A, globsec_of, node_labels`

2. **Instantiate the model**

   ```python
   model = IOClimateModel(
       Z=Z,
       FD=FD,
       X=X,
       globsec_of=globsec_of,
       A=A,
       node_labels=node_labels,
   )
   ```

3. **Run a scenario**

   ```python
   results = model.run(
       country_codes=["IT"],
       sector_codes=["P_C10-12"],
       supply_shock_pct=5.0,
       demand_shock_pct=0.0,
       gamma=0.5,
       max_iter=100,
       tol=1e-6,
       return_history=True,
   )
   ```

4. **Visualise impacts**

   The notebook includes:
   - country-level choropleth maps
   - bar charts by country and by sector
   - top impacted country–sectors

## Streamlit app (optional)

The Streamlit app (`app.py`) provides a minimal UI for demonstration.

Prerequisites:

- Databricks Connect configured (VS Code Databricks extension or CLI)

Run:

```bash
streamlit run app.py
```

## Interpreting outputs

`model.run(...)` returns a dictionary with, at minimum:

- `Z_final`: reallocated intermediate matrix
- `X_supply_final`: feasible gross output after global feasibility constraint
- `FD_post_final`: final demand after monotone demand reductions
- `FD_implied_final`: final demand implied by accounting in the last iteration
- `iterations`, `converged`

For detailed algorithm notes, see `docs/MODEL.md`.
