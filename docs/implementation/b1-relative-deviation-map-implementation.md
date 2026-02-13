# B1 Implementation Report: Relative Deviation Map

## Scope
This report documents the implementation of roadmap item **B1 (Relative Deviation Map)**, including the follow-up refinements requested during review.

---

## 1) Mathematical Strategy

### Deviation logic
The map uses the requirement-defined metric:

\[
\text{loss\_pct\_deviation}_i = \text{loss\_pct}_i - \operatorname{mean}(\text{loss\_pct})
\]

where the mean is computed across countries in the currently selected scenario context.

Implementation is centralized in:
- `compute_relative_deviation(df_country, metric='loss_pct', output_col='loss_pct_deviation')`

If the baseline column is missing, the function raises a clear `ValueError` so the failure mode is explicit and testable.

### Zero denominator handling
The selected requirement formula is a **difference-from-mean** metric (not a ratio), so denominator zero edge cases do not apply to B1's core metric.

---

## 2) Divergent Visual Strategy

### Color scale
The choropleth uses a diverging palette with a neutral midpoint at zero:
- Colorscale: `RdBu_r`
- Midpoint: `zmid=0.0`
- Range: symmetric around zero using `[-max_abs, +max_abs]`

This guarantees comparable color intensity for positive and negative deviations and aligns to the requirement for symmetric ranges when applicable.

### Legend and number formatting
As requested during review:
- Legend title is set to **"Loss % deviation"**.
- Legend tick formatting uses scientific notation (`.1e`) with `pp` suffix for compact readability (e.g., `1e-3 pp`).

### Hover detail preservation
The deviation map hover includes:
- Country (name + ISO2)
- Loss % deviation (pp)
- Loss impact (%)
- Loss impact (abs)

This preserves richer interpretability context while still emphasizing deviation as the mapped metric.

---

## 3) Data Sourcing and Processing Flow

### Baseline vs runtime data
- Runtime scenario impacts are produced in post-processing and exposed in `pp.df_country`.
- Baseline for B1 is the scenario mean of `loss_pct` over that same `pp.df_country` frame.

### Integration point
`build_dashboard_bundle(...)` defaults map rendering to `loss_pct_deviation` and routes map generation through the deviation pipeline.

This maintains a single metric contract for downstream components relying on deviation semantics.

---

## 4) TDD Execution Summary

### Red phase (tests first)
Added/expanded tests for:
1. Positive/negative/zero deviation values.
2. Extreme variance behavior with symmetric divergent bounds and fixed midpoint.
3. Graceful failure when baseline data is missing.
4. Legend/hover refinements (title, scientific notation, richer hover fields).

### Green phase (implementation)
Implemented:
- `compute_relative_deviation(...)`
- `_symmetric_color_range(...)`
- `plot_relative_deviation_map(...)`
- Dashboard bundle default wiring for deviation map

### Refactor/optimization notes
- Deviation logic is centralized in one helper to avoid semantic drift.
- Computation happens on the country-level table already produced by post-processing, keeping transformation cost low.
- Existing pipeline structure remains modular and test-friendly.

---

## 5) Assumptions

1. **Baseline scope**: mean of `loss_pct` across countries for the selected scenario state.
2. **Units**:
   - `loss_pct_deviation` is represented in **percentage points (pp)**.
   - `loss_abs` remains in the project's existing absolute monetary unit conventions.
3. **Missing optional hover fields**: if `loss_abs` is absent in an intermediate frame, hover gracefully carries `NaN` for that element rather than failing map rendering.

---

## 6) Visual Thresholds / Breakpoints

The implementation uses a continuous diverging scale with dynamic, scenario-relative bounds:
- Neutral point: `0.0 pp`
- Lower bound: `-max(|deviation|)`
- Upper bound: `+max(|deviation|)`

Interpretation guidance:
- Near zero (small absolute deviation): near-neutral tone.
- Large positive deviation: strong red side.
- Large negative deviation: strong blue side.

No fixed static cutoffs are hardcoded in B1; severity is contextualized to current scenario spread.

---

## 7) ADR/DAG Compliance Check

- Aligns with ADR pipeline principle: derived deviations are produced as shared analytics transforms before visualization.
- Preserves map centering behavior inherited from existing map rendering path.
- Establishes B1 metric contract required by B2/B3 in the DAG.
- Does **not** introduce circular dependency with B4 supply-chain heatmap; B1 remains an upstream metric concern for map/deviation features.

---

## 8) Validation Summary

Automated checks executed:
- `pytest -q` (full suite passing after changes)

Coverage includes formula correctness, divergent color behavior, missing baseline error handling, and presentation-layer formatting/hover expectations.
