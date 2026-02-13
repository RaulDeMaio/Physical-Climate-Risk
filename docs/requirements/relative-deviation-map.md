# Requirement: Relative Deviation Choropleth

## Functional Requirements
1. Choropleth coloring in the map view must use `loss_pct_deviation = loss_pct - mean(loss_pct)` for the selected scenario context.
2. The map legend must clearly indicate the metric as deviation from mean and display symmetric positive/negative ranges when applicable.
3. The color scale must visually separate below-average, near-average, and above-average impacts.
4. The map must support scenario reactivity (country, hazard, intensity, year where applicable) and recompute deviation dynamically.

## Technical Constraints
- Mean computation scope must be explicitly defined (e.g., across countries for selected scenario) and implemented consistently in one analytics function.
- Color mapping should use a diverging palette with a neutral midpoint at zero.
- Must preserve map centering behavior already configured for Europe.
- Avoid expensive recomputation by caching transformed dataframe artifacts when inputs are unchanged.

## Acceptance Criteria
- For any selected scenario, map colors correspond to deviation values, not raw `loss_pct`.
- Legend and tooltip labels explicitly reference deviation metric.
- Zero-deviation points are mapped to the palette midpoint.
- Validation test with controlled sample data confirms formula correctness.
