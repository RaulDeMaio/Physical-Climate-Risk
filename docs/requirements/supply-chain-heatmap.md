# Requirement: Supply Chain Heatmap Visualization

## Functional Requirements
1. Add a dedicated heatmap view in the Supply Chain area.
2. Heatmap must visualize both absolute and percentage change perspectives, selectable by user toggle.
3. Matrix axes must represent source-target sector/country dimensions consistent with model outputs.
4. Support zoom or drill features for large matrices, with readable labels and value tooltips.

## Technical Constraints
- Use data transformations that avoid dense matrix recomputation for every UI interaction.
- Must handle sparse/zero-heavy matrices without misleading color scaling.
- Color scale and normalization must be documented and deterministic.
- Must remain compatible with Streamlit rendering limitations and Databricks Apps runtime.

## Acceptance Criteria
- New heatmap tab is accessible and functional.
- Users can switch between absolute and percentage change without reload failures.
- Tooltips provide cell-level values with correct formatting.
- Controlled dataset test confirms axis mapping and value correctness.
