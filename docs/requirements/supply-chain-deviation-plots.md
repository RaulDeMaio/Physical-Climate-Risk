# Requirement: Deviation-Based Supply Chain Plots

## Functional Requirements
1. Replace current standard supply-chain bar charts with horizontal plots based on deviation from mean sectoral impact.
2. Plot must show positive and negative deviations around a clearly marked zero reference line.
3. Ranking option must support top/bottom anomalous sectors and full distribution view.
4. Users must retain existing filters (country, hazard, intensity, sector scope).

## Technical Constraints
- Reuse the same deviation-definition logic as map analytics to maintain metric consistency.
- Chart rendering must remain performant for full sector lists.
- Accessibility: ensure labels remain readable and color encoding is distinguishable.
- Existing chart container contracts in Streamlit tabs must remain backward compatible.

## Acceptance Criteria
- Supply Chain tab shows horizontal deviation plots instead of previous baseline bars.
- Zero-centered layout is visible and correctly interpreted.
- Sector ordering aligns with selected ranking mode.
- Regression checks show filter interactions still function correctly.
