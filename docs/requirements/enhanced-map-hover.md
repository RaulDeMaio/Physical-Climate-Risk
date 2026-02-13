# Requirement: Enhanced Map Hover Tooltips

## Functional Requirements
1. Map hover tooltip must display both:
   - deviation-based value used for coloring, and
   - original raw `loss_pct` value.
2. Tooltip must include clear field labels and percentage formatting.
3. Tooltip ordering must prioritize interpretability: region, deviation metric, raw metric, optional metadata.
4. Tooltip values must update instantly on scenario changes.

## Technical Constraints
- Tooltip content must be generated from a single transformed dataset to avoid mismatch between color and displayed values.
- Number formatting utilities must be shared with global percent formatting conventions.
- Tooltip payload must not materially degrade map rendering performance.

## Acceptance Criteria
- Hovering any region shows both deviation and raw `loss_pct` simultaneously.
- Values match underlying dataframe values within rounding tolerance.
- No tooltip rendering errors for missing or null observations.
- UX review confirms improved interpretability for analysts.
