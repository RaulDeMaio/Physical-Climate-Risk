# Requirement: Hazard Intensity Education Layer

## Functional Requirements
1. Provide tooltips and/or info-boxes describing intensity levels: moderate, severe, extreme, very_extreme.
2. Include concise explanation of calibration logic connecting historical losses to intensity levels.
3. Educational content must be visible near relevant controls and accessible without leaving workflow.
4. Content must be reusable across tabs where intensity appears.

## Technical Constraints
- Text content source should be centralized to avoid inconsistencies.
- UI element choice (tooltip, expander, modal, info panel) must remain Streamlit-native.
- Must support markdown formatting and future localization extension.
- Educational content updates should not require code changes to core model modules.

## Acceptance Criteria
- Users can access intensity definitions directly from the UI.
- Calibration explanation is present and understandable in context.
- No broken UI elements on Databricks Apps deployment target.
- Product review confirms reduced ambiguity in scenario selection.
