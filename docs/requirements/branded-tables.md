# Requirement: Branded Tables in Data Explorer

## Functional Requirements
1. The Data Explorer tables must apply OE brand tokens from `design_tokens.json` for header background, header text color, row striping, and border style.
2. Styling must be applied consistently across all tabular outputs in the Streamlit app, including filtered and sorted views.
3. The theme must support light and dark mode fallbacks when brand tokens are missing.
4. Numeric alignment must be right-aligned, textual columns left-aligned, and key metric columns visually emphasized.
5. Table export (CSV/XLSX) must preserve data values without injecting style artifacts.

## Technical Constraints
- Implement using Streamlit-compatible table components (`st.dataframe`, `st.data_editor`, or custom HTML/CSS wrappers) without breaking virtualization for large datasets.
- No hard-coded hex colors in Python modules; token lookup must be centralized.
- Styling logic must be decoupled from business logic (e.g., a dedicated UI styling utility module).
- Must remain compatible with Databricks Apps runtime and current dependency versions.

## Acceptance Criteria
- All Data Explorer tables visibly reflect OE branding (header, striping, typography hierarchy).
- No regression in sorting/filtering performance on representative datasets.
- Visual style degrades gracefully when a token is unavailable.
- Automated/UI smoke check confirms data values are unchanged before vs. after styling.
