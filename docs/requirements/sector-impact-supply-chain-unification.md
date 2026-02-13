# Requirement: Sector Impact + Supply Chain Tab Unification

## Functional Requirements
1. Replace separate “Sector Impact” and “Supply Chain” tabs with one integrated sectoral analysis workspace.
2. Unified workspace must include key visuals from both legacy tabs with coherent layout and navigation.
3. Shared scenario controls must synchronize all charts in the combined tab.
4. Contextual explanations should clarify how direct sector impacts relate to network propagation effects.

## Technical Constraints
- Refactor tab composition without altering core simulation outputs.
- Maintain acceptable load/render times by lazy-loading heavier visual components.
- Preserve URL/session-state behavior where applicable for user navigation continuity.
- Avoid duplicated computations across formerly separate tabs.

## Acceptance Criteria
- One combined tab replaces the two prior tabs in navigation.
- All critical analyses previously available remain accessible.
- Control interactions update all child charts consistently.
- Usability walkthrough confirms improved end-to-end sectoral analysis flow.
