# Requirement: Unified Linkage Visualization

## Functional Requirements
1. Merge “Weakened” and “Strengthened” linkage charts into a single unified comparative visualization.
2. Visualization must clearly distinguish direction/sign (negative vs positive linkage change).
3. Users must be able to sort and filter within the unified view.
4. The chart must preserve current explanatory metadata (sector names, magnitude, scenario context).

## Technical Constraints
- Data model should represent linkage deltas in one normalized schema with sign semantics.
- Existing callback/filter interfaces must continue to work with minimal API changes.
- Visual encoding should avoid ambiguity when magnitudes are similar but signs differ.

## Acceptance Criteria
- Only one linkage chart is presented, containing both strengthened and weakened relationships.
- Analysts can identify direction and magnitude without switching views.
- Filtering/sorting behavior matches or improves existing functionality.
- No duplicate or missing linkage records in merged output.
