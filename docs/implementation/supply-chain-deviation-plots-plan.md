# Supply Chain Deviation Plots — Realigned Plan

## Scope correction
This implementation follows the requirement that **deviation = distance from scenario mean relative sectoral impact**, consistent with map analytics, not model-iteration history deltas.

## Phase 1: Statistical strategy

### Deviation definition (aligned to requirements)
- Baseline metric for sector anomalies: `loss_pct` from post-processing.
- Sector deviation metric:
  - `loss_pct_deviation = loss_pct - mean(loss_pct)`
- This is the same mean-centered logic used by map deviation (`compute_relative_deviation`).

### Ranking and bucketing strategy
- No synthetic time bucketing is used for this requirement.
- Ranking modes:
  - `top_bottom`: at most 20 positive and 20 negative anomalies.
  - `full_distribution`: all sectors sorted by deviation.

### Visual encoding
- Horizontal diverging bars centered on `x=0`.
- Negative and positive deviations use diverging colors.
- Hover includes both absolute and percentage loss values.

## Phase 2: TDD workflow
1. RED: tests for mean-centered deviation math, zero/null handling, top/bottom cap, and large-input performance.
2. GREEN: implemented reusable preparation + plotting functions.
3. REFACTOR: unified logic via generic mean-deviation helper and reused it for country map deviation.

## Phase 3: assumptions and constraints
- Deviation unit is percentage points (`pp`) around scenario average.
- UI trade-off: prioritizes anomaly readability and ranking over time-series animation.
- DAG alignment: remains in B3 lane and consumes post-processed sector metrics from upstream model output.
