# B4 Supply Chain Heatmap — Implementation Report

## Phase 1: Implementation & Data Plan

### Data Transformation
- Source data comes from linkage deltas (`delta`, `delta_rel`) generated from baseline/final IO linkages in post-processing.
- We added a sparse full-linkage table (`df_links_all`) so the heatmap uses the complete source-target surface instead of only top-k link changes.
- Heatmap preparation flow:
  1. choose perspective: `absolute -> delta`, `percentage -> delta_rel`
  2. aggregate duplicate source-target pairs via groupby sum
  3. pivot to deterministic source-target matrix
  4. normalize intensity using deterministic quantile-capped magnitude scaling to `[0,1]`.

### Performance Strategy
- Avoid recomputation on UI toggles by precomputing `df_links_all` once during postprocess and only switching value column at render-time.
- Use sparse linkage filtering (`min_abs_delta`) to avoid carrying exact-zero deltas through the UI pipeline.
- Use Plotly heatmap trace (WebGL-backed rendering path where available) and Streamlit container-width rendering for responsive behavior.
- Normalization uses vectorized pandas/numpy operations (no Python loops over cells).

### Coordinate Mapping
- Implemented reusable coordinate-bound mapper for node overlays with two policies:
  - `clip`: clamp out-of-bounds nodes to viewport edge
  - `drop`: remove out-of-bounds nodes.
- Mapping outputs include `in_bounds` plus mapped coordinates (`x_mapped`, `y_mapped`) for downstream overlays/tooltips.

## Phase 2: TDD Execution

### Red
- Added tests for:
  - quantile-capped normalization behavior
  - empty/null datasets
  - viewport bounds clipping.

### Green
- Implemented `src/io_climate/supply_chain_heatmap.py`:
  - `normalize_intensity`
  - `build_heatmap_frame`
  - `map_coordinates_to_viewport`
- Implemented `plot_supply_chain_heatmap` and integrated two perspectives (absolute/percentage) into dashboard bundle.
- Added Supply Chain tab UI control to switch heatmap perspective.
- Extended postprocessing with `linkage_delta_table` and `df_links_all` to support full heatmap data binding.

### Refactor
- Kept transformations deterministic and vectorized.
- Reused precomputed tables in bundle to reduce interaction-time cost.
- Heatmap display remains responsive via Streamlit `use_container_width=True` and rerender on data updates.

## Phase 3: Final Notes

### Assumptions
- Color scale uses diverging `RdBu_r` with midpoint at zero.
- Heat intensity is based on `abs(value)` with 95th percentile capping to reduce outlier domination.
- Near-zero link changes are omitted from `df_links_all` by epsilon threshold.

### Performance Metrics (Stress Test)
Synthetic workload (`180,000` sparse linkage rows over `220x220` matrix):
- absolute frame build: `~0.201s`
- percentage frame build: `~0.168s`

### Integration Issues
- Real-time feed wiring is not yet available in this repo; current integration binds to model-run outputs in-memory.
- If future streaming introduces partial updates, an incremental matrix-update adapter should be added to avoid rebuilding from scratch.


## Revision: Meaningful Visualization Improvements
- Added **aggregation selector** (`sector`, `country`, `node`) so users can avoid unreadable dense node-level matrices and start from sector-level structure.
- Heatmap color rendering now uses **symmetric quantile clipping** (98th percentile of `abs(delta)`) to prevent outliers from flattening color contrast.
- Hover now shows both **raw value** and **displayed clipped value**, plus normalized intensity, so scaling decisions remain transparent.
- Axis labels are automatically hidden for very large matrices to reduce visual noise.
