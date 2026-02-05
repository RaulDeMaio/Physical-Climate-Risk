# Future Improvements & Roadmap

This document outlines the planned enhancements for the Physical Climate Risk Propagation App, categorized by priority and timeframe.

## 🚀 Near Future (UI/UX & Visualization)

### Visual Enhancement
- **Value Labels**: Add explicit data labels (text values) on bar charts and maps for immediate readability.
- **Percentage Formatting**: Change all decimal percentages (e.g., `0.054`) to a human-readable format (e.g., `5.4%`).
- **Map Centering**: Adjust the default geographic projection to center on **Europe**.
- **Branded Tables**: Enhance the Data Explorer with OE-standard coloring (header backgrounds, row striping) as defined in `design_tokens.json`.

### Analytic Refinement
- **Relative Deviation Map**: Change the choropleth color scale from raw `loss_pct` to **"deviation from mean loss_pct"** to highlight outliers.
- **Enhanced Map Hover**: Include both the deviation-based color and the raw `loss_pct` as labels in the hover tooltips.
- **Deviation Plots**: Replace standard bar charts with horizontal **"deviation from mean"** plots to emphasize anomalous sectoral impacts.
- **Heatmap Visualization**: Implement a dedicated heatmap tab to visualize absolute vs. percentage changes across the entire input-output matrix.

### UX & Interface
- **Hazard Education**: Add tooltips or info-boxes explaining the different **Intensity** levels (moderate, severe, extreme, very_extreme) and the underlying calibration logic.
- **Supply Chain Consolidation**: Combine the "Weakened" and "Strengthened" linkage charts into a single, unified visualization for easier side-by-side comparison.

## 🏔️ Far Future (Infrastructure & Operations)

### External Deployment
- **Databricks Apps Promotion**: Transition from local Streamlit execution to a production-grade deployment on **Databricks Apps**.
- **CI/CD Integration**: Automate the sync between this repository and the Databricks Workspace repository.

### Enterprise Security
- **Security Audit**: Review authentication and authorization flows for data access.
- **Encryption**: Ensure all cached data (Parquet/Excel) follows enterprise encryption standards.
- **Secrets Management**: Move all configuration (host, tokens) into **Databricks Secrets** or Vault instead of `.databrickscfg`.
