# Streamlit Hazard App

This directory contains the implementation of the **Physical Climate Risk Propagation Dashboard**.

## How to Run Locally

### Prerequisites

1.  **Databricks Connect (v16.1+)**: The app uses Databricks Connect with **Serverless Compute**. Ensure you have version 16.1 or higher:
    ```bash
    uv pip install databricks-connect>=16.1
    ```
    > ⚠️ **Known Issue**: Versions below 16.1 do not support `.serverless()` mode. You will see:
    > `Serverless mode is not yet supported in this version of Databricks Connect.`

2.  **Databricks Configuration**: Create/update `~/.databrickscfg` with a profile named `local`:
    ```ini
    [local]
    host = https://your-workspace.cloud.databricks.com/
    token = dapi-xxxx
    ```
    The app reads the `local` profile by default (or set `DATABRICKS_CONFIG_PROFILE` env var).

### Launch

```bash
streamlit run app/main.py --server.headless true
```
- `--server.headless true`: Prevents Streamlit from auto-opening a browser (useful in WSL/Docker).

### Usage

1.  The app connects to Databricks via **Serverless Compute**.
2.  Use the **Sidebar** to configure:
    - **Hazard Calibration**: Pick a country (e.g., IT) and hazard (e.g., Heatwave).
    - **Manual Shock**: Select specific sectors and shock %.
3.  Click **Run Simulation** to see economic impact results.

## Deployment to Databricks Apps

1.  Sync this branch to your Databricks Workspace Repos.
2.  Create a new App pointing to `app/main.py`.
3.  The app auto-detects the Databricks Runtime and uses native Spark.

## Architecture

| File              | Purpose                                                |
|-------------------|--------------------------------------------------------|
| `app/main.py`     | Entry point, UI layout, visualization logic            |
| `app/data.py`     | Data loading, Model initialization (cached)            |
| `app/utils.py`    | Environment detection, Spark session handling          |
| `src/`            | Reused core logic from notebook refactoring            |

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `[CONNECT_URL_NOT_SET]` | Missing Databricks config | Add `host` and `token` to `~/.databrickscfg` |
| `Serverless mode is not yet supported` | Old databricks-connect | Upgrade: `uv pip install databricks-connect>=16.1` |
| `Cluster id or serverless are required` | Profile missing compute target | Add `cluster_id=xxx` or upgrade to use `.serverless()` |
