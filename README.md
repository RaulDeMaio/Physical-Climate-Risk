# Physical Climate Risk Propagation Model

## Overview
This repository implements a **multi-country, multi-sector Input–Output (IO) model**
to quantify **direct and indirect economic impacts of physical climate hazards**
across Europe.

The model combines:
- Eurostat Supply–Use / Social Accounting Matrix (SAM) data
- A dynamic IO-based propagation mechanism with bottlenecks and inventories
- A hazard-calibration module based on **historical economic losses**
- Scenario tooling and post-processing for economic risk analysis

The objective is to assess **business-continuity risk**, not asset damage, by
measuring how climate shocks propagate through production networks.

---

## Model Structure

### 1. Core Economic Data
- **SAM / IO matrix** (Eurostat, EU27, ~63 sectors)
- Country–sector nodes: `CC::P_SECTOR`
- Baseline variables:
  - `Z`: intermediate demand matrix
  - `X`: gross output
  - `FD`: final demand
  - `A`: technical coefficients
  - `L`: Leontief inverse

### 2. Propagation Algorithm
At each iteration the model computes:
1. **Demand-driven output** (`X_dem`)
2. **Capacity constraints** (`X_cap`)
3. **Bottlenecks and rationing**
4. **Constrained flows** (`Z_con`)
5. **Needed flows** (`Z_need`)
6. **Inventories and excess demand**
7. **Within-sector trade reallocation** (γ mechanism)
8. **Globally feasible output** using fixed global technology
9. **Implied final demand**
10. **Iterative demand adjustment until convergence**

Convergence is checked on **final demand consistency**.

---

## Hazard Calibration

### Data Sources
- **EEA economic losses from climate extremes (1980–2024)**
- Hazards:
  - Hydrological
  - Meteorological
  - Geophysical
  - Climatological (heatwaves / other)
- **Eurostat national accounts output (P1)** via API

### Calibration Logic
1. Split annual country losses across hazards using historical hazard shares
2. Convert losses and output to **constant 2024 EUR**
3. Compute **intensity = losses / output**
4. Build **percentile-based intensity levels** (moderate, severe, extreme)
5. Map intensities to **supply capacity shocks**
6. Demand shocks emerge endogenously from the model

Fallback logic:
- Country-specific deflators when CLV exists
- EU-wide implicit deflator when CLV is missing

---

## Repository Structure

```
src/
 ├─ data_io/
 │   ├─ eurostat_sam.py
 │   ├─ eurostat_output.py
 │
 ├─ io_climate/
 │   ├─ model.py
 │   ├─ propagation.py
 │
 ├─ scenarios.py
 ├─ calibration.py
 ├─ postprocess.py
 ├─ viz.py
 │
notebooks/
 ├─ main_refactored_final.ipynb
```

---

## Outputs
- Output losses (absolute and %)
- Value added impacts
- Country and sector rankings
- Structural IO changes
- Hazard-specific scenario impacts

---

## Intended Use
- Climate stress testing
- Business continuity analysis
- Macroeconomic risk assessment

---

## Disclaimer
Results depend on assumptions regarding:
- Fixed production technology
- Uniform sectoral vulnerability
- Historical-loss-based calibration

They should be interpreted as **risk indicators**, not forecasts.
