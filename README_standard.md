# Physical Climate Risk Propagation Model

## Overview

This repository contains a **physical climate risk propagation model** based on **multi-regional input–output (MRIO) analysis**.  
The model quantifies **direct and indirect economic impacts** of climate-related shocks by propagating **simultaneous supply and demand disruptions** through inter-sectoral and cross-country production networks.

The framework is designed to:
- Capture **business continuity risks** rather than asset losses
- Account for **supply bottlenecks, inventories, and trade reallocation**
- Produce impacts on **output, value added**, and other macro-economic indicators
- Be extensible towards **hazard-based calibration** and **interactive scenario analysis**

The current implementation uses **EU-27 Social Accounting Matrices (SAMs)** derived from Eurostat data.

---

## Conceptual Model

At a high level, the model operates as follows:

1. **Baseline economy**
   - Production structure is defined by a SAM-derived intermediate input matrix `Z`
   - Gross output `X`, final demand `FD`, and technical coefficients `A` are computed

2. **Shock definition**
   - Users specify **supply shocks** (capacity reductions) and **demand shocks**
   - Shocks can target **different countries and sectors**
   - Shocks are expressed in percentage terms

3. **Propagation & reallocation**
   - Demand-driven production requirements are compared to post-shock capacity
   - Bottlenecks emerge when constrained suppliers limit downstream production
   - Excess demand for inputs is partially reallocated within **global sectors**
   - Reallocation strength is controlled by parameter `γ ∈ [0,1]`

4. **Global feasibility & iteration**
   - Global sectoral technology is assumed fixed (Leontief production)
   - If post-shock demand cannot be met, aggregate demand is endogenously reduced
   - The algorithm iterates until **demand consistency** is reached

5. **Post-processing**
   - Output, value added, and structural changes in input linkages are computed
   - Results are aggregated by country and sector
   - Diagnostics support scientific validation and scenario comparison

---

## Repository Structure

```
Physical-Climate-Risk/
│
├── src/
│   ├── data_io/
│   │   ├── eurostat_sam.py        # Load SAM and extract model blocks
│   │   └── sector_decoder.py     # Decode SAM sector codes to names
│   │
│   ├── io_climate/
│   │   ├── model.py              # IOClimateModel class (main engine)
│   │   ├── propagation.py        # Single-step propagation & reallocation logic
│   │   ├── scenarios.py          # User-friendly shock construction
│   │   └── postprocess.py        # Output, VA, and structural change diagnostics
│   │
│   └── __init__.py
│
├── notebooks/
│   └── main.ipynb                # End-to-end execution & visualization
│
├── README.md
└── requirements.txt
```

---

## Running the Model

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Open `notebooks/main.ipynb`
3. Load the SAM and initialize the model
4. Define a scenario (countries, sectors, shock sizes)
5. Run the model and inspect results

---

## Current Capabilities

✔ Multi-country, multi-sector propagation  
✔ Simultaneous supply and demand shocks  
✔ Bottlenecks and inventory-based reallocation  
✔ Output and value added impacts  
✔ Structural change diagnostics  
✔ Ready for dashboard integration  

---

## Planned Extensions

- Hazard-based calibration (floods, heatwaves, droughts)
- Employment and income satellite accounts
- Interactive web application (Streamlit)
- Stress testing and sensitivity analysis

---

## Disclaimer

This model is a **research and exploratory tool**.  
Results depend on modeling assumptions and should be interpreted as **scenario-based insights**, not forecasts.
