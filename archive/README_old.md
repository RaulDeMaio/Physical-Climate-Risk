# Physical Climate Risk Propagation Model

## Abstract

This repository implements a **multi-regional input–output (MRIO) model** designed to quantify the **direct and indirect economic impacts of physical climate risks**. The framework propagates simultaneous **supply-side capacity shocks** and **final-demand shocks** through a high-resolution **EU-27 Social Accounting Matrix (SAM)**, accounting for production bottlenecks, inventory constraints, and limited substitution possibilities across global sectors.

The model is intended as a **scenario-analysis and stress-testing tool** for applications in climate risk assessment, macro-financial analysis, and economic resilience studies. It focuses on **business continuity impacts** rather than asset destruction, and is suitable for both exploratory research and policy-oriented analysis.

---

## 1. Conceptual Framework

### 1.1 Economic Structure

The baseline economy is represented by a Social Accounting Matrix (SAM), from which the following core objects are derived:

- Intermediate input matrix \( Z \in \mathbb{R}^{n \times n} \)
- Gross output vector \( X \in \mathbb{R}^{n} \)
- Final demand vector \( FD \in \mathbb{R}^{n} \)
- Technical coefficients matrix:
\[
A_{ij} = \frac{Z_{ij}}{X_j}
\]

where each index \( i \) corresponds to a **country–sector node**.

Production technologies are assumed to be **Leontief**, implying fixed input proportions and no substitution across sectors at the global level.

---

### 1.2 Shock Design

Two types of shocks are considered:

- **Supply (capacity) shocks**: exogenous reductions in productive capacity
\[
X^{cap}_i = X_i (1 - s^p_i)
\]

- **Demand shocks**: exogenous reductions in final demand
\[
FD^{post}_i = FD_i (1 - s^d_i)
\]

Supply and demand shocks can target **different countries and sectors**, allowing for flexible scenario design.

---

### 1.3 Propagation and Bottlenecks

For a given demand vector, a *demand-driven* production requirement is computed as:
\[
X^{dem} = L FD^{post}, \quad L = (I - A)^{-1}
\]

When production requirements exceed post-shock capacity, **rationing factors** are computed:
\[
r_i = \min \left(1, \frac{X^{cap}_i}{X^{dem}_i} \right)
\]

Bottlenecks propagate through input–output linkages, constraining downstream production.

---

### 1.4 Inventory-Based Reallocation

When intermediate input demand cannot be met, the model allows for **partial reallocation of supply** within *global sectors*.

Let:
- \( Z^{con} \): constrained intermediate flows
- \( Z^{need} \): flows required to satisfy demand-only production

Excess input demand:
\[
E = \max(Z^{need} - Z^{con}, 0)
\]

Available inventories are redistributed proportionally within global sectors, scaled by a parameter:
\[
\gamma \in [0,1]
\]

This mechanism captures **limited trade reallocation** without assuming full substitutability.

---

### 1.5 Global Feasibility and Iteration

After reallocation, total feasible output is computed using **fixed global-sector technology**, enforcing Leontief feasibility at the aggregate level.

If implied final demand is lower than post-shock demand, aggregate demand is endogenously reduced. The algorithm iterates until **demand consistency** is achieved.

Convergence is assessed on the final demand vector.

---

## 2. Model Outputs

The model produces:

- Gross output impacts (absolute and percentage)
- Value added impacts (IO-consistent residual)
- Country- and sector-level aggregations
- Structural changes in production networks, measured as:
\[
\Delta A = A^{final} - A^{baseline}
\]

These diagnostics enable the identification of **amplification mechanisms** and **critical economic linkages**.

---

## 3. Implementation

### 3.1 Code Structure

```
src/
├── data_io/
│   ├── eurostat_sam.py        # SAM ingestion and block extraction
│   └── sector_decoder.py     # Sector code to description mapping
│
├── io_climate/
│   ├── model.py              # Core iterative model
│   ├── propagation.py        # Single-step propagation logic
│   ├── scenarios.py          # Scenario and shock construction
│   └── postprocess.py        # Impact metrics and structural diagnostics
│
└── notebooks/
    └── main.ipynb            # End-to-end execution and visualization
```

The core propagation logic is implemented as a **pure function**, facilitating testing, reproducibility, and methodological extensions.

---

## 4. Intended Applications

- Climate stress testing of economic systems
- Assessment of indirect losses and cascade effects
- Comparison of hazard scenarios and adaptation strategies
- Support for climate-related macro-financial analysis

---

## 5. Planned Extensions

- Calibration of shocks using historical climate hazard data
- Integration of employment and income satellite accounts
- Regional downscaling (sub-national resolution)
- Interactive web-based interface for scenario exploration

---

## 6. Disclaimer

This model is a **research-oriented analytical tool**. Results depend on assumptions regarding technology, substitution limits, and shock calibration. Outputs should be interpreted as **scenario-based insights**, not forecasts or predictions.
