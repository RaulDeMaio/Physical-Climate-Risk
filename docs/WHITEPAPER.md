# Whitepaper: Physical Climate Risk Propagation in Production Networks

**Date**: February 2026  
**Subject**: Methodology and Implementation documentation for the Physical-Climate-Risk model.

---

## 1. Abstract
This document describes the methodology for a multi-country, multi-sector Input-Output (IO) simulation model designed to quantify the propagation of physical climate risks through European production networks. Unlike traditional "asset-damage" models, this framework focuses on **business continuity risk**, modeling how supply-side shocks and production bottlenecks create cascading economic losses across international boundaries and sectors.

## 2. Theoretical Framework
The model is based on the theory of **Production Networks** and **Input-Output Analysis**. It specifically implements an **Adaptive Input-Output** approach, where standard Leontief assumptions (fixed coefficients) are augmented with non-linear rationing rules and inventory dynamics to capture **short-term** bottlenecks.

### 2.1 The Z-Matrix and technical coefficients
The core of the economy is represented by the intermediate flow matrix $Z$, where entry $Z_{ij}$ represents the volume of goods/services produced by sector $i$ and consumed by sector $j$.
The technical coefficients matrix $A$ is defined as:
$$A_{ij} = \frac{Z_{ij}}{X_j}$$
where $X_j$ is the total gross output of sector $j$.

### 2.2 Propagation Mechanism (Non-linear rationing)
When a climate shock reduces the capacity of a specific node (country-sector), the impact propagates downstream through **supply constraints** and upstream through **demand reduction**.

The model employs an iterative loop that solves for consistency between:

1.  **Demand-driven output requirements**: What sectors *want* to produce given final demand.
2.  **Supply constraints**: What sectors *can* produce given shocks and input availability.

### 2.3 Bottlenecks and Inventories
The "rationing factor" $r_i$ for producer $i$ is:
$$r_i = \min\left(1, \frac{X^{cap}_i}{X^{dem}_i}\right)$$
The "bottleneck" $s_j$ for user $j$ is the minimum availability among all its critical inputs:
$$s_j = \min_{i: A_{ij} > 0} r_i$$

Inventories are tracked to allow for partial mitigation of shocks by using stockpiled inputs before production stops.

---

## 3. Hazard Calibration Methodology
The model converts climate "hazards" (floods, heatwaves, etc.) into economic "shocks" using historical empirical data from the European Environment Agency (EEA) and Eurostat.

### 3.1 Intensity Definition
We define hazard intensity $I_{c,h,t}$ for country $c$, hazard $h$, at time $t$ as:
$$I_{c,h,t} = \frac{\text{Historical Loss}}{\text{Total Economic Output}}$$
This allows us to calibrate a "one-in-ten-year" flood in Germany differently from a "one-in-ten-year" flood in Italy, based on their respective historical vulnerabilities and economic scales.

### 3.2 Scenario Mapping
Historical intensities are used to build percentile-based scenario levels:

- **Moderate**: 50th percentile (Median year).
- **Severe**: 75th percentile.
- **Extreme**: 90th percentile.

These intensities are directly mapped to the supply shock scalar $\phi$, representing the fraction of daily production capacity lost at the shocked node.

---

## 4. Software Architecture
The implementation follows a modular Python architecture designed for integration into distributed data pipelines (e.g., Databricks/Pyspark).

### 4.1 Module Structure

- `src/data_io/`: Handles ingestion of SAM matrices and national accounts via Eurostat API.
- `src/io_climate/model.py`: Orchestrates the simulation, managing iteration limits and convergence checks.
- `src/io_climate/propagation.py`: Implementation of the core rationing and reallocation logic.
- `src/io_climate/calibration.py`: Logic for loss-fraction calculation and hazard mapping.
- `src/io_climate/scenarios.py`: High-level interface for defining multi-country stress tests.

### 4.2 Integrated Workflow
The reference implementation for the end-to-end pipeline is `main_refactored_final_with_calibration.ipynb`. This notebook provides a unified interface to load datasets, calibrate shocks based on historical EEA data, and execute the IO propagation in a single session.

### 4.2 Computational Performance
The model is optimized for high-dimensional IO matrices (e.g., EU27 x 64 sectors $\approx$ 1,728 nodes). The propagation loop uses vectorized NumPy operations, allowing for rapid scenario exploration and sensitivity analysis.

---

## 5. Technical Assumptions & Limitations
- **Fixed Technology ($A$ matrix)**: The model assumes sectors cannot change their recipe for production in the short term (fixed technical coefficients).
- **Infinite Elasticity of Reallocation**: The $\gamma$ parameter controls the ability to switch suppliers within the same global sector, assuming perfectly integrated markets for those sectors.
- **Endogenous Demand**: Final demand is reduced as a consequence of supply chain failures, but the model does not currently account for price elasticity or inflationary feedback.

---

## 6. Glossary

- **SAM (Social Accounting Matrix)**: A comprehensive data table representing all economic transactions in an economy.
- **Leontief Inverse ($L$)**: $(I-A)^{-1}$, representing the total (direct and indirect) requirements per unit of final demand.
- **Rationing**: The process of distributing scarce outputs to downstream users when demand exceeds supply capacity.
