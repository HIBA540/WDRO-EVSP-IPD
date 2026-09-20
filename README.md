# WDRO-EVSP: Wasserstein Distributionally Robust Optimization for EV 

Code companion for the paper on Adaptive Wasserstein Distributionally Robust Optimization with Forecast-Informed Radius Calibration for Electric Vehicle Scheduling Under Multi-Source.

## Structure

- `config/` — Configuration globale
- `src/data/` — Chargement données, features ξ, flotte EV
- `src/calibration/` — 4 méthodes de calibration (EK, Bai, Gotoh, IPD)
- `src/forecast/` — Moteur de prévision ξ
- `src/optimization/` — MILP WDRO + recourse OOS
- `src/evaluation/` — KPIs (E_OOS, CVaR90, Reliability, ΔJ%)
- `src/visualization/` — Figures
- `src/export/` — JSON, CSV, LaTeX, Excel
- `scripts/` — Point d'entrée

## Installation

```bash
pip install -r requirements.txt
