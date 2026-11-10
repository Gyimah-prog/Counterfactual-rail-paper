# Counterfactual Project 2

Results of the monitor, adapt, validate, and refresh framework on the FRA Class 7 corridor data.

## Contents
- `Counterfactual_Project_2_Results.docx` is the full methodology and results report.
- `figures/` holds nine figures, each as PNG (300 dpi) and PDF.
- `tables/results_tables.xlsx` holds every analysis table, one sheet each. The CSV files hold monthly monitoring, stress tests, and sensitivity runs.
- `data/` holds per case results for every method, per case fresh generation results, the updated 2013 to 2015 model, and run settings.
- `code/` holds the scripts in run order.

## Inputs
The scripts read the canonical cache in `counterfactual_work/vscode/scripts/.cache` (`data.pkl`, `model_base_nb.pkl`, `cf_2014_base_nb.pkl`). Set the path in `code/cf2lib.py` (variable `C`).

## Run order
1. `s01_reproduce.py` checks the locked P2 numbers (476 of 1,672 and 144 of 2,092).
2. `s02_fresh.py 0 2` and `s02_fresh.py 1 2` run fresh generation in two workers (about 16 minutes on two cores). Train the updated model first with the block at the top of the original run, or reuse `data/model_updated_2013_2015.pkl` placed in `out/`.
3. `s03_ladder.py 3 8 main` runs the ladder. Sensitivity runs use `SKIP_MV=1` with window, cap, tag, and tail level arguments.
4. `s04_stress.py`, `s05_monitor.py`, `s06_analysis.py`, `s07_figures.py`, `s08_numbers.py`, then `node build_report.js`.

Packages used: numpy 2.5, pandas 3.0, scikit-learn 1.9, xgboost 3.4, POT 0.9.7, statsmodels, matplotlib.
