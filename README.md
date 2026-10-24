# Counterfactual Explanations for Railway Track Geometry Maintenance

Code and results for two related studies on counterfactual explanations for FRA Class 7 track geometry defect prediction.

## Repository layout

| Folder | Contents |
|---|---|
| `paper1/` | Main paper. Counterfactual generation, temporal validity, and recalibration analysis. |
| `paper1/src`, `paper1/scripts` | Library code and the scripts that compute results and draw figures. |
| `paper1/notebooks` | Reproduction notebook (clean and executed) and the main analysis notebook. |
| `paper1/figures` | Notebook figures and `paper_ready/` figures (PNG and PDF). |
| `paper1/outputs_soft_penalty_cf` | Soft penalty counterfactual comparison results. |
| `paper1/results` | Headline numbers (JSON). |
| `project2/` | Project 2. Monitor, adapt, validate, and refresh framework. See `project2/README.md` for run order. |

## Data

Raw inspection data (`RawD.xlsx`) and the cached pickles (`data.pkl`, model and counterfactual caches) are not included. The scripts expect them in a local `.cache` folder. Set the path in `paper1/scripts/_cf_lib.py` and `project2/code/cf2lib.py`.

## Contact

Benjamin Gyimah Boateng, University of Maryland (bboaten1@umd.edu)
