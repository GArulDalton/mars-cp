# MARS-CP: Regime Switching Online Conformal Prediction with Test Martingales

[![tests](https://github.com/GArulDalton/mars-cp/actions/workflows/tests.yml/badge.svg)](https://github.com/GArulDalton/mars-cp/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](requirements.txt)

Official implementation of the paper

> **Regime Switching Online Conformal Prediction with Test Martingales**
> G. Arul Dalton and P. Sankar, Department of Computer Science and Engineering, Saveetha Engineering College, Chennai, India.
> Submitted to *Machine Learning* (Springer), 2026.

MARS-CP (Martingale-triggered Adaptive Regime-Switching Conformal Prediction) is a plug-in wrapper that gives any online conformal predictor an explicit, statistically controlled notion of *regime*. A two-sided conformal test martingale monitors the nonconformity scores and raises an alarm only when the evidence of a distribution change exceeds 1/δ, so the probability of any false alarm on an exchangeable stream is at most δ (Ville’s inequality). On an alarm, MARS-CP localises the change point, rebuilds the calibration buffer from post-change scores with a conservative bridge, and re-initialises the wrapped updater.

Supported inner updaters: **ACI** (Gibbs & Candès, 2021), **quantile tracking / conformal PID** (Angelopoulos et al., 2023) and **DtACI** (Gibbs & Candès, 2024).

## Key results (α = 0.1)

| Setting | Effect of wrapping ACI with MARS-CP |
|---|---|
| Recurring abrupt shifts (S1, 20 seeds) | local coverage error −29%, under-coverage time −57% (p < 0.001) |
| NAB machine temperature | mean interval width −22.5% at matched coverage (0.898 vs 0.900) |
| Stationary stream (S3, 20 seeds) | zero false alarms; identical output to ACI |
| Slow gradual drift (S2) | local coverage error +12% (known limitation) |

Full tables are in `results/tables.json` and the per-seed raw results in `results/*.csv`.

## Repository layout

```
mars-cp/
├── src/
│   ├── methods.py         # Split CP, Rolling CP, ACI, QT, DtACI, Jumper martingale, MARSCP wrapper
│   ├── experiments.py     # synthetic streams S1–S4, metrics, benchmark runner
│   ├── data_prep.py       # sliding-refit HistGradientBoosting forecaster -> score streams
│   ├── download_data.py   # fetches the four public datasets
│   ├── sensitivity.py     # δ, W, α sweeps and change-detection statistics
│   ├── summarize.py       # raw pickles -> per-seed CSV files
│   ├── stats_figs.py      # Wilcoxon tests and Figs. 4–7
│   ├── fig1.py            # Fig. 1 (architecture)
│   ├── new_diagrams.py    # Figs. 2–3 (regime life cycle, experimental pipeline)
│   ├── make_tables.py     # results/tables.json (Tables 3–8)
│   └── paths.py           # data/, results/, figures/ locations
├── examples/quickstart.py # 20-line usage example
├── tests/test_methods.py  # unit and property tests (pytest)
├── results/               # per-seed results used in the paper
├── figures/               # Figs. 1–7 (600 dpi)
├── manuscript/            # Node.js scripts that build the Word manuscript and cover letter
├── run_all.sh             # reproduces every number, table and figure (Linux/macOS)
├── run_all.bat            # same, for Windows
├── .github/workflows/     # continuous integration: tests on Python 3.10–3.12
├── requirements.txt  CITATION.cff  LICENSE
```

## Installation

```bash
git clone https://github.com/GArulDalton/mars-cp.git
cd mars-cp
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10–3.12, NumPy 2, scikit-learn 1.8.

## Quick start

```bash
python examples/quickstart.py
```

```python
import numpy as np
from methods import mars_cp                 # with src/ on the Python path

scores = np.abs(y - y_hat)                   # nonconformity scores of any forecaster
q = mars_cp(scores, alpha=0.1, n0=500, inner="aci", delta=0.01)
# interval at time t >= 500:  [y_hat[t] - q[t-500], y_hat[t] + q[t-500]]
```

Main arguments of `mars_cp`: `inner` (`"aci"`, `"qt"`, `"dtaci"`), `delta` (false-alarm budget, default 0.01), `W` (calibration window, 500), `J` (jumper rate, 0.01), `n_min` (bridge length, 20), `memory` (regime memory, `True`), `two_sided`, `localize`, `reset` (switches used in the ablation).

## Reproducing the paper

Linux / macOS:

```bash
./run_all.sh            # full run, about 15–25 minutes on one CPU core
QUICK=1 ./run_all.sh    # fewer seeds, a few minutes (numbers will differ slightly)
python -m pytest -q tests
```

Windows (Command Prompt):

```bat
run_all.bat
REM quick check:  set QUICK=1 && run_all.bat
python -m pytest -q tests
```

Hyperparameters were tuned only on validation seeds 100–103; test seeds are 0–19 (synthetic) and 0–4 (real).

To rebuild the manuscript files after re-running the experiments:

```bash
cd manuscript && npm install && npm run build
```

## Datasets

All data are public and downloaded by `src/download_data.py`: ELEC2 (Harries, 1999), UCI Individual Household Electric Power Consumption (Hébrail & Bérard, 2012, https://doi.org/10.24432/C58K54), Melbourne daily minimum temperatures, and NAB `machine_temperature_system_failure` (Lavin & Ahmad, 2015).

## Citation

```bibtex
@article{aruldalton2026marscp,
  title   = {Regime Switching Online Conformal Prediction with Test Martingales},
  author  = {Arul Dalton, G. and Sankar, P.},
  journal = {Machine Learning},
  note    = {Under review},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`. Contact: G. Arul Dalton, aruldalton@saveetha.ac.in.
