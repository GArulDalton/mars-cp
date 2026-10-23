#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper.
# Full run: about 15-25 minutes on one CPU core. Quick check: QUICK=1 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
python src/download_data.py          # 1. public datasets -> data/
python src/data_prep.py              # 2. sliding-refit forecaster -> data/streams.pkl
python src/experiments.py synth      # 3. synthetic benchmark (S1-S4)
python src/experiments.py real       # 4. real-world benchmark
python src/experiments.py ablation   # 5. ablation study
python src/sensitivity.py            # 6. delta / W / alpha sweeps and detection statistics
python src/summarize.py              # 7. per-seed CSV files
python src/stats_figs.py             # 8. Wilcoxon tests and Figs. 4-7
python src/fig1.py                   # 9. Fig. 1 (architecture)
python src/new_diagrams.py           # 10. Figs. 2-3 (block diagrams)
python src/make_tables.py            # 11. results/tables.json (Tables 3-8)
echo "Done. See results/ and figures/."
