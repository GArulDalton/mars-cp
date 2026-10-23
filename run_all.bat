@echo off
REM Reproduce every number, table and figure in the paper (Windows).
REM Full run: about 15-25 minutes on one CPU core. Quick check: set QUICK=1 before running.
cd /d "%~dp0"
python src\download_data.py      || exit /b 1
python src\data_prep.py          || exit /b 1
python src\experiments.py synth  || exit /b 1
python src\experiments.py real   || exit /b 1
python src\experiments.py ablation || exit /b 1
python src\sensitivity.py        || exit /b 1
python src\summarize.py          || exit /b 1
python src\stats_figs.py         || exit /b 1
python src\fig1.py               || exit /b 1
python src\new_diagrams.py       || exit /b 1
python src\make_tables.py        || exit /b 1
echo Done. See results\ and figures\.
