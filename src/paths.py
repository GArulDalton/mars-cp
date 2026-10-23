"""Central paths: every script can be run from any directory."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for p in (DATA, RESULTS, FIGURES):
    p.mkdir(exist_ok=True)
