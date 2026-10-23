"""Minimal MARS-CP example on a synthetic stream with recurring abrupt regime shifts.
Run from the repository root:  python examples/quickstart.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from methods import aci, mars_cp
from experiments import synth, metrics, N0

y, yhat, change_points = synth("S1-RecurringAbrupt", seed=0)
scores = np.abs(y - yhat)                      # nonconformity scores
for name, q in [("ACI", aci(scores, 0.1, N0)),
                ("MARS(ACI)", mars_cp(scores, 0.1, N0, inner="aci", seed=0))]:
    m = metrics(scores, q, change_points)
    print(f"{name:10s} coverage={m['coverage']:.3f} width={m['width']:.3f} "
          f"LCE={m['LCE']:.4f} UCT={m['UCT']:.2f}%")
print("MARS alarms at steps:", mars_cp.last.alarms)
# Prediction interval at time t (t >= N0):  [yhat[t] - q[t-N0], yhat[t] + q[t-N0]]
