"""Smoke and property tests. Run: python -m pytest -q tests"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from methods import cq, aci, quantile_tracker, dtaci, mars_cp, Jumper
from experiments import synth, metrics, N0

def test_conformal_quantile():
    s = np.arange(1, 100, dtype=float)
    assert cq(s, 0.1) == 90.0
    assert np.isinf(cq(s[:5], 0.1))          # too few scores -> infinite interval
    assert cq(s, 1.0) == 0.0

def test_jumper_is_fair_bet_under_uniform_pvalues():
    rng = np.random.default_rng(0); logs = []
    for _ in range(200):
        j = Jumper(J=0.01); [j.update(p) for p in rng.uniform(size=300)]; logs.append(j.logM)
    assert np.mean(np.exp(logs)) < 3.0        # martingale: E[M_T] = 1

def test_no_alarm_on_stationary_stream():
    y, yh, _ = synth("S3-Stationary", 0); s = np.abs(y - yh)
    mars_cp(s, 0.1, N0, inner="aci", seed=0)
    assert len(mars_cp.last.alarms) == 0

def test_coverage_near_target_all_wrappers():
    y, yh, cps = synth("S1-RecurringAbrupt", 1); s = np.abs(y - yh)
    for inner in ["aci", "qt", "dtaci"]:
        q = mars_cp(s, 0.1, N0, inner=inner, seed=1)
        assert abs(metrics(s, q, cps)["coverage"] - 0.9) < 0.02

def test_mars_improves_aci_on_abrupt_shifts():
    y, yh, cps = synth("S1-RecurringAbrupt", 2); s = np.abs(y - yh)
    base = metrics(s, aci(s, 0.1, N0), cps)["LCE"]
    wrap = metrics(s, mars_cp(s, 0.1, N0, inner="aci", seed=2), cps)["LCE"]
    assert wrap < base
