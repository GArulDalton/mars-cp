import numpy as np, pickle, pandas as pd
import experiments as E
from methods import mars_cp, aci

res = {"delta": [], "W": [], "alpha": [], "detect": []}
from paths import DATA, RESULTS
st = pickle.load(open(DATA / "streams.pkl", "rb"))
yM, yhM = st["MachineTemp"]

def one(s, cps, a, **kw):
    E.ALPHA = a
    q = mars_cp(s, a, E.N0, **kw)
    m = E.metrics(s, q, cps, target=1 - a); m["alarms"] = len(mars_cp.last.alarms); return m

for d in [0.001, 0.01, 0.05, 0.1]:
    for sd in range(10):
        y, yh, c = E.synth("S1-RecurringAbrupt", sd)
        m = one(np.abs(y - yh), c, 0.1, inner="aci", delta=d, seed=sd); m.update(stream="S1", delta=d); res["delta"].append(m)
        y, yh, c = E.synth("S3-Stationary", sd)
        m = one(np.abs(y - yh), c, 0.1, inner="aci", delta=d, seed=sd); m.update(stream="S3", delta=d); res["delta"].append(m)
    m = one(np.abs(yM - yhM), [], 0.1, inner="aci", delta=d); m.update(stream="MachineTemp", delta=d); res["delta"].append(m)
print("delta done", flush=True)
for W in [250, 500, 1000]:
    for sd in range(10):
        y, yh, c = E.synth("S1-RecurringAbrupt", sd)
        m = one(np.abs(y - yh), c, 0.1, inner="aci", W=W, seed=sd); m.update(stream="S1", W=W); res["W"].append(m)
    m = one(np.abs(yM - yhM), [], 0.1, inner="aci", W=W); m.update(stream="MachineTemp", W=W); res["W"].append(m)
print("W done", flush=True)
for a in [0.05, 0.1, 0.2]:
    for name, (y, yh) in st.items():
        s = np.abs(y - yh)
        E.ALPHA = a
        qb = aci(s, a, E.N0); mb = E.metrics(s, qb, [], target=1 - a); mb.update(stream=name, alpha=a, method="ACI")
        m = one(s, [], a, inner="aci"); m.update(stream=name, alpha=a, method="MARS(ACI)")
        res["alpha"] += [mb, m]
print("alpha done", flush=True)
E.ALPHA = 0.1
# change detection: delay and false alarms on S1/S4 (20 test seeds)
for kind in ["S1-RecurringAbrupt", "S4-MixedRecurring", "S3-Stationary"]:
    for sd in range(20):
        y, yh, cps = E.synth(kind, sd); s = np.abs(y - yh)
        mars_cp(s, 0.1, E.N0, inner="aci", seed=sd); al = np.array(mars_cp.last.alarms)
        used = set(); delays = []
        for c in cps:
            hit = [a for a in al if c <= a < c + 400 and a not in used]
            if hit: delays.append(hit[0] - c); used.add(hit[0])
        res["detect"].append(dict(stream=kind, seed=sd, n_cp=len(cps), detected=len(delays),
                                  mean_delay=np.mean(delays) if delays else np.nan,
                                  false_alarms=len(al) - len(used)))
pickle.dump(res, open(RESULTS / "res_sens.pkl", "wb"))
print("all done")
