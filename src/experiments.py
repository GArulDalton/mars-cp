import numpy as np, pickle, time, json, sys, os
from paths import DATA, RESULTS
NSEED_SYN = 2 if os.environ.get("QUICK") else 20
NSEED_REAL = 1 if os.environ.get("QUICK") else 5
NSEED_ABL = 1 if os.environ.get("QUICK") else 3
from methods import split_cp, rolling_cp, aci, dtaci, quantile_tracker, mars_cp

ALPHA = 0.1; N0 = 500; SEG = 1250

def synth(kind, seed, T=10000, d=5):
    r = np.random.default_rng(seed)
    X = r.normal(size=(T, d)); beta = np.ones(d) / np.sqrt(d)
    yhat = X @ beta; e = r.normal(size=T)
    sig = np.ones(T); bias = np.zeros(T); cps = []
    if kind == "S1-RecurringAbrupt":
        pat = [1.0, 3.0, 1.0, 0.4] * 2
        for k, v in enumerate(pat): sig[k*SEG:(k+1)*SEG] = v
        cps = [k*SEG for k in range(1, 8)]
    elif kind == "S2-Gradual":
        tt = np.arange(T); sig = 1 + 1.5 * (1 - np.cos(2*np.pi*tt/5000)) / 2
    elif kind == "S3-Stationary":
        pass
    elif kind == "S4-MixedRecurring":
        pat = ["A", "bias", "A", "heavy"] * 2
        for k, v in enumerate(pat):
            sl = slice(k*SEG, (k+1)*SEG)
            if v == "bias": bias[sl] = 2.0
            if v == "heavy": e[sl] = r.standard_t(2.5, size=SEG)
        cps = [k*SEG for k in range(1, 8)]
    y = yhat + bias + sig * e
    return y, yhat, cps

def metrics(s, q, cps=(), w=100, target=1-ALPHA):
    s = s[N0:]; cov = (s <= q).astype(float)
    clip = np.max(s)                             # half-width of the trivial interval
    qc = np.minimum(q, clip)
    roll = np.convolve(cov, np.ones(w)/w, mode="valid")
    out = {"coverage": cov.mean(), "width": 2*qc.mean(), "inf_pct": 100*np.mean(np.isinf(q)),
           "LCE": np.mean(np.abs(roll - target)), "worst_roll": roll.min(),
           "UCT": 100*np.mean(roll < target - 0.05)}
    if len(cps):
        post = [cov[c - N0:c - N0 + 200].mean() for c in cps]
        out["post_shift_cov"] = float(np.mean(post)); out["worst_post_shift"] = float(np.min(post))
    return out

METHODS = {
    "SplitCP": (split_cp, {}), "RollingCP": (rolling_cp, {}),
    "ACI": (aci, {}), "MARS(ACI)": (mars_cp, {"inner": "aci"}),
    "QT": (quantile_tracker, {}), "MARS(QT)": (mars_cp, {"inner": "qt"}),
    "DtACI": (dtaci, {}), "MARS(DtACI)": (mars_cp, {"inner": "dtaci"}),
}
ABLATIONS = {
    "MARS(ACI) full": {"inner": "aci"}, "w/o regime memory": {"inner": "aci", "memory": False},
    "w/o martingale reset": {"inner": "aci", "reset": False},
    "one-sided martingale": {"inner": "aci", "two_sided": False},
    "w/o change localization": {"inner": "aci", "localize": False},
}

def run_stream(y, yhat, cps, seed, methods=METHODS):
    s = np.abs(y - yhat); res = {}
    for name, (fn, kw) in methods.items():
        t0 = time.time()
        q = fn(s, ALPHA, N0, seed=seed, **kw) if fn is mars_cp else fn(s, ALPHA, N0, **kw)
        m = metrics(s, q, cps); m["time_ms_per_step"] = 1000*(time.time()-t0)/len(q)
        if fn is mars_cp:
            m["alarms"] = len(mars_cp.last.alarms); m["recalls"] = len(mars_cp.last.recalls)
        res[name] = m
    return res

if __name__ == "__main__":
    part = sys.argv[1]
    allres = {}
    if part == "synth":
        for kind in ["S1-RecurringAbrupt", "S2-Gradual", "S3-Stationary", "S4-MixedRecurring"]:
            allres[kind] = [run_stream(*synth(kind, sd), sd) for sd in range(NSEED_SYN)]
            print(kind, "done", flush=True)
    elif part == "real":
        streams = pickle.load(open(DATA / "streams.pkl", "rb"))
        for name, (y, yh) in streams.items():
            allres[name] = [run_stream(y, yh, [], sd) for sd in range(NSEED_REAL)]
            print(name, "done", flush=True)
    elif part == "ablation":
        abl = {k: (mars_cp, v) for k, v in ABLATIONS.items()}
        for kind in ["S1-RecurringAbrupt", "S4-MixedRecurring"]:
            allres[kind] = [run_stream(*synth(kind, sd), sd, abl) for sd in range(NSEED_SYN)]
        streams = pickle.load(open(DATA / "streams.pkl", "rb"))
        for name, (y, yh) in streams.items():
            allres[name] = [run_stream(y, yh, [], sd, abl) for sd in range(NSEED_ABL)]
            print(name, "done", flush=True)
    pickle.dump(allres, open(RESULTS / f"res_{part}.pkl", "wb"))
