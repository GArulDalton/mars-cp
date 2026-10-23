"""Convert raw result pickles (results/res_*.pkl) into per-seed CSV files."""
import pickle, pandas as pd
from paths import RESULTS

def to_frame(path):
    R = pickle.load(open(path, "rb")); rows = []
    for ds, runs in R.items():
        for sd, r in enumerate(runs):
            for m, v in r.items():
                rows.append(dict(ds=ds, m=m, seed=sd, **v))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    for part in ["synth", "real", "ablation"]:
        p = RESULTS / f"res_{part}.pkl"
        if p.exists():
            d = to_frame(p); d.to_csv(RESULTS / f"res_{part}.csv", index=False)
            cols = [c for c in ["coverage", "width", "inf_pct", "LCE", "UCT", "alarms"] if c in d]
            print(f"== {part}"); print(d.groupby(["ds", "m"], sort=False)[cols].mean().round(4).to_string())
