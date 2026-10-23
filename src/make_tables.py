"""Build results/tables.json (the exact numbers shown in Tables 3-8 of the paper)."""
import json, pickle, numpy as np, pandas as pd
from paths import RESULTS
syn = pd.read_csv(RESULTS / "res_synth.csv"); real = pd.read_csv(RESULTS / "res_real.csv")
abl = pd.read_csv(RESULTS / "res_ablation.csv"); W = pd.read_csv(RESULTS / "wilcoxon.csv")
S = pickle.load(open(RESULTS / "res_sens.pkl", "rb"))
ORDER = ["SplitCP", "RollingCP", "ACI", "MARS(ACI)", "QT", "MARS(QT)", "DtACI", "MARS(DtACI)"]

def tab(df, dss, cols, with_sd=True):
    out = []
    for ds in dss:
        for m in ORDER:
            d = df[(df.ds == ds) & (df.m == m)]
            if not len(d): continue
            row = [ds, m]
            for c, f in cols:
                mu, sd = d[c].mean(), d[c].std()
                cell = f.format(mu)
                if with_sd and c != "coverage" and not np.isnan(sd) and sd > 1e-12: cell += " ± " + f.format(sd)
                row.append("—" if cell.startswith("nan") else cell)
            out.append(row)
    return out

T = {}
T["synth"] = tab(syn, ["S1-RecurringAbrupt", "S2-Gradual", "S4-MixedRecurring"],
                 [("coverage", "{:.3f}"), ("width", "{:.3f}"), ("LCE", "{:.4f}"), ("UCT", "{:.2f}"), ("worst_post_shift", "{:.3f}")])
T["real"] = tab(real, ["ELEC2", "Power", "MelbTemp", "MachineTemp"],
                [("coverage", "{:.3f}"), ("width", "{:.3f}"), ("inf_pct", "{:.2f}"), ("LCE", "{:.4f}"), ("UCT", "{:.2f}"), ("time_ms_per_step", "{:.3f}")], with_sd=False)
T["abl"] = []
for ds in ["S1-RecurringAbrupt", "S4-MixedRecurring", "ELEC2", "Power", "MachineTemp"]:
    for m in ["MARS(ACI) full", "w/o regime memory", "w/o martingale reset", "one-sided martingale", "w/o change localization"]:
        d = abl[(abl.ds == ds) & (abl.m == m)]
        if len(d):
            T["abl"].append([ds, m, f"{d.coverage.mean():.3f}", f"{d.width.mean():.3f}", f"{d.LCE.mean():.4f}",
                             f"{d.UCT.mean():.2f}", f"{d.alarms.mean():.1f}", f"{d.recalls.mean():.1f}"])
T["wil"] = [[r.ds, r.base, r.metric, f"{r.base_mean:.4f}", f"{r.mars_mean:.4f}", f"{r.rel:+.1f}",
             "<0.001" if r.p < 0.001 else f"{r.p:.3f}"] for r in W.itertuples()]
det = pd.DataFrame(S["detect"]).groupby("stream").agg(n_cp=("n_cp", "mean"), det=("detected", "mean"),
      delay=("mean_delay", "mean"), dsd=("mean_delay", "std"), fa=("false_alarms", "mean"))
T["det"] = [[i, f"{r.n_cp:.0f}", f"{r.det:.2f}", "—" if np.isnan(r.delay) else f"{r.delay:.1f} ± {r.dsd:.1f}", f"{r.fa:.2f}"] for i, r in det.iterrows()]
dd = pd.DataFrame(S["delta"]).groupby(["stream", "delta"])[["coverage", "width", "LCE", "alarms"]].mean()
T["delta"] = [[i[0], str(i[1]), f"{r.coverage:.3f}", f"{r.width:.3f}", f"{r.LCE:.4f}", f"{r.alarms:.1f}"] for i, r in dd.iterrows()]
aa = pd.DataFrame(S["alpha"]); T["alpha"] = []
for (st, a), g in aa.groupby(["stream", "alpha"]):
    b = g[g.method == "ACI"].iloc[0]; m = g[g.method == "MARS(ACI)"].iloc[0]
    T["alpha"].append([st, str(a), f"{b.coverage:.3f} / {m.coverage:.3f}", f"{b.width:.3f} / {m.width:.3f}",
                       f"{b.LCE:.4f} / {m.LCE:.4f}", f"{b.inf_pct:.2f} / {m.inf_pct:.2f}"])
ww = pd.DataFrame(S["W"]).groupby(["stream", "W"])[["coverage", "width", "LCE"]].mean()
T["W"] = [[i[0], str(i[1]), f"{r.coverage:.3f}", f"{r.width:.3f}", f"{r.LCE:.4f}"] for i, r in ww.iterrows()]
json.dump(T, open(RESULTS / "tables.json", "w"), indent=1, ensure_ascii=False)
print("wrote", RESULTS / "tables.json")
