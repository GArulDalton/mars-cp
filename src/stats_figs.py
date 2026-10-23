import pickle, numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import wilcoxon
import experiments as E
from methods import aci, mars_cp
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.grid":True,"grid.alpha":0.3})
from paths import DATA, RESULTS, FIGURES
FIG=str(FIGURES)
syn=pd.read_csv(RESULTS/"res_synth.csv"); real=pd.read_csv(RESULTS/"res_real.csv")
# ---- paired Wilcoxon (synthetic, 20 seeds)
rows=[]
for ds in ["S1-RecurringAbrupt","S2-Gradual","S4-MixedRecurring"]:
    for b in ["ACI","QT","DtACI"]:
        for met in ["LCE","width","UCT"]:
            x=syn[(syn.ds==ds)&(syn.m==b)].sort_values("seed")[met].values
            y=syn[(syn.ds==ds)&(syn.m==f"MARS({b})")].sort_values("seed")[met].values
            try: p=wilcoxon(x,y).pvalue
            except ValueError: p=1.0
            rows.append(dict(ds=ds,base=b,metric=met,base_mean=x.mean(),mars_mean=y.mean(),rel=100*(y.mean()-x.mean())/x.mean(),p=p))
W=pd.DataFrame(rows); W.to_csv(RESULTS/"wilcoxon.csv",index=False); print(W.round(4).to_string())

# ---- Fig 4 synthetic trace S1 seed 0
y,yh,cps=E.synth("S1-RecurringAbrupt",0); s=np.abs(y-yh)
qa=aci(s,0.1,E.N0); qm=mars_cp(s,0.1,E.N0,inner="aci",seed=0); al=mars_cp.last.alarms
t=np.arange(E.N0,len(s)); roll=lambda q: np.convolve((s[E.N0:]<=q).astype(float),np.ones(100)/100,"valid")
fig,axs=plt.subplots(2,1,figsize=(7.2,4.2),sharex=True)
axs[0].plot(t,s[E.N0:],".",ms=0.8,color="#bbb",label="score $s_t$")
axs[0].plot(t,np.minimum(qa,15),lw=1,label="ACI $q_t$",color="#d62728"); axs[0].plot(t,np.minimum(qm,15),lw=1,label="MARS(ACI) $q_t$",color="#1f77b4")
for c in cps: axs[0].axvline(c,color="k",ls=":",lw=0.7)
axs[0].plot(al,[14]*len(al),"v",color="#1f77b4",ms=4,label="alarms"); axs[0].set_ylim(0,15); axs[0].set_ylabel("threshold / score"); axs[0].legend(ncol=4,fontsize=7,loc="lower center",bbox_to_anchor=(0.5,1.0),frameon=False)
axs[1].plot(t[99:],roll(qa),lw=0.9,color="#d62728",label="ACI"); axs[1].plot(t[99:],roll(qm),lw=0.9,color="#1f77b4",label="MARS(ACI)")
axs[1].axhline(0.9,color="k",lw=0.7); axs[1].set_ylim(0.5,1.0); axs[1].set_ylabel("rolling coverage (100)"); axs[1].set_xlabel("time step"); axs[1].legend(fontsize=7)
for c in cps: axs[1].axvline(c,color="k",ls=":",lw=0.7)
plt.tight_layout(); plt.savefig(f"{FIG}/fig4_s1_trace.png",dpi=600); plt.close()

# ---- Fig 5 MachineTemp trace
st=pickle.load(open(DATA/"streams.pkl","rb")); y,yh=st["MachineTemp"]; s=np.abs(y-yh)
qa=aci(s,0.1,E.N0); qm=mars_cp(s,0.1,E.N0,inner="aci",seed=0); al=mars_cp.last.alarms
t=np.arange(E.N0,len(s)); cap=np.percentile(s,99.9)
fig,ax=plt.subplots(figsize=(7.2,2.6))
ax.plot(t,s[E.N0:],".",ms=0.6,color="#bbb",label="score")
ax.plot(t,np.minimum(qa,cap),lw=0.8,color="#d62728",label="ACI"); ax.plot(t,np.minimum(qm,cap),lw=0.8,color="#1f77b4",label="MARS(ACI)")
ax.plot(al,[cap*0.97]*len(al),"v",ms=3,color="#1f77b4",label="alarms"); ax.set_ylim(0,cap); ax.set_xlabel("time step (5-min)"); ax.set_ylabel("half-width")
ax.legend(ncol=4,fontsize=7,loc="lower center",bbox_to_anchor=(0.5,1.0),frameon=False); plt.tight_layout(); plt.savefig(f"{FIG}/fig5_machine_trace.png",dpi=600); plt.close()

# ---- Fig 6 relative change wrapper vs base (all datasets)
allr=pd.concat([syn,real]); dsl=["S1-RecurringAbrupt","S2-Gradual","S4-MixedRecurring","ELEC2","Power","MelbTemp","MachineTemp"]
g=allr.groupby(["ds","m"])[["width","LCE"]].mean()
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9),sharey=True)
xs=np.arange(len(dsl)); bw=0.26; cols={"ACI":"#d62728","QT":"#2ca02c","DtACI":"#9467bd"}
for j,met in enumerate(["width","LCE"]):
    for i,b in enumerate(["ACI","QT","DtACI"]):
        v=[100*(g.loc[(d,f"MARS({b})"),met]-g.loc[(d,b),met])/g.loc[(d,b),met] for d in dsl]
        axs[j].barh(xs+(i-1)*bw,v,height=bw,color=cols[b],label=f"MARS({b}) vs {b}")
    axs[j].axvline(0,color="k",lw=0.7); axs[j].set_xlabel(f"relative change in {'mean width' if met=='width' else 'LCE'} (%)")
axs[0].set_yticks(xs); axs[0].set_yticklabels([d.replace("-RecurringAbrupt","").replace("-MixedRecurring","").replace("-Gradual","") for d in dsl]); axs[0].invert_yaxis()
fig.legend(*axs[0].get_legend_handles_labels(),fontsize=7,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(0.55,1.04)); plt.tight_layout(); plt.savefig(f"{FIG}/fig6_relative.png",dpi=600,bbox_inches="tight"); plt.close()

# ---- Fig 7 sensitivity
S=pickle.load(open(RESULTS/"res_sens.pkl","rb")); dd=pd.DataFrame(S["delta"])
fig,axs=plt.subplots(1,3,figsize=(7.2,2.4))
for st_,c in [("S1","#1f77b4"),("MachineTemp","#ff7f0e")]:
    z=dd[dd.stream==st_].groupby("delta")[["LCE","width","alarms"]].mean()
    axs[0].plot(z.index,z.LCE,"o-",color=c,label=st_); axs[1].plot(z.index,z.width/z.width.iloc[1],"o-",color=c,label=st_)
z3=dd[dd.stream=="S3"].groupby("delta")["alarms"].mean(); axs[2].plot(z3.index,z3.values,"o-",color="k",label="S3 (stationary)")
for a in axs: a.set_xscale("log"); a.set_xlabel(r"$\delta$")
axs[0].set_ylabel("LCE"); axs[1].set_ylabel(r"width / width($\delta$=0.01)"); axs[2].set_ylabel("false alarms / stream"); axs[0].legend(fontsize=7); axs[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/fig7_sensitivity.png",dpi=600); plt.close()
print("figs ok")
