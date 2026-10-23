import matplotlib; matplotlib.use("Agg")
from paths import FIGURES
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
plt.rcParams.update({"font.family":"DejaVu Sans","mathtext.fontset":"dejavusans"})
def box(ax,x,y,w,h,t,c,fs=7.2,ls="-"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.5",fc=c,ec="#222",lw=0.9,ls=ls))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs)
def arr(ax,p,q,t="",dx=0,dy=1.4,fs=6.6):
    ax.annotate("",xy=q,xytext=p,arrowprops=dict(arrowstyle="-|>",lw=0.9,color="#222",shrinkA=0,shrinkB=0,mutation_scale=8))
    if t: ax.text((p[0]+q[0])/2+dx,(p[1]+q[1])/2+dy,t,fontsize=fs,ha="center",color="#333",style="italic")
def line(ax,pts):
    xs,ys=zip(*pts); ax.plot(xs,ys,color="#222",lw=0.9)

# ---------------- Fig 2: regime lifecycle (response path) ----------------
fig,ax=plt.subplots(figsize=(7.4,4.6)); ax.set_xlim(0,120); ax.set_ylim(0,74); ax.axis("off")
box(ax,2,58,30,11,"Steady regime $k$\ninner updater on buffer $\\mathcal{B}_k$\nemits threshold $q_t$","#fff3d6")
box(ax,42,58,30,11,"Score $s_t$ revealed\nconformal p-value $p_t$\nagainst history $\\mathcal{H}_k$","#e9f7ec")
box(ax,82,58,34,11,"Two-sided jumper update\n$\\log M_t$ (states $\\varepsilon\\in\\{-1,0,+1\\}$)","#e9f7ec")
arr(ax,(32.5,63.5),(41.5,63.5)); arr(ax,(72.5,63.5),(81.5,63.5))
# decision diamond
cx,cy=99,44
ax.add_patch(Polygon([[cx,cy+8],[cx+15,cy],[cx,cy-8],[cx-15,cy]],closed=True,fc="#ffffff",ec="#222",lw=0.9))
ax.text(cx,cy,"$\\log M_t\\geq\\log(1/\\delta)$ ?",ha="center",va="center",fontsize=7)
arr(ax,(99,57.5),(99,52.2))
line(ax,[(84,44),(17,44)]); arr(ax,(17,44),(17,57.5)); ax.text(52,45.4,"no: keep regime $k$ (next step $t+1$)",fontsize=6.6,ha="center",style="italic",color="#333")
arr(ax,(99,35.8),(99,31.5)); ax.text(102,33.5,"yes: alarm",fontsize=6.6,style="italic",color="#333")
box(ax,82,20,34,11,"Change localisation\n$\\hat c=\\arg\\min_{j\\leq t}\\log M_j$","#fde8e8")
box(ax,42,20,30,11,"Close regime $k$\nkeep pre-$\hat c$ scores as $\mathcal{B}_{prev}$;\nbank them if memory is on","#f1e8fb")
box(ax,2,20,30,11,"Open regime $k{+}1$\n$\\mathcal{B}_{k+1},\\mathcal{H}_{k+1}\\leftarrow\\{s_j\\}_{j>\\hat c}$\nreset $M\\leftarrow1$","#fde8e8")
arr(ax,(81.5,25.5),(72.5,25.5)); arr(ax,(41.5,25.5),(32.5,25.5))
box(ax,2,2,30,11,"Re-initialise updater\nACI: $\\alpha_t\\leftarrow\\alpha$;  QT: $q_t\\leftarrow Q(\\mathcal{B};\\alpha)$\nDtACI: levels $\\leftarrow\\alpha$, keep weights","#fde8e8",fs=6.6)
box(ax,42,2,30,11,"Conservative bridge for\nfirst $n_{\\min}$ scores, Eq. (7)","#fde8e8")
box(ax,82,2,34,11,"Optional recall at $n_{\\mathrm{match}}$ scores\nKS test vs. bank; if $p>\\rho$:\nprepend scores, restore state","#f1e8fb",fs=6.8)
arr(ax,(17,19.5),(17,13.5)); arr(ax,(32.5,7.5),(41.5,7.5)); arr(ax,(72.5,7.5),(81.5,7.5))
line(ax,[(116.5,7.5),(119.5,7.5),(119.5,72),(0.6,72),(0.6,63.5)]); arr(ax,(0.6,63.5),(1.5,63.5))
ax.text(60,73,"regime $k{+}1$ becomes the steady regime",fontsize=6.6,ha="center",style="italic",color="#333")
plt.savefig(str(FIGURES)+"/fig2_regime_lifecycle.png",dpi=600,bbox_inches="tight"); plt.close()

# ---------------- Fig 3: experimental pipeline ----------------
fig,ax=plt.subplots(figsize=(7.4,4.0)); ax.set_xlim(0,120); ax.set_ylim(0,64); ax.axis("off")
box(ax,1,44,26,15,"Real streams\nELEC2, Power,\nMelbTemp, MachineTemp","#e8f0fb")
box(ax,1,22,26,15,"Synthetic streams S1–S4\n(abrupt, gradual,\nstationary, mixed)","#e8f0fb")
box(ax,33,44,24,15,"Lag and calendar\nfeatures","#e8f0fb")
box(ax,63,44,26,15,"Sliding-refit\nHistGB forecaster\n(refit 250, window 3000)","#e8f0fb")
box(ax,95,44,23,15,"Score stream\n$s_t=|Y_t-\\hat f_t(X_t)|$","#e8f0fb")
arr(ax,(27.5,51.5),(32.5,51.5)); arr(ax,(57.5,51.5),(62.5,51.5)); arr(ax,(89.5,51.5),(94.5,51.5))
line(ax,[(27.5,29.5),(106.5,29.5)]); arr(ax,(106.5,29.5),(106.5,43.5))
ax.text(91,26.6,"known forecaster\nand residual model",fontsize=6.6,ha="center",style="italic",color="#333")
box(ax,63,4,26,17,"MARS-CP wrappers\nMARS(ACI), MARS(QT),\nMARS(DtACI)","#fff3d6")
box(ax,33,4,24,17,"Baselines\nSplit CP, Rolling CP,\nACI, QT, DtACI","#eeeeee")
box(ax,95,4,23,17,"Metrics and statistics\ncoverage, width, Inf%,\nLCE, UCT, post-shift;\nWilcoxon, ablation,\nsensitivity","#e9f7ec",fs=6.6)
line(ax,[(106.5,43.5),(106.5,38),(45,38)]); arr(ax,(45,38),(45,21.5)); arr(ax,(76,38),(76,21.5))
ax.text(60,39.3,"identical score stream to every method",fontsize=6.6,ha="center",style="italic",color="#333")
arr(ax,(89.5,12.5),(94.5,12.5))
line(ax,[(45,3.5),(45,0.8),(92,0.8),(92,8)]); arr(ax,(92,8),(94.5,8))
box(ax,1,4,26,11,"Protocol: tuned on\nvalidation seeds 100–103,\nfrozen; test seeds 0–19","#f1e8fb",fs=6.6,ls="--")
plt.savefig(str(FIGURES)+"/fig3_pipeline.png",dpi=600,bbox_inches="tight"); plt.close()
print("ok")
