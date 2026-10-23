import matplotlib; matplotlib.use("Agg")
from paths import FIGURES
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family":"DejaVu Sans"})
fig,ax=plt.subplots(figsize=(7.6,4.2)); ax.set_xlim(0,120); ax.set_ylim(0,66); ax.axis("off")
def box(x,y,w,h,t,c,fs=7.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.5",fc=c,ec="#333",lw=0.9))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs)
def arr(p,q,t="",dx=0,dy=1.5,style="->"):
    ax.annotate("",xy=q,xytext=p,arrowprops=dict(arrowstyle=style,lw=0.9,color="#333",shrinkA=0,shrinkB=0))
    if t: ax.text((p[0]+q[0])/2+dx,(p[1]+q[1])/2+dy,t,fontsize=6.3,ha="center",color="#555",style="italic")
# row 1: prediction path
box(1,50,17,11,"Data stream\n$(X_t,Y_t)$","#e8f0fb")
box(23,50,20,11,"Black-box\nforecaster $\\hat f_t$","#e8f0fb")
box(48,50,24,11,"Nonconformity score\n$s_t=|Y_t-\\hat f_t(X_t)|$","#e8f0fb")
box(78,50,40,11,"Inner online updater\n(ACI / QT / DtACI) on regime\nbuffer $\\mathcal{B}_k$ $\\rightarrow$ threshold $q_t$","#fff3d6")
arr((18.5,55.5),(22.5,55.5)); arr((43.5,55.5),(47.5,55.5)); arr((72.5,55.5),(77.5,55.5))
# output
box(78,33,40,9,"Prediction interval\n$\\hat C_t(X_t)=[\\hat f_t(X_t)-q_t,\\ \\hat f_t(X_t)+q_t]$","#eeeeee")
arr((98,49.5),(98,42.5))
# row 2: monitoring
box(48,30,24,12,"Conformal p-value\n$p_t$ of $s_t$ w.r.t.\ncurrent-regime history","#e9f7ec")
box(1,30,40,12,"Two-sided simple-jumper conformal\ntest martingale $M_t$;\nalarm when $M_t\\geq 1/\\delta$ (Ville)","#e9f7ec")
arr((60,49.5),(60,42.5)); arr((47.5,36),(41.5,36))
# row 3: response
box(1,8,26,13,"Change-point\nlocalisation\n$\\hat c=\\arg\\min_{j}\\ \\log M_j$","#fde8e8")
box(33,8,30,13,"Regime reset: new buffer\n$\\mathcal{B}_{k+1}=\\{s_j\\}_{j>\\hat c}$,\nconservative bridge,\nre-initialise updater","#fde8e8")
box(70,8,48,13,"Regime-memory bank (optional)\nstore closed regimes; KS-test matching\nof new regime $\\rightarrow$ recall buffer + state","#f1e8fb")
arr((14,29.5),(14,21.5),"alarm",dx=4,dy=0)
arr((27.5,14.5),(32.5,14.5)); arr((63.5,14.5),(69.5,14.5),"store / recall",dy=3.2)
ax.plot([48,48],[21.5,26],color="#333",lw=0.9); ax.plot([48,75],[26,26],color="#333",lw=0.9)
ax.plot([75,75],[26,31],color="#333",lw=0.9); arr((75,31),(75,49.5))
ax.text(61,27.3,"re-initialised $\\mathcal{B}_{k+1}$, $\\alpha_t$",fontsize=6.3,ha="center",color="#555",style="italic")


ax.text(60,64.5,"prediction path",fontsize=7,ha="center",color="#1f4e8c",weight="bold")
ax.text(21,45,"monitoring path",fontsize=7,ha="center",color="#1c6b35",weight="bold")
ax.text(60,3,"response path (triggered only on alarm)",fontsize=7,ha="center",color="#8c1f1f",weight="bold")
plt.savefig(str(FIGURES)+"/fig1_framework.png",dpi=600,bbox_inches="tight")
