"""Online conformal prediction methods operating on a stream of nonconformity scores
s_t = |y_t - yhat_t|. Every method outputs a threshold q_t BEFORE s_t is revealed;
the interval is [yhat_t - q_t, yhat_t + q_t] and err_t = 1{s_t > q_t}."""
import numpy as np
from bisect import insort, bisect_left, bisect_right
from collections import deque
from scipy.stats import ks_2samp

INF = np.inf

def cq(scores, a):
    """Split-conformal quantile with finite-sample correction at miscoverage a."""
    n = len(scores)
    if a <= 0 or n == 0:
        return INF
    if a >= 1:
        return 0.0
    k = int(np.ceil((n + 1) * (1 - a)))
    if k > n:
        return INF
    return float(np.partition(np.asarray(scores), k - 1)[k - 1])

# ---------------------------------------------------------------- baselines
def split_cp(s, alpha, n0, **kw):
    cal = s[:n0]; q = cq(cal, alpha)
    return np.full(len(s) - n0, q)

def rolling_cp(s, alpha, n0, W=500, **kw):
    out = np.empty(len(s) - n0)
    for i, t in enumerate(range(n0, len(s))):
        out[i] = cq(s[max(0, t - W):t], alpha)
    return out

def aci(s, alpha, n0, W=500, gamma=0.005, **kw):
    """Adaptive Conformal Inference (Gibbs & Candes, 2021)."""
    a = alpha; out = np.empty(len(s) - n0)
    for i, t in enumerate(range(n0, len(s))):
        q = cq(s[max(0, t - W):t], a); out[i] = q
        a += gamma * (alpha - float(s[t] > q))
    return out

def dtaci(s, alpha, n0, W=500, I=100, gammas=(0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128), **kw):
    """Dynamically-tuned ACI (Gibbs & Candes, JMLR 2024), weighted-average output."""
    g = np.array(gammas); k = len(g)
    sigma = 1.0 / (2 * I)
    eta = np.sqrt(3.0 / I) * np.sqrt((np.log(k * I) + 2) / ((1 - alpha) ** 2 * alpha ** 3 + alpha ** 2 * (1 - alpha) ** 3))
    ai = np.full(k, alpha); w = np.ones(k) / k
    out = np.empty(len(s) - n0)
    for i, t in enumerate(range(n0, len(s))):
        cal = np.asarray(s[max(0, t - W):t])
        a = float(np.dot(w, ai)); q = cq(cal, a); out[i] = q
        beta = (np.sum(cal >= s[t]) + 1) / (len(cal) + 1)  # largest alpha still covering
        loss = alpha * (beta - ai) - np.minimum(0.0, beta - ai)
        w = w * np.exp(-eta * loss); w /= w.sum(); w = (1 - sigma) * w + sigma / k
        ai = ai + g * (alpha - (beta < ai).astype(float))
    return out

def quantile_tracker(s, alpha, n0, lr=0.05, B=100, **kw):
    """Quantile tracking / P-control step of Conformal PID (Angelopoulos et al., 2023)."""
    q = cq(s[:n0], alpha); out = np.empty(len(s) - n0)
    for i, t in enumerate(range(n0, len(s))):
        out[i] = q
        scale = np.max(s[max(0, t - B):t])
        q = q + lr * scale * (float(s[t] > q) - alpha)
    return out

# ---------------------------------------------------------------- proposed
class Jumper:
    """Simple-jumper conformal test martingale (Vovk et al., 2021).
    eps=-1 bets on small p-values (scores grow), eps=+1 on large (scores shrink)."""
    def __init__(self, J=0.01, eps=(-1.0, 0.0, 1.0), power=(), two_sided=True):
        self.J = J; self.eps = list(eps)
        # optional power-betting states  k p^(k-1) (low side) and k (1-p)^(k-1) (high side)
        self.pw = [(k, 0) for k in power] + ([(k, 1) for k in power] if two_sided else [])
        self.reset()
    def reset(self):
        n = len(self.eps) + len(self.pw)
        self.C = np.ones(n) / n; self.logM = 0.0
    def bet(self, p):
        p = min(max(p, 1e-12), 1 - 1e-12)
        f = [1 + e * (p - 0.5) for e in self.eps]
        f += [k * (p if side == 0 else 1 - p) ** (k - 1) for k, side in self.pw]
        return np.array(f)
    def update(self, p):
        tot = self.C.sum()
        self.C = (1 - self.J) * self.C + self.J * tot / len(self.C)
        self.C = self.C * self.bet(p)
        m = self.C.sum(); self.logM += np.log(m); self.C /= m
        return self.logM

class MARSCP:
    """MARS-CP: Martingale-triggered, Adaptive, Regime-memory Split conformal wrapper.
    inner in {'aci','qt','dtaci'} is the within-regime online updater being wrapped."""
    GAMMAS = np.array([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128])
    def __init__(self, alpha=0.1, W=500, gamma=0.005, delta=0.01, J=0.01, n_min=20,
                 n_match=30, recall_p=0.2, bank_size=8, two_sided=True, memory=True,
                 localize=True, reset=True, power=(), inner="aci", lr=0.05, B=100, I=100, seed=0):
        self.__dict__.update(locals()); del self.__dict__["self"]
        self.rng = np.random.default_rng(seed)
        self.mart = Jumper(J, (-1.0, 0.0, 1.0) if two_sided else (-1.0, 0.0), power, two_sided)
        self.bank = []; self.alarms = []; self.recalls = []
        k = len(self.GAMMAS); a = alpha
        self.sigma = 1.0 / (2 * I)
        self.eta = np.sqrt(3.0 / I) * np.sqrt((np.log(k * I) + 2) / ((1 - a) ** 2 * a ** 3 + a ** 2 * (1 - a) ** 3))

    # ---- inner-updater state
    def _init_inner(self, a=None):
        a = self.alpha if a is None else a
        self.a = a
        self.ai = np.full(len(self.GAMMAS), a)
        if not hasattr(self, "w"):                  # expert weights persist across regimes
            self.w = np.ones(len(self.GAMMAS)) / len(self.GAMMAS)
        self.q = self._cal_q(a)
    def _cal(self):
        return (self.mem + self.seg)[-self.W:]
    def _cal_q(self, a):
        cal = self._cal(); q = cq(cal, a)
        if q == INF and cal: q = max(cal)          # tiny post-change set: use its max
        return q

    def start(self, init_scores):
        self.seg = list(init_scores)[-self.W:]; self.hist = sorted(init_scores)
        self.segbuf = list(init_scores); self.path = [0.0]
        self.prev = None; self.recalled = False; self.mem = []; self.recent = deque(init_scores[-self.B:], maxlen=self.B)
        self._init_inner()

    def threshold(self):
        if self.inner == "qt":
            q = self.q
        elif self.inner == "dtaci":
            q = cq(self._cal(), float(np.dot(self.w, self.ai)))
        else:
            q = cq(self._cal(), self.a)
        if self.prev is not None and len(self.seg) < self.n_min and not self.mem:
            q = max(min(q, max(self.seg)) if self.seg else 0.0, cq(self.prev, self.alpha))  # conservative bridge
        return q

    def _inner_update(self, s, q, err):
        if self.inner == "qt":
            self.q = self.q + self.lr * max(self.recent) * (err - self.alpha)
        elif self.inner == "dtaci":
            cal = np.asarray(self._cal())
            beta = (np.sum(cal >= s) + 1) / (len(cal) + 1)
            loss = self.alpha * (beta - self.ai) - np.minimum(0.0, beta - self.ai)
            self.w = self.w * np.exp(-self.eta * loss); self.w /= self.w.sum()
            self.w = (1 - self.sigma) * self.w + self.sigma / len(self.w)
            self.ai = self.ai + self.GAMMAS * (self.alpha - (beta < self.ai).astype(float))
        else:
            self.a += self.gamma * (self.alpha - err)

    def observe(self, s, q, t):
        err = float(s > q)
        self.recent.append(s)
        self._inner_update(s, q, err)
        n = len(self.hist)
        hi = bisect_right(self.hist, s); lo = bisect_left(self.hist, s)
        p = (n - hi + self.rng.uniform() * (hi - lo + 1)) / (n + 1)
        lm = self.mart.update(p)
        insort(self.hist, s); self.segbuf.append(s); self.path.append(lm)
        self.seg.append(s); self.seg = self.seg[-self.W:]
        # regime-memory recall (one attempt per regime)
        if self.memory and not self.recalled and len(self.segbuf) >= self.n_match and self.bank:
            self.recalled = True
            pv = [ks_2samp(self.segbuf, bs).pvalue for bs, _ in self.bank]
            j = int(np.argmax(pv))
            if pv[j] > self.recall_p:
                bs, ba = self.bank.pop(j); self.mem = list(bs); self.recalls.append(t)
                self._init_inner(ba if self.inner == "aci" else None)
        # change alarm: Ville's inequality bounds false-alarm prob. by delta
        if self.reset and lm >= np.log(1.0 / self.delta):
            self.alarms.append(t)
            off = len(self.segbuf) - (len(self.path) - 1)
            c = off + int(np.argmin(self.path)) if self.localize else max(0, len(self.segbuf) - 20)
            old, new = self.segbuf[:c], self.segbuf[c:][-self.W:]
            oldcal = (self.mem + old)[-self.W:] if old else list(self.seg)
            if self.memory and len(oldcal) >= self.n_match:
                self.bank.append((oldcal, float(np.clip(self.a, 0.0, 1.0))))
                self.bank = self.bank[-self.bank_size:]
            self.prev = oldcal
            self.seg = list(new); self.segbuf = list(new); self.hist = sorted(new)
            self.path = [0.0]; self.mart.reset(); self.recalled = False; self.mem = []
            self._init_inner()
        return err

def mars_cp(s, alpha, n0, **kw):
    m = MARSCP(alpha=alpha, **kw); m.start(list(s[:n0]))
    out = np.empty(len(s) - n0)
    for i, t in enumerate(range(n0, len(s))):
        q = m.threshold(); out[i] = q
        m.observe(float(s[t]), q, t)
    mars_cp.last = m
    return out
