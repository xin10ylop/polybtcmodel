"""Value of knowing the EXACT reference (Chainlink priceToBeat, published by Polymarket) vs the market.
Displacement from the exact reference, scaled by remaining-time volatility, as an extra signal."""
import os
import numpy as np, pandas as pd, sys, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
period = sys.argv[1:3] if len(sys.argv) >= 3 else ("2026-08-20", "2026-09-30")
m = pd.read_parquet("/home/user/data/master.parquet")
paths = np.load("/home/user/data/truth/paths.npy"); tw = pd.read_parquet("/home/user/data/truth/windows.parquet")
ti = {t: i for i, t in enumerate(tw["T"].values)}
m = m[np.isfinite(m.price_to_beat)]
rows = np.array([ti[t] for t in m.index]); lp = np.log(paths[rows])            # col 120 = T
# causal basis estimate: median over the previous 288 windows of log(binance[T-30]/ptb)
raw = np.log(m.ref30.values) - np.log(m.price_to_beat.values)
basis = pd.Series(raw, index=m.index).rolling(288, min_periods=50).median().shift(1).values
m["basis_bp"] = basis * 1e4
mk, g = polysim.load_windows()
W, grid = polysim.path_grid(g, -60, 300)
pos = {w: i for i, w in enumerate(W)}
keep = np.array([t in pos for t in m.index]); m = m[keep]; lp = lp[keep]; basis = basis[keep]
grid = grid[[pos[t] for t in m.index]]
sel = (m.date.values >= period[0]) & (m.date.values <= period[1]) & np.isfinite(basis)
y = m.up.values
print("period", period, "windows", sel.sum(), " basis bp median", np.nanmedian(m.basis_bp.values[sel]).round(2))
def logit(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
sig5 = m.vol_prev5m_bp.values / 1e4
out = []
for d in (0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240):
    S = lp[:, 120 + d - 1]                                   # last known close before decision second
    x_exact = S - np.log(m.price_to_beat.values) - basis
    x_proxy = S - lp[:, 120 - 30]
    tau = 300 - 30 - d                                       # effective remaining seconds (end ref at ~T+270)
    z_exact = x_exact / (np.maximum(sig5, 1e-5) * np.sqrt(max(tau, 5) / 300))
    z_proxy = x_proxy / (np.maximum(sig5, 1e-5) * np.sqrt(max(tau, 5) / 300))
    mkt = grid[:, 60 + d]
    mod = m[f"m2_{d}_close_p"].values if f"m2_{d}_close_p" in m.columns else np.full(len(m), np.nan)
    ok = sel & np.isfinite(mkt) & np.isfinite(mod) & np.isfinite(z_exact)
    days = m.date.values[ok]; ud = np.unique(days); f1 = np.isin(days, ud[::2])
    def oof(X):
        p = np.full(ok.sum(), np.nan)
        for tr, te in ((f1, ~f1), (~f1, f1)):
            lr = LogisticRegression(C=10).fit(X[tr], y[ok][tr]); p[te] = lr.predict_proba(X[te])[:, 1]
        return p
    Xm = logit(mkt[ok])[:, None]
    r = dict(d=d, n=int(ok.sum()), auc_mkt=roc_auc_score(y[ok], mkt[ok]), auc_zexact=roc_auc_score(y[ok], z_exact[ok]), auc_zproxy=roc_auc_score(y[ok], z_proxy[ok]),
             auc_model=roc_auc_score(y[ok], mod[ok]))
    p1 = oof(np.column_stack([Xm, z_proxy[ok]])); p2 = oof(np.column_stack([Xm, z_exact[ok]])); p3 = oof(np.column_stack([Xm, z_exact[ok], logit(mod[ok])]))
    r.update(auc_mkt_zproxy=roc_auc_score(y[ok], p1), auc_mkt_zexact=roc_auc_score(y[ok], p2), auc_mkt_zexact_model=roc_auc_score(y[ok], p3))
    # realized frequency when the exact-ref stack disagrees with the market
    for delta in (0.05, 0.10):
        hi = p2 - mkt[ok] > delta; lo = mkt[ok] - p2 > delta
        r[f"n_hi{delta}"] = int(hi.sum()); r[f"edge_hi{delta}"] = float(np.mean(y[ok][hi]) - np.mean(mkt[ok][hi])) if hi.sum() > 20 else np.nan
        r[f"n_lo{delta}"] = int(lo.sum()); r[f"edge_lo{delta}"] = float(np.mean(mkt[ok][lo]) - np.mean(y[ok][lo])) if lo.sum() > 20 else np.nan
    out.append(r)
print(pd.DataFrame(out).round(4).to_string(index=False))
