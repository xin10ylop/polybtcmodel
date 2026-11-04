"""Does the model know more than the Polymarket price at the same instant? (post-reference offsets)
AUC of outcome vs market price, vs model, vs combination; and realized frequency by (market, model) cells.
Restricted to the TUNE period unless a period is given."""
import os
import numpy as np, pandas as pd, sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim
period = sys.argv[1:3] if len(sys.argv) >= 3 else ("2026-08-20", "2026-09-30")
m = pd.read_parquet("/home/user/data/master.parquet")
mk, g = polysim.load_windows()
W, grid = polysim.path_grid(g, -60, 300)                       # col j = rel -60 + j
keep = np.isin(W, m.index.values); W, grid = W[keep], grid[keep]
mm = m.loc[W]
sel = (mm.date.values >= period[0]) & (mm.date.values <= period[1])
W, grid, mm = W[sel], grid[sel], mm[sel]
y = mm.up.values
print("period", period, "windows", len(W))
def logit(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
rows = []
for d in (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 210, 240):
    mkt = grid[:, 60 + d]
    for model, col in (("m2", f"m2_{d}_close_p"), ("bm", f"bm_{d}_close_p")):
        if col not in mm.columns:
            continue
        mod = mm[col].values
        ok = np.isfinite(mkt) & np.isfinite(mod)
        if ok.sum() < 300:
            continue
        X = np.column_stack([logit(mkt[ok]), logit(mod[ok])])
        # 2-fold by alternating days to get out-of-fold combination AUC inside the period
        days = mm.date.values[ok]; ud = np.unique(days); f1 = np.isin(days, ud[::2])
        p = np.full(ok.sum(), np.nan)
        for tr, te in ((f1, ~f1), (~f1, f1)):
            lr = LogisticRegression(C=10).fit(X[tr], y[ok][tr]); p[te] = lr.predict_proba(X[te])[:, 1]
        lr = LogisticRegression(C=10).fit(X, y[ok])
        rows.append(dict(d=d, model=model, n=int(ok.sum()), auc_market=roc_auc_score(y[ok], mkt[ok]), auc_model=roc_auc_score(y[ok], mod[ok]),
                         auc_combo_oof=roc_auc_score(y[ok], p), w_market=lr.coef_[0][0], w_model=lr.coef_[0][1],
                         ll_market=-np.mean(y[ok] * np.log(np.clip(mkt[ok], 1e-3, 1)) + (1 - y[ok]) * np.log(np.clip(1 - mkt[ok], 1e-3, 1))),
                         ll_combo=-np.mean(y[ok] * np.log(p) + (1 - y[ok]) * np.log(1 - p))))
pd.set_option("display.width", 220)
print(pd.DataFrame(rows).round(4).to_string(index=False))
# mispricing cells at a few offsets: realized frequency when model - market > delta
print("\n== realized P(up) when model disagrees with market by > delta (model higher) / (model lower) ==")
for d in (-15, -5, 0, 10, 30, 60, 120):
    mkt = grid[:, 60 + d]; col = f"m2_{d}_close_p"
    if col not in mm.columns: continue
    mod = mm[col].values; ok = np.isfinite(mkt) & np.isfinite(mod)
    out = []
    for delta in (0.03, 0.05, 0.08, 0.12):
        hi = ok & (mod - mkt > delta); lo = ok & (mkt - mod > delta)
        out.append(f"d>{delta}: n={hi.sum():4d} mkt={np.mean(mkt[hi]) if hi.sum() else np.nan:.3f} real={np.mean(y[hi]) if hi.sum() else np.nan:.3f} | n={lo.sum():4d} mkt={np.mean(mkt[lo]) if lo.sum() else np.nan:.3f} real={np.mean(y[lo]) if lo.sum() else np.nan:.3f}")
    print(f"offset {d:+d}: " + " || ".join(out))
