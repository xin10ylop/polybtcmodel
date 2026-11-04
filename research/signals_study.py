"""Which pre-reference signals (and combinations) predict the effective-window direction?
Labels: Binance effective move sign (model-side, no Polymarket data) and the Polymarket outcome (evaluation only).
Stacker fitted on TUNE only; reported on TEST (and OLD regime)."""
import numpy as np, pandas as pd, sys, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
TUNE = ("2026-08-20", "2026-09-30"); TEST = ("2026-10-01", "2026-11-03"); OLD = ("2026-08-01", "2026-08-19")
m = pd.read_parquet("/home/user/data/master.parquet")
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def period(name):
    a, b = {"TUNE": TUNE, "TEST": TEST, "OLD": OLD}[name]; return (m.date.values >= a) & (m.date.values <= b)
pd.set_option("display.width", 250)
yb = m.up_binance_eff.values; yp = m.up.values
print("windows", len(m), "| label agreement", (yb == yp).mean().round(4))
cands = []
for d in (-60, -50, -45, -40, -35, -30):
    for c in ("m2_%d_close_p", "m2_%d_close_pup_now", "m2_%d_h60_pup_now", "m2_%d_h30_pup_now", "m2_%d_h10_pup_now",
              "bm_%d_close_p", "bm_%d_close_pup_now", "bm_%d_h60_pup_now", "bm_%d_h30_pup_now", "bm_%d_h10_pup_now"):
        col = c % d
        if col in m.columns: cands.append(col)
for off in (-120, -60):
    for k in ("z_s5", "z_s2", "z_s1", "z_f1", "z_s10s"):
        col = f"bp_{off}_{k}"
        if col in m.columns: cands.append(col)
rows = []
for c in cands:
    v = m[c].values; r = {"signal": c}
    for pn in ("OLD", "TUNE", "TEST"):
        s = period(pn) & np.isfinite(v)
        r[f"auc_bin_{pn}"] = roc_auc_score(yb[s], v[s]) if s.sum() > 300 else np.nan
        r[f"auc_pm_{pn}"] = roc_auc_score(yp[s], v[s]) if s.sum() > 300 else np.nan
    rows.append(r)
res = pd.DataFrame(rows).set_index("signal")
print("\n== single-signal AUC vs Binance effective label / Polymarket outcome ==")
print(res.round(4).to_string())
# stackers (logistic on logits / z), fitted on TUNE with the Binance label
def stack(cols, name):
    X = np.column_stack([logit(m[c].values) if not c.startswith("bp_") else m[c].values for c in cols])
    ok = np.isfinite(X).all(1)
    tr = period("TUNE") & ok
    lr = LogisticRegression(C=1.0).fit(X[tr], yb[tr])
    out = {"stack": name, "n_feat": len(cols)}
    for pn in ("OLD", "TUNE", "TEST"):
        s = period(pn) & ok
        p = lr.predict_proba(X[s])[:, 1]
        out[f"auc_bin_{pn}"] = roc_auc_score(yb[s], p); out[f"auc_pm_{pn}"] = roc_auc_score(yp[s], p)
        out[f"acc_pm_{pn}"] = np.mean((p > 0.5) == (yp[s] == 1))
        q = np.abs(p - 0.5) >= np.quantile(np.abs(p - 0.5), 0.7)
        out[f"acc_pm_top30_{pn}"] = np.mean((p[q] > 0.5) == (yp[s][q] == 1))
    out["coef"] = np.round(lr.coef_[0], 3).tolist()
    return out
stacks = []
for d in (-60, -45, -35):
    base = [c for c in (f"m2_{d}_close_p", f"bm_{d}_close_p") if c in m.columns]
    if len(base) == 2: stacks.append(stack(base, f"m2+bm close_p @{d}"))
    allc = [c for c in cands if c.endswith(("close_p", "close_pup_now", "h60_pup_now", "h30_pup_now")) and f"_{d}_" in c]
    if len(allc) >= 2: stacks.append(stack(allc, f"m2+bm all heads @{d}"))
    if f"bp_-60_z_s5" in m.columns and base:
        stacks.append(stack(base + ["bp_-60_z_s5", "bp_-60_z_s1"], f"m2+bm close_p @{d} + bp@-60"))
if stacks:
    print("\n== stackers (fit on TUNE, Binance label) ==")
    print(pd.DataFrame(stacks).round(4).to_string(index=False))
