"""Forecast skill of each model/decision offset against the actual Polymarket outcome."""
import numpy as np, pandas as pd, glob, sys
from sklearn.metrics import roc_auc_score, log_loss
which = sys.argv[1] if len(sys.argv) > 1 else "m2"
fc = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"/home/user/data/forecasts/{which}_*.parquet"))], ignore_index=True)
mk = pd.read_parquet("/home/user/data/polymarket/markets.parquet").set_index("window_start")
mk = mk[mk.outcome_up_price.isin([0.0, 1.0])]
fc = fc[fc["T"].isin(mk.index)].copy()
fc["up"] = (mk.loc[fc["T"].values, "outcome_up_price"].values == 1.0).astype(int)
tw = pd.read_parquet("/home/user/data/truth/windows.parquet").set_index("T")
print(which, "rows", len(fc), "windows", fc["T"].nunique(), "range", pd.to_datetime(fc["T"].min(), unit="s"), pd.to_datetime(fc["T"].max(), unit="s"))
cols = [c for c in fc.columns if c.endswith(("_p", "_pup_now", "_pm", "_pd", "_praw", "_pst")) and ("close" in c or "h60" in c or "h30" in c or "h10" in c)]
rows = []
for d, sub in fc.groupby("d"):
    r = {"d": d, "n": len(sub)}
    for c in cols:
        v = sub[c].values; ok = np.isfinite(v)
        if ok.sum() < 200 or len(np.unique(sub.up.values[ok])) < 2:
            continue
        r[c] = roc_auc_score(sub.up.values[ok], v[ok])
    rows.append(r)
res = pd.DataFrame(rows).set_index("d")
pd.set_option("display.width", 250)
print("AUC vs Polymarket outcome (UP resolves) by decision offset d (seconds from nominal open):")
print(res.round(4).to_string())
# hit rates at confidence thresholds for the main candidates
for c in [c for c in ["close_p", "close_pup_now", "h60_pup_now", "h30_pup_now", "close_praw", "h60_p"] if c in fc.columns]:
    print(f"\n{c}: by d, accuracy and coverage at |p-0.5| >= thr")
    rows = []
    for d, sub in fc.groupby("d"):
        v = sub[c].values; ok = np.isfinite(v)
        if ok.sum() < 200:
            continue
        r = {"d": d}
        for thr in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
            m = ok & (np.abs(v - 0.5) >= thr)
            r[f"acc@{thr}"] = np.mean((v[m] > 0.5) == (sub.up.values[m] == 1)) if m.sum() > 50 else np.nan
            r[f"cov@{thr}"] = m.sum() / ok.sum()
        rows.append(r)
    print(pd.DataFrame(rows).set_index("d").round(3).to_string())
