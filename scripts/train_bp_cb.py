"""btc-predict core + Coinbase-premium satellite features, walk-forward (2-month blocks, trained only on
data before each block) -> z forecasts for s5/s1 at every minute; compared with the core-only forecasts."""
import os, sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/btc-predict"); os.chdir("/home/user/btc-predict")
from btcpredict.features import subminute as SM, matrix as F, coinbase as CB
from btcpredict.modeling import gbm as M, evaluate as E
from btcpredict.config import FEAT, FEATURE_SETS, SEC
MONTHS = ["2026-08", "2026-09", "2026-10", "2026-11"]
OUT = "/home/user/data/forecasts"
df = pd.read_parquet(FEAT / "panel_exp.parquet"); df = df[df["ts"] >= "2026-02-03"].reset_index(drop=True)
cb = pd.read_parquet(SEC / "coinbase" / "coinbase_1m.parquet")
Xcb = CB.features(df[["ts", "s_close", "sigma", "s_vol60"]], cb)
print("Xcb", Xcb.shape, "nan share", float(Xcb.isna().mean().mean()))
X = pd.concat([F.model_matrix(df), F.extra_features(df), Xcb.reset_index(drop=True)], axis=1)
res = {"ts": df["ts"].values}
for key, tok, hmin in (("s5", "5", 5), ("s1", "1", 1)):
    core = [c for c in open(FEATURE_SETS / ("core.txt" if hmin == 1 else "core_fast.txt")).read().split() if c in X.columns]
    cols = core + [c for c in Xcb.columns if c in X.columns]
    yz = df[f"y_s{tok}"] / F.target_scale(df, hmin)
    folds = E.make_folds(df["ts"], MONTHS, purge_minutes=hmin + 10, block=2)
    for nm, cc in (("core", core), ("core+cb", cols)):
        t = time.time(); pred = E.walk_forward(M.lgbm_factory(rounds=400), X[cc], yz, folds, verbose=False)
        res[f"z_{key}_{nm}"] = pred.values; m = pred.notna().values
        r = E.metrics(df[f"y_s{tok}"].values[m], pred.values[m]); print(key, nm, len(cc), "feats", {k: round(v, 4) for k, v in r.items() if k in ("ic", "hit", "auc", "hit_top10")}, f"{time.time()-t:.0f}s", flush=True)
pd.DataFrame(res).to_parquet(f"{OUT}/bp_cb_minute.parquet", index=False); print("saved")
