"""[July out-of-sample fold] btc-predict (core variant: BTC spot + perp + ETH, no cross-venue satellites) retrained HONESTLY
for the Polymarket period: walk-forward, 2-month blocks, training only on data before each block.
Original models/ and feature code untouched; everything here only calls btcpredict's public functions.
Outputs: /home/user/data/forecasts/bp_minute_july.parquet  (one row per minute boundary T with z forecasts)
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/btc-predict")
os.chdir("/home/user/btc-predict")
from btcpredict.features import panel as P, subminute as SM, matrix as F
from btcpredict.modeling import gbm as M, evaluate as E
from btcpredict.config import FEAT, FEATURE_SETS

START, END = "2026-02-01", "2026-11-03"
MONTHS = ["2026-07", "2026-08"]      # one 2-month block: trained on 2026-02-03 .. 2026-06-30, scored on Jul-Aug
OUT = "/home/user/data/forecasts"; os.makedirs(OUT, exist_ok=True)

t0 = time.time()
pp = FEAT / "panel_exp.parquet"
if pp.exists():
    df = pd.read_parquet(pp)
else:
    df = P.build(START, END, workers=3, out_name="panel_exp")
print("panel", df.shape, f"{time.time()-t0:.0f}s", flush=True)
df = df[df["ts"] >= "2026-02-03"].reset_index(drop=True)
ys = SM.build(START, END)
df = df.merge(ys, on="ts", how="left")
X = pd.concat([F.model_matrix(df), F.extra_features(df)], axis=1)
print("X", X.shape, f"{time.time()-t0:.0f}s", flush=True)
res = {"ts": df["ts"].values, "sigma": df["sigma"].values, "s_close": df["s_close"].values, "f_close": df["f_close"].values}
for tok in ["1", "5"]:
    for venue in ["s", "f"]:
        if venue == "f":
            continue
        sub = tok.endswith("s")
        hmin = int(tok[:-1]) / 60 if sub else int(tok)
        h = 1 if sub else int(tok)
        key = f"{venue}{tok}"
        cols_file = FEATURE_SETS / ("core.txt" if h == 1 else "core_fast.txt")
        cols = [c for c in open(cols_file).read().split() if c in X.columns]
        scale = F.target_scale(df, hmin)
        yz = df[f"y_{venue}{tok}"] / scale
        folds = E.make_folds(df["ts"], MONTHS, purge_minutes=h + 10, block=2)
        fp = M.lgbm_factory(rounds=400)
        pred = E.walk_forward(fp, X[cols], yz, folds, verbose=True)
        res[f"z_{key}"] = pred.values
        res[f"y_{key}"] = df[f"y_{venue}{tok}"].values
        m = pred.notna().values
        r = E.metrics(df[f"y_{venue}{tok}"].values[m], pred.values[m])
        print(key, len(cols), "feats", {k: round(v, 4) for k, v in r.items()}, f"{time.time()-t0:.0f}s", flush=True)
pd.DataFrame(res).to_parquet(f"{OUT}/bp_minute_july.parquet", index=False)
print("done")
