"""Production btc-predict core booster (spot+perp+ETH features) for the live agreement filter:
trained on every minute of the local archive (2026-02 -> 2026-11-03). Writes models/bp_{key}_core.txt + features json."""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/btc-predict"); os.chdir("/home/user/btc-predict")
from btcpredict.features import panel as P, subminute as SM, matrix as F
from btcpredict.modeling import gbm as M
from btcpredict.config import FEAT, FEATURE_SETS
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"); os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(FEAT / "panel_exp.parquet"); df = df[df["ts"] >= "2026-02-03"].reset_index(drop=True)
X = pd.concat([F.model_matrix(df), F.extra_features(df)], axis=1)
for key, tok, hmin in (("s5", "5", 5), ("s1", "1", 1)):
    cols = [c for c in open(FEATURE_SETS / ("core.txt" if hmin == 1 else "core_fast.txt")).read().split() if c in X.columns]
    yz = df[f"y_s{tok}"] / F.target_scale(df, hmin)
    tr = np.where(np.isfinite(yz.values))[0]
    fp = M.lgbm_factory(rounds=400); t = time.time()
    fp(X[cols].iloc[tr], yz.iloc[tr], X[cols].iloc[-5:], None)
    fp.last_model.save_model(os.path.join(OUT, f"bp_{key}_core.txt"))
    json.dump(cols, open(os.path.join(OUT, f"bp_{key}_core_features.json"), "w"))
    print(key, len(cols), "features", len(tr), "rows", f"{time.time()-t:.0f}s", "trained through", df["ts"].iloc[tr[-1]])
