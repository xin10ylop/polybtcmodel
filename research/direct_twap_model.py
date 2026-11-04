"""Can a model trained directly on the TWAP-difference sign (the Polymarket-relevant target) beat the
frozen btcmovement2 close head at T-45?  Walk-forward by month, pure Binance data, features = the
btcmovement2 feature subset stored in the master table at d=-45 plus the three models' outputs."""
import os, sys, numpy as np, pandas as pd, warnings, lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
m = pd.read_parquet("/home/user/data/master.parquet")
paths = np.load("/home/user/data/truth/paths.npy"); tw = pd.read_parquet("/home/user/data/truth/windows.parquet")
ti = {t: i for i, t in enumerate(tw["T"].values)}; rows = np.array([ti[t] for t in m.index]); lp = np.log(paths[rows])
m["y_twap"] = (lp[:, 120 + 240:120 + 300].mean(1) >= lp[:, 120 - 60:120].mean(1)).astype(int)
feat_raw = [c for c in m.columns if c.startswith("m2_-45_") and not c.endswith(("_p", "_pm", "_pd", "_pup_now", "_pc_now", "_H", "p0_model", "price_now"))]
feat_models = ["m2_-45_close_p", "bm_-45_close_p", "bp_-60_z_s5", "bp_-60_z_s1", "m2_-45_h60_pup_now", "m2_-45_h30_pup_now", "bm_-45_h30_pup_now"]
sets = {"raw features only": feat_raw, "model outputs only": feat_models, "raw + model outputs": feat_raw + feat_models}
months = ["2026-08", "2026-09", "2026-10", "2026-11"]
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=20.0, verbose=-1, num_threads=4)
out = []
for nm, cols in sets.items():
    X = m[cols].astype(float).values; y = m.y_twap.values; pred = np.full(len(m), np.nan)
    for mo in months:
        te = m.month.values == mo; tr = (m.month.values < mo) & np.isfinite(X).all(1)
        if tr.sum() < 3000 or te.sum() == 0: continue
        b = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=300)
        pred[te] = b.predict(X[te])
    for mo in months:
        s = (m.month.values == mo) & np.isfinite(pred) & np.isfinite(m["m2_-45_close_p"].values)
        if s.sum() < 500: continue
        out.append(dict(model=nm, month=mo, n=int(s.sum()), auc_pm=roc_auc_score(m.up.values[s], pred[s]),
                        auc_m2_baseline=roc_auc_score(m.up.values[s], m["m2_-45_close_p"].values[s]),
                        acc_top30=np.mean((pred[s] > 0.5) == (m.up.values[s] == 1))))
    m[f"direct[{nm}]"] = pred
print(pd.DataFrame(out).round(4).to_string(index=False))
m[[c for c in m.columns if c.startswith("direct[")] + ["y_twap"]].to_parquet("/home/user/data/direct_twap_preds.parquet")
