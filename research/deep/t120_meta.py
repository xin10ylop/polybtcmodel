"""Second-stage 'is this signal reliable?' model for the T-120 rule.  The direction comes from the current model (walk-forward scores, base_r300); a small
model then predicts whether that direction will be right, from things that do not depend on direction (hour, volatility, activity) and from how the
signal lines up with recent moves.  Walk-forward by month on out-of-sample scores only: month m is ranked by a model fit on months before m.
Compared at the SAME number of trades as the live selection (top 30 % by |score|, weekends + weekday 16-24 UTC).
SELECT = Jan - May 2026 (first three score months only train), CONFIRM = Jun - Nov 2026.     python research/deep/t120_meta.py"""
import glob, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
DEEP = "/home/user/data/deep"; S = pd.read_parquet(f"{DEEP}/t120_lab_base.parquet")[["T", "ym", "x_twap", "base_r300"]].rename(columns={"base_r300": "s"})
F = pd.read_parquet(glob.glob(f"{DEEP}/feats_BTC_d-120_ph0_*.parquet")[0]); D = S.merge(F, on="T", how="inner", suffixes=("", "_f")); D = D[(D.x_twap != 0) & np.isfinite(D.s)].reset_index(drop=True)
t = pd.to_datetime(D["T"], unit="s"); d = np.sign(D.s.values); D["ok"] = (d == np.sign(D.x_twap.values)).astype(int); D["a"] = np.abs(D.s.values)
D["hr"] = t.dt.hour.values; D["wd"] = t.dt.dayofweek.values; D["wkend"] = (D.wd >= 5).astype(int); D["good"] = ((D.wd >= 5) | (D.hr >= 16)).astype(int)
D["hs"] = np.sin(2 * np.pi * D.hr / 24); D["hc"] = np.cos(2 * np.pi * D.hr / 24)
AG = ["r_60", "r_300", "r_900", "r_3600", "mtf_15m", "mtf_1h", "mtf_4h", "mtf_1d", "pimb_300", "simb_300", "pimb_900", "simb_900", "reg_trend_4h", "reg_trend_1d", "prev_ret_v", "bdev_300"]
for c in AG: D["ag_" + c] = d * D[c].values
NEU = ["a", "hr", "hs", "hc", "wd", "wkend", "good", "lrv_300", "lrv_3600", "lrv_86400", "lrvsa_300", "lrvsa_3600", "lcnt_300", "lvol_300", "lcnt_ratio", "seas_now", "jump_300", "jump_3600", "reg_vr300_1d", "reg_rv_1d_7d", "reg_lvolume_1d_7d"]
X1 = NEU; X2 = NEU + ["ag_" + c for c in AG]; months = sorted(D.ym.unique()); P = {k: np.full(len(D), np.nan) for k in ("logit_neutral", "lgb_neutral", "lgb_all")}
for m in months[3:]:
    tr = (D.ym.values < m) & (D["T"].values + 300 < pd.Timestamp(m + "-01").timestamp() - 86400); te = D.ym.values == m
    sc = StandardScaler().fit(D.loc[tr, X1]); P["logit_neutral"][te] = LogisticRegression(C=0.05, max_iter=500).fit(sc.transform(D.loc[tr, X1]), D.ok.values[tr]).predict_proba(sc.transform(D.loc[te, X1]))[:, 1]
    par = dict(objective="binary", learning_rate=0.02, num_leaves=7, min_data_in_leaf=500, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0, max_bin=31, verbose=-1, num_threads=2, seed=0)
    for k, cols in (("lgb_neutral", X1), ("lgb_all", X2)): P[k][te] = lgb.train(par, lgb.Dataset(D.loc[tr, cols].values, D.ok.values[tr]), 150).predict(D.loc[te, cols].values)
thr = D.groupby("ym").a.transform(lambda v: v.quantile(0.70)).values; live = (D.a.values >= thr) & (D.good.values == 1); top30 = D.a.values >= thr
def pick(p, n_by_month, mask=None):   # the n strongest by p in each month (same trade count as the reference selection)
    k = np.zeros(len(D), bool)
    for m, n in n_by_month.items():
        i = np.where((D.ym.values == m) & np.isfinite(p) & (True if mask is None else mask))[0]; k[i[np.argsort(-p[i])[:n]]] = True
    return k
rows = []
for nm, per in (("SELECT Jan-May 26", (D.ym.values >= months[3]) & (D.ym.values <= "2026-05")), ("CONFIRM Jun-Nov 26", D.ym.values >= "2026-06")):
    n_live = D[per & live].groupby("ym").size().to_dict(); n_30 = D[per & top30].groupby("ym").size().to_dict()
    def row(lbl, k): k = k & per; rows.append({"period": nm, "selection": lbl, "trades": int(k.sum()), "hit %": round(100 * D.ok.values[k].mean(), 2), "edge x trades": round((D.ok.values[k].mean() - 0.5) * 2 * k.sum())})
    row("LIVE: top 30 % by |score|, trading hours", live); row("top 30 % by |score|, all hours", top30)
    for k_, p in P.items():
        row(f"{k_}: same count as LIVE, any hour", pick(p, n_live)); row(f"{k_}: same count as LIVE, among top 30 % only", pick(p, n_live, top30)); row(f"{k_}: same count as LIVE, trading hours only", pick(p, n_live, D.good.values == 1))
        row(f"{k_}: count of top 30 %, any hour", pick(p, n_30))
pd.set_option("display.width", 220); print(pd.DataFrame(rows).to_string(index=False))
print("\none standard error of a hit rate at these counts: about", round(100 * 0.5 / np.sqrt(D[(D.ym.values >= '2026-06') & live].shape[0]), 2), "points")
