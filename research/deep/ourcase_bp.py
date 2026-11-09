"""Deep dive 2b: the btc-predict / eth1 feature panel (minute features, information < minute boundary m = T-60) retargeted to OUR label:
TWAP60 over [m+300, m+360) >= TWAP60 over [m, m+60)  (i.e. the window that opens at T = m+60).  Every minute boundary is a training row;
walk-forward by month from 2026-06 (panel starts 2026-02-03), evaluated on the 5-minute grid against the Polymarket outcome, next to the
existing 5-minute z forecast (target: point return m -> m+300).
  ASSET=BTC python research/deep/ourcase_bp.py"""
import os, sys, json, time, warnings, datetime as dt, numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
ASSET = os.environ.get("ASSET", "BTC"); THREADS = int(os.environ.get("THREADS", "3"))
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, auc, DEEP  # noqa: E402
if ASSET == "BTC":
    sys.path.insert(0, "/home/user/btc-predict"); os.chdir("/home/user/btc-predict")
    from btcpredict.features import matrix as F; from btcpredict.config import FEAT, FEATURE_SETS
    df = pd.read_parquet(FEAT / "panel_exp.parquet"); SEC = "/home/user/btc-predict/data/sec/spot"; cols = open(FEATURE_SETS / "core_fast.txt").read().split()
else:
    sys.path.insert(0, "/home/user/xin10ylop/eth1"); os.chdir("/home/user/xin10ylop/eth1")
    from ethpredict.features import matrix as F
    df = pd.read_parquet("/home/user/xin10ylop/eth1/data/features/panel_ep.parquet"); SEC = "/home/user/btc-predict/data/sec/eth"
    cols = json.load(open("/home/user/xin10ylop/eth1/models/meta.json"))["spot"]["targets"]["s5"]["features"]["core"]
df = df[df["ts"] >= "2026-02-03"].reset_index(drop=True); X = pd.concat([F.model_matrix(df), F.extra_features(df)], axis=1); cols = [c for c in cols if c in X.columns]
print(ASSET, "panel", df.shape, "features", len(cols), flush=True)
# labels from the 1-s spot closes
d0 = pd.Timestamp(df.ts.min()).normalize(); d1 = pd.Timestamp(df.ts.max()).normalize(); days = pd.date_range(d0, d1); c = np.full(len(days) * 86400, np.nan)
for k, d in enumerate(days):
    f = f"{SEC}/{d.date()}.parquet"
    if os.path.exists(f): c[k * 86400:(k + 1) * 86400] = pd.read_parquet(f, columns=["c"])["c"].values
m_ = ~np.isfinite(c); idx = np.where(~m_, np.arange(len(c)), 0); np.maximum.accumulate(idx, out=idx); c = c[idx]; cs = np.concatenate([[0.0], np.cumsum(c)])
t0 = int(d0.timestamp()); Tm = ((pd.DatetimeIndex(df.ts) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).values.astype(np.int64) + 60      # forecast instant m
i = Tm - t0; ok = (i >= 0) & (i + 360 <= len(c)); ii = np.where(ok, i, 0); tw0 = (cs[ii + 60] - cs[ii]) / 60; tw1 = (cs[ii + 360] - cs[ii + 300]) / 60
x = np.where(ok, 1e4 * np.log(tw1 / tw0), np.nan); y = (x >= 0).astype(int); z = np.clip(x / (1e4 * df["sigma"].values * np.sqrt(5) + 1e-9), -4, 4)
LGB = dict(learning_rate=0.02, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0, max_bin=63, verbose=-1, num_threads=THREADS)
Xv = X[cols].values.astype(np.float32); ym = pd.DatetimeIndex(df.ts).strftime("%Y-%m").values; P = {k: np.full(len(df), np.nan) for k in ("bp_twap_bin", "bp_twap_reg")}; tt = time.time()
for mo in ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11"]:
    m0 = int(pd.Timestamp(mo + "-01").timestamp()); tr = ok & np.isfinite(x) & (Tm + 360 < m0 - 86400); te = ok & (ym == mo)
    b = lgb.train(dict(LGB, objective="binary"), lgb.Dataset(Xv[tr], y[tr]), num_boost_round=300); P["bp_twap_bin"][te] = b.predict(Xv[te])
    b = lgb.train(dict(LGB, objective="huber", alpha=1.5), lgb.Dataset(Xv[tr], z[tr]), num_boost_round=300); P["bp_twap_reg"][te] = b.predict(Xv[te])
    print(f"  {mo}: train {int(tr.sum())}, test {int(te.sum())}, {time.time() - tt:.0f}s", flush=True)
out = pd.DataFrame({"T": Tm + 60, **P}); out = out[(out["T"] % 300 == 0) & np.isfinite(out.bp_twap_bin)]; out.to_parquet(f"{DEEP}/ourcase_bp_{ASSET}.parquet", index=False)
M = load(ASSET, ["m2_-45_close_p", "bp_-60_z_s5"]); o = out.set_index("T").reindex(M.index); rows = []
for nm, v in (("panel model, TWAP binary", o.bp_twap_bin.values), ("panel model, TWAP regression", o.bp_twap_reg.values), ("existing z_s5 @T-60", M["bp_-60_z_s5"].values), ("FROZEN m2 close head", M["m2_-45_close_p"].values)):
    r = {"model": nm}
    for mo in ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]: s = M.month.values == mo; r[mo] = auc(M.up.values[s], v[s])
    r["Jun-Nov"] = auc(M.up.values, v); rows.append(r)
print("\n== AUC vs the POLYMARKET outcome =="); print(pd.DataFrame(rows).set_index("model").round(3).to_string())
