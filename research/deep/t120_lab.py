"""Experiment harness for the T-120 model (BTC).  Same walk-forward as research/deep/ourcase_model.py (a model per month, trained on everything
that ended at least a day before the month, 3 grid phases; tested on the real 5-minute grid), one column of out-of-sample scores per variant.
  python research/deep/t120_lab.py v1,v2,...      ->  /home/user/data/deep/t120_lab_<variant>.parquet"""
import os, sys, glob, time, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
DEEP = "/home/user/data/deep"; DEC = -120; FIRST = "2025-10"; THREADS = int(os.environ.get("THREADS", "4")); PHASES = [int(x) for x in os.environ.get("PHASES", "0,120,240").split(",")]
D = pd.concat([pd.read_parquet(f) for ph in PHASES for f in glob.glob(f"{DEEP}/feats_BTC_d{DEC}_ph{ph}_*.parquet")], ignore_index=True).sort_values("T").reset_index(drop=True)
LAB = ["T", "phase", "y_twap", "x_twap", "y_pt", "x_pt", "x_dec", "price"]; BASE = [c for c in D.columns if c not in LAB and D[c].std() > 0 and c not in ("e", "x_bps", "x_v", "wmax_v", "wmin_v", "ncross", "fup", "tsc")]
bd = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("/home/user/data/extra/bookdepth/*.parquet"))]).sort_index(); bd = bd[~bd.index.duplicated()]
for lv in ("0.2", "1.0", "2.0", "5.0"): bd[f"bk_imb{lv}"] = (bd[f"n-{lv}"] - bd[f"n+{lv}"]) / (bd[f"n-{lv}"] + bd[f"n+{lv}"])
bd["bk_outer"] = ((bd["n-1.0"] - bd["n-0.2"]) - (bd["n+1.0"] - bd["n+0.2"])) / ((bd["n-1.0"] - bd["n-0.2"]) + (bd["n+1.0"] - bd["n+0.2"]))
bd["bk_ldepth"] = np.log(bd["n-1.0"] + bd["n+1.0"]); bd["bk_ldepth_rel"] = bd.bk_ldepth - bd.bk_ldepth.rolling(2880, min_periods=200).median()
bd["bk_ldepth02"] = np.log(bd["n-0.2"] + bd["n+0.2"]); bd["bk_ldepth02_rel"] = bd.bk_ldepth02 - bd.bk_ldepth02.rolling(2880, min_periods=200).median()
BK = ["bk_imb0.2", "bk_imb1.0", "bk_imb2.0", "bk_imb5.0", "bk_outer", "bk_ldepth_rel", "bk_ldepth02_rel"]; idx = bd.index.values.astype(np.int64)
def asof(when):
    j = np.searchsorted(idx, when, side="right") - 1; ok = j >= 0; v = np.where(ok[:, None], bd[BK].values[np.maximum(j, 0)], np.nan); v[np.where(ok, when - idx[np.maximum(j, 0)], 1e9) > 120] = np.nan; return v
now = asof(D["T"].values + DEC - 2); ago = asof(D["T"].values + DEC - 302)
for k, c in enumerate(BK): D[c] = now[:, k].astype(np.float32)
D["bk_dimb1.0_300"] = (now[:, 1] - ago[:, 1]).astype(np.float32); D["bk_dimb0.2_300"] = (now[:, 0] - ago[:, 0]).astype(np.float32)
# Binance futures metrics (5-minute open interest and long/short ratios).  The row stamped ts describes the interval [ts, ts+300) (checked: the taker ratio
# at ts matches our own buy/sell volume over [ts, ts+300) with correlation 0.998), so a row is only usable from ts + 300; 30 s more for the live fetch.
mt = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("/home/user/data/extra/metrics/*.parquet"))]).sort_index(); mt = mt[~mt.index.duplicated()]
mt = mt.reindex(np.arange(mt.index.min(), mt.index.max() + 300, 300)); M = pd.DataFrame(index=mt.index); oi = np.log(mt.sum_open_interest)
for k, nm in ((1, "5"), (3, "15"), (12, "60"), (48, "240")): M[f"oi_d{nm}"] = oi - oi.shift(k)
M["oi_rel1d"] = oi - oi.rolling(288, min_periods=100).median()
for col, nm in (("sum_toptrader_long_short_ratio", "tlp"), ("count_toptrader_long_short_ratio", "tla"), ("count_long_short_ratio", "ls")):
    v = np.log(mt[col]); M[f"{nm}_rel1d"] = v - v.rolling(288, min_periods=100).median(); M[f"{nm}_d15"] = v - v.shift(3); M[f"{nm}_d60"] = v - v.shift(12)
MET = list(M.columns); mi = M.index.values; jm = np.searchsorted(mi, D["T"].values + DEC - 330, side="right") - 1
for c in MET: D[c] = np.where(jm >= 0, M[c].values[np.maximum(jm, 0)], np.nan).astype(np.float32)
ts = pd.to_datetime(D["T"], unit="s"); D["ym"] = ts.dt.strftime("%Y-%m"); y = D.y_twap.values.astype(int); x = D.x_twap.values; Tt = D["T"].values; MONTHS = [m for m in sorted(D.ym.unique()) if m >= FIRST]
LGB = dict(learning_rate=0.02, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0, max_bin=63, verbose=-1, num_threads=THREADS)
BOOK_ALL = ["bk_imb0.2", "bk_imb1.0", "bk_imb2.0", "bk_imb5.0", "bk_outer", "bk_ldepth_rel", "bk_dimb1.0_300"]
V = {   # name: features, objective, extra lgb params, rounds, sample-weight half-life in days (None = equal), seeds
 "base":      dict(f=BASE, obj="reg", p={}, r=(150, 300, 500), hl=None, seeds=(0,)),
 "book_all":  dict(f=BASE + BOOK_ALL, obj="reg", p={}, r=(300,), hl=None, seeds=(0,)),
 "book_02":   dict(f=BASE + ["bk_imb0.2", "bk_ldepth02_rel", "bk_dimb0.2_300"], obj="reg", p={}, r=(300,), hl=None, seeds=(0,)),
 "recent180": dict(f=BASE, obj="reg", p={}, r=(300,), hl=180, seeds=(0,)),
 "recent365": dict(f=BASE, obj="reg", p={}, r=(300,), hl=365, seeds=(0,)),
 "bin":       dict(f=BASE, obj="bin", p={}, r=(150, 300, 500), hl=None, seeds=(0,)),
 "big":       dict(f=BASE, obj="reg", p=dict(num_leaves=31, min_data_in_leaf=800), r=(300, 500), hl=None, seeds=(0,)),
 "slow":      dict(f=BASE, obj="reg", p=dict(learning_rate=0.01), r=(600, 1000), hl=None, seeds=(0,)),
 "seeds":     dict(f=BASE, obj="reg", p={}, r=(300,), hl=None, seeds=(1, 2, 3)),
 "metrics":   dict(f=BASE + MET, obj="reg", p={}, r=(300,), hl=None, seeds=(0,)),
 "metrics_oi": dict(f=BASE + [c for c in MET if c.startswith("oi_")], obj="reg", p={}, r=(300,), hl=None, seeds=(0,)),
 "metrics_ls": dict(f=BASE + [c for c in MET if not c.startswith("oi_")], obj="reg", p={}, r=(300,), hl=None, seeds=(0,)),
 "win90":     dict(f=BASE, obj="reg", p={}, r=(300,), hl=None, seeds=(0,), win=90),     # learning curve: train only on the last N days
 "win180":    dict(f=BASE, obj="reg", p={}, r=(300,), hl=None, seeds=(0,), win=180),
 "win270":    dict(f=BASE, obj="reg", p={}, r=(300,), hl=None, seeds=(0,), win=270),
 "win365":    dict(f=BASE, obj="reg", p={}, r=(300,), hl=None, seeds=(0,), win=365),
 "win450":    dict(f=BASE, obj="reg", p={}, r=(300,), hl=None, seeds=(0,), win=450),
 "win540":    dict(f=BASE, obj="reg", p={}, r=(300,), hl=None, seeds=(0,), win=540),
 "hist2024":  dict(f=BASE, obj="reg", p={}, r=(300, 500), hl=None, seeds=(0,), since="2024-01-01"),   # older training history (scripts/fetch_hist_feats.py)
 "hist2023":  dict(f=BASE, obj="reg", p={}, r=(300, 500), hl=None, seeds=(0,), since="2023-01-01"),
 "hist2022":  dict(f=BASE, obj="reg", p={}, r=(300, 500), hl=None, seeds=(0,), since="2022-01-01"),
 "small":     dict(f=BASE, obj="reg", p=dict(num_leaves=7, min_data_in_leaf=1000), r=(300, 600), hl=None, seeds=(0,)),
}
for name in sys.argv[1].split(","):
    c = V[name]; X = D[c["f"]].values.astype(np.float32); P = {}; t0 = time.time()
    for mo in MONTHS:
        m0 = int(pd.Timestamp(mo + "-01").timestamp()); tr = (Tt + 300 < m0 - 86400) & (Tt >= m0 - 86400 * c.get("win", 10**6)) & (Tt >= pd.Timestamp(c.get("since", "2025-01-12")).timestamp()); te = (D.ym.values == mo) & (D.phase.values == 0)
        if tr.sum() < 20000 or te.sum() == 0: continue
        w = None if c["hl"] is None else np.exp(-(m0 - Tt[tr]) / 86400 / c["hl"] * np.log(2)).astype(np.float32)
        if c["obj"] == "reg":
            A = np.column_stack([np.ones(tr.sum()), D["lrv_300"].values[tr], D["lrv_3600"].values[tr]]); coef = np.linalg.lstsq(A, np.log(np.abs(x[tr]) + 0.3), rcond=None)[0]
            z = np.clip(x / np.exp(np.column_stack([np.ones(len(D)), D["lrv_300"].values, D["lrv_3600"].values]) @ coef), -4, 4); tgt = z[tr]; par = dict(LGB, objective="huber", alpha=1.5, **c["p"])
        else: tgt = y[tr]; par = dict(LGB, objective="binary", **c["p"])
        for sd in c["seeds"]:
            b = lgb.train(dict(par, seed=sd, bagging_seed=sd, feature_fraction_seed=sd), lgb.Dataset(X[tr], tgt, weight=w), num_boost_round=max(c["r"]))
            for r in c["r"]: P.setdefault(f"{name}_r{r}" + (f"_s{sd}" if len(c["seeds"]) > 1 else ""), np.full(len(D), np.nan))[te] = b.predict(X[te], num_iteration=r) - (0.5 if c["obj"] == "bin" else 0.0)
        print(f"  {name} {mo}: train {int(tr.sum())}, {time.time() - t0:.0f}s", flush=True)
    keep = (D.phase.values == 0) & np.isin(D.ym.values, MONTHS)
    pd.DataFrame({"T": Tt, "ym": D.ym.values, "y_twap": D.y_twap.values, "x_twap": x, **P})[keep].to_parquet(f"{DEEP}/t120_lab_{name}.parquet", index=False); print(f"saved {name} ({time.time() - t0:.0f}s)", flush=True)
