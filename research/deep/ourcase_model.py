"""Deep dive 2: a model built for OUR target.  Inputs: the btcmovement2 / eth3 feature engine at T-45 (scripts/dump_feats.py, long history),
optionally the book-depth imbalances.  Target: Binance TWAP60(end) vs TWAP60(start) (binary), or the standardized TWAP move (regression).
Walk-forward by month: the model for month M is trained on every window that ended at least one day before M (all grid phases when
PHASES has several; evaluation on phase 0 = the real Polymarket grid).  OOS months 2025-10 .. 2026-11.
  selection set   2025-10 .. 2026-05 : Binance label only, never used by anything in this project
  confirmation    2026-06 .. 2026-11 : Polymarket outcome (AUC next to the frozen close head on the same windows)
  ASSET=BTC PHASES=0 TAG=ph0 python research/deep/ourcase_model.py"""
import os, sys, glob, time, warnings, numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, auc, DEEP  # noqa: E402
ASSET = os.environ.get("ASSET", "BTC"); PHASES = [int(x) for x in os.environ.get("PHASES", "0").split(",")]; TAG = os.environ.get("TAG", "ph0")
BOOK = os.environ.get("BOOK", "0") == "1"; TARGET = os.environ.get("TARGET", "twap"); FIRST = os.environ.get("FIRST", "2025-10"); THREADS = int(os.environ.get("THREADS", "3"))
LEARNERS = os.environ.get("LEARNERS", "logit,lgb_bin,lgb_reg").split(",")
D = pd.concat([pd.read_parquet(f) for ph in PHASES for f in glob.glob(f"{DEEP}/feats_{ASSET}_ph{ph}_*.parquet")], ignore_index=True).sort_values("T").reset_index(drop=True)
LAB = ["T", "phase", "y_twap", "x_twap", "y_pt", "x_pt", "x_dec", "price"]
FEATS = [c for c in D.columns if c not in LAB and D[c].std() > 0 and c not in ("e", "x_bps", "x_v", "wmax_v", "wmin_v", "ncross", "fup", "tsc")]
if BOOK:
    bd_dir = "/home/user/data/extra/bookdepth" if ASSET == "BTC" else "/home/user/data_eth/extra/bookdepth"
    bd = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{bd_dir}/*.parquet"))]).sort_index(); bd = bd[~bd.index.duplicated()]
    for lv in ("0.2", "1.0", "2.0", "5.0"): bd[f"bk_imb{lv}"] = (bd[f"n-{lv}"] - bd[f"n+{lv}"]) / (bd[f"n-{lv}"] + bd[f"n+{lv}"])
    bd["bk_outer"] = ((bd["n-1.0"] - bd["n-0.2"]) - (bd["n+1.0"] - bd["n+0.2"])) / ((bd["n-1.0"] - bd["n-0.2"]) + (bd["n+1.0"] - bd["n+0.2"]))
    bd["bk_ldepth"] = np.log(bd["n-1.0"] + bd["n+1.0"]); bd["bk_ldepth_rel"] = bd.bk_ldepth - bd.bk_ldepth.rolling(2880, min_periods=200).median()
    BK = ["bk_imb0.2", "bk_imb1.0", "bk_imb2.0", "bk_imb5.0", "bk_outer", "bk_ldepth_rel"]; idx = bd.index.values.astype(np.int64)
    def asof(when):
        j = np.searchsorted(idx, when, side="right") - 1; ok = j >= 0; v = np.where(ok[:, None], bd[BK].values[np.maximum(j, 0)], np.nan); v[np.where(ok, when - idx[np.maximum(j, 0)], 1e9) > 120] = np.nan; return v
    now = asof(D["T"].values - 47); ago = asof(D["T"].values - 347)
    for k, c in enumerate(BK): D[c] = now[:, k].astype(np.float32)
    D["bk_dimb1.0_300"] = (now[:, 1] - ago[:, 1]).astype(np.float32); FEATS += BK + ["bk_dimb1.0_300"]
    print("book coverage by year-month (share of windows with a snapshot):", D.groupby(pd.to_datetime(D["T"], unit="s").dt.strftime("%Y-%m"))["bk_imb1.0"].apply(lambda s: s.notna().mean()).round(2).to_dict())
if os.environ.get("XF", "0") == "1":      # cross-asset inputs (scripts/dump_cross.py): the other asset's returns / flow / basis / variance and relative returns
    xf = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DEEP}/xfeat_*.parquet"))], ignore_index=True).drop_duplicates("T")
    oth, own = ("eth", "btc") if ASSET == "BTC" else ("btc", "eth"); xf = xf.set_index("T").reindex(D["T"].values)
    XC = [c for c in xf.columns if c.startswith(oth + "_")]
    for c in XC: D["x_" + c] = xf[c].values
    for k in (10, 30, 60, 300, 900): D[f"x_rel_{k}"] = (xf[f"{oth}_r_{k}"].values - xf[f"{own}_r_{k}"].values).astype(np.float32)
    FEATS += ["x_" + c for c in XC] + [f"x_rel_{k}" for k in (10, 30, 60, 300, 900)]; print("cross-asset coverage", float(np.isfinite(D["x_rel_60"]).mean()))
ts = pd.to_datetime(D["T"], unit="s"); D["ym"] = ts.dt.strftime("%Y-%m"); y = D["y_twap" if TARGET == "twap" else "y_pt"].values.astype(int); x = D["x_twap" if TARGET == "twap" else "x_pt"].values
print(f"{ASSET} {TAG}: rows {len(D)}, features {len(FEATS)}, phases {PHASES}, target {TARGET}, {D.ym.min()}..{D.ym.max()}", flush=True)
MONTHS = [m for m in sorted(D.ym.unique()) if m >= FIRST]
X = D[FEATS].values.astype(np.float32); Xl = np.nan_to_num(X, nan=0.0)
LGB = dict(learning_rate=0.02, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0, max_bin=63, verbose=-1, num_threads=THREADS)
ROUNDS = tuple(int(r) for r in os.environ.get("ROUNDS", "150,300,500").split(",")); P = {}; t0 = time.time(); IMP = {}
for mo in MONTHS:
    m0 = int(pd.Timestamp(mo + "-01").timestamp()); tr = (D["T"].values + 300 < m0 - 86400); te = (D.ym.values == mo) & (D.phase.values == 0)
    if tr.sum() < 20000 or te.sum() == 0: continue
    if "logit" in LEARNERS:
        sc = StandardScaler().fit(Xl[tr]); lr = LogisticRegression(C=0.002, max_iter=300).fit(np.clip(sc.transform(Xl[tr]), -5, 5), y[tr])
        P.setdefault("logit", np.full(len(D), np.nan))[te] = lr.predict_proba(np.clip(sc.transform(Xl[te]), -5, 5))[:, 1]
    if "lgb_bin" in LEARNERS:
        b = lgb.train(dict(LGB, objective="binary"), lgb.Dataset(X[tr], y[tr]), num_boost_round=max(ROUNDS))
        for r in ROUNDS: P.setdefault(f"lgb_bin{r}", np.full(len(D), np.nan))[te] = b.predict(X[te], num_iteration=r)
        IMP[mo] = pd.Series(b.feature_importance("gain"), index=FEATS)
    if "lgb_reg" in LEARNERS:
        ax = np.log(np.abs(x[tr]) + 0.3); A = np.column_stack([np.ones(tr.sum()), D["lrv_300"].values[tr], D["lrv_3600"].values[tr]]); coef = np.linalg.lstsq(A, ax, rcond=None)[0]
        scale = np.exp(np.column_stack([np.ones(len(D)), D["lrv_300"].values, D["lrv_3600"].values]) @ coef); z = np.clip(x / scale, -4, 4)
        b = lgb.train(dict(LGB, objective="huber", alpha=1.5), lgb.Dataset(X[tr], z[tr]), num_boost_round=max(ROUNDS))
        for r in ROUNDS: P.setdefault(f"lgb_reg{r}", np.full(len(D), np.nan))[te] = b.predict(X[te], num_iteration=r)
    print(f"  {mo}: train {int(tr.sum())}, test {int(te.sum())}, {time.time() - t0:.0f}s", flush=True)
if "lgb_bin300" in P and "lgb_reg300" in P:
    rk = lambda v: pd.Series(v).rank(pct=True).values
    P["avg(bin300,reg300)"] = np.where(np.isfinite(P["lgb_bin300"]), (rk(np.where(np.isfinite(P["lgb_bin300"]), P["lgb_bin300"], np.nan)) + rk(np.where(np.isfinite(P["lgb_reg300"]), P["lgb_reg300"], np.nan))) / 2, np.nan)
out = pd.DataFrame({"T": D["T"].values, "ym": D.ym.values, "y_twap": D.y_twap.values, "y_pt": D.y_pt.values, "x_twap": D.x_twap.values, **P})[(D.phase.values == 0) & np.isin(D.ym.values, MONTHS)]
out.to_parquet(f"{DEEP}/ourcase_{ASSET}_{TAG}.parquet", index=False)
M = load(ASSET, ["m2_-45_close_p", "bp_-60_z_s5"]).reindex(out["T"].values); up = M.up.values; m2 = M["m2_-45_close_p"].values
rows = []
for nm in P:
    r = {"model": nm}
    for mo in MONTHS: s = out.ym.values == mo; r[mo[2:]] = auc(out[("y_twap" if TARGET == "twap" else "y_pt")].values[s], out[nm].values[s])
    sel = out.ym.values <= "2026-05"; r["SELECT 25-10..26-05"] = auc(y[(D.phase.values == 0) & np.isin(D.ym.values, MONTHS)][sel], out[nm].values[sel]); rows.append(r)
print("\n== AUC vs the Binance label, walk-forward months =="); print(pd.DataFrame(rows).set_index("model").round(3).to_string())
rows = []; pm = np.isfinite(up)
for nm, v in list(P.items()) + [("FROZEN m2 close head", None)]:
    r = {"model": nm}; vv = out[nm].values if v is not None else m2
    for mo in [m for m in MONTHS if m >= "2026-06"]: s = (out.ym.values == mo) & pm; r[mo[2:]] = auc(up[s], vv[s])
    s = pm & (out.ym.values >= "2026-06"); r["Jun-Nov"] = auc(up[s], vv[s]); rows.append(r)
print("\n== AUC vs the POLYMARKET outcome, Jun-Nov 2026 (confirmation) =="); print(pd.DataFrame(rows).set_index("model").round(3).to_string())
if IMP:
    imp = pd.DataFrame(IMP).mean(1).sort_values(ascending=False); print("\ntop 25 features by gain (mean over monthly models):", (imp / imp.sum()).head(25).round(3).to_dict())
