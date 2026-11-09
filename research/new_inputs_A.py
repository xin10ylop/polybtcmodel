"""Can NEW INFORMATION lift A's directional accuracy?  Inputs none of the three models see, all public archives
(scripts/fetch_extra_feeds.py): Bybit and OKX perp trades (cross-venue flow and lead-lag), Binance futures book depth
(bid/ask notional within 0.2-5 %), Binance futures metrics (open interest, long/short ratios, taker ratio).
Features are built from bars complete at T-47 (decision T-45 minus live latency).  Walk-forward by month:
Aug..Nov predicted from all earlier months, June/July predicted backwards from Aug..Nov.  Learners: logistic and a
small LightGBM on [frozen heads] vs [frozen heads + new features].  Then the A trade rule is re-run with the new
probability as the signal (polysim, same exits as live) and compared with the frozen rule per period."""
import os, sys, glob, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, "/home/user/polybtcmodel/polybtc"); import polysim  # noqa: E402
EXTRA = "/home/user/data/extra"; TL = "/home/user/data/timeline"; TL0 = int(pd.Timestamp("2025-01-01").timestamp())
D0 = pd.Timestamp("2026-06-01"); D1 = pd.Timestamp("2026-11-08"); BASE = int(D0.timestamp()); NSEC = int((D1 - D0).total_seconds())
LAT = 47  # last complete 1-s bar used: T-47
OUT = "/home/user/polybtcmodel/reports"; os.makedirs(OUT, exist_ok=True)

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

# ------------------------------------------------------------------ windows
HEADS = ["m2_-45_close_p", "bp_-60_z_s5", "bp_-60_z_s1"]   # btc-movement is not scored for Nov 4-7 and the live rule does not use it
RAW = ['m2_-45_x_bps', 'm2_-45_lrv_300', 'm2_-45_lrvsa_300', 'm2_-45_lrv_3600', 'm2_-45_seas_now', 'm2_-45_jump_300', 'm2_-45_pls_5', 'm2_-45_pls_10',
       'm2_-45_bdev_60', 'm2_-45_bdev_300', 'm2_-45_r_10', 'm2_-45_r_30', 'm2_-45_pimb_10', 'm2_-45_prev_ret_v', 'm2_-45_reg_ll_beta_1h', 'm2_-45_reg_catchup_1h',
       'm2_-45_reg_vr300_1d', 'm2_-45_reg_rv_1d_7d', 'm2_-45_reg_trend_1d', 'm2_-45_close_mu', 'm2_-45_close_sig', 'm2_-45_h60_mu', 'm2_-45_h10_mu']
SRC = [("Jun", "/home/user/data/master_june.parquet", "/home/user/data/polymarket/markets_2026-06-01_2026-07-01.parquet"),
       ("Jul", "/home/user/data/master_july_L0.parquet", "/home/user/data/polymarket/markets_2026-07-01_2026-08-01.parquet"),
       ("main", "/home/user/data/master.parquet", "/home/user/data/polymarket/markets.parquet"),
       ("Nov", "/home/user/data/master_nov.parquet", "/home/user/data/polymarket/markets_nov.parquet")]
parts = []
for nm, mp, _ in SRC:
    m = pd.read_parquet(mp, columns=["date", "up"] + HEADS + RAW); m["src"] = nm; parts.append(m)
W = pd.concat(parts); W = W[~W.index.duplicated(keep="last")].sort_index(); W["date"] = pd.to_datetime(W.date)
W["month"] = W.date.dt.strftime("%b"); W.loc[W.date >= "2026-11-01", "month"] = "Nov"
print("windows", len(W), W.month.value_counts().to_dict(), flush=True)
T = W.index.values.astype(np.int64); c = T - LAT - BASE      # index of the last usable second in the June-Nov arrays

# ------------------------------------------------------------------ 1-s venue arrays
def venue(name):
    close = np.full(NSEC, np.nan); buy = np.zeros(NSEC, np.float32); sell = np.zeros(NSEC, np.float32); have = np.zeros(NSEC, bool)
    for f in sorted(glob.glob(f"{EXTRA}/{name}/*.parquet")):
        d = pd.Timestamp(os.path.basename(f)[:10]); i0 = int((d - D0).total_seconds())
        if i0 < 0 or i0 + 86400 > NSEC: continue
        t = pd.read_parquet(f); close[i0:i0 + 86400] = t.close.values; buy[i0:i0 + 86400] = t.buy.values; sell[i0:i0 + 86400] = t.sell.values; have[i0:i0 + 86400] = True
    close = pd.Series(close).ffill().to_numpy().copy(); close[~have] = np.nan
    return close, np.cumsum(buy, dtype=np.float64), np.cumsum(sell, dtype=np.float64)

def binance_perp():
    pc = np.load(f"{TL}/p_close.npy", mmap_mode="r"); pb = np.load(f"{TL}/p_buy.npy", mmap_mode="r"); ps = np.load(f"{TL}/p_sell.npy", mmap_mode="r")
    o = BASE - TL0; close = np.array(pc[o:o + NSEC], dtype=np.float64); close = pd.Series(close).replace(0, np.nan).ffill().to_numpy().copy()
    return close, np.cumsum(np.array(pb[o:o + NSEC], dtype=np.float64)), np.cumsum(np.array(ps[o:o + NSEC], dtype=np.float64))

def wsum(cs, i, k): return cs[i] - cs[np.maximum(i - k, 0)]
def ret(close, i, k): return 1e4 * np.log(close[i] / close[np.maximum(i - k, 0)])

F = pd.DataFrame(index=W.index)
V = {"bn": binance_perp()}
for nm in ("bybit1s", "okx1s"):
    if glob.glob(f"{EXTRA}/{nm}/*.parquet"): V[nm[:2]] = venue(nm)
for v, (close, cb, cs) in V.items():
    for k in (10, 30, 60, 300): F[f"ret_{v}_{k}"] = ret(close, c, k)
    for k in (10, 60, 300):
        b, s = wsum(cb, c, k), wsum(cs, c, k); F[f"imb_{v}_{k}"] = (b - s) / (b + s + 1e-9)
    F[f"vol_{v}_300rel"] = np.log1p(wsum(cb, c, 300) + wsum(cs, c, 300)) - np.log1p((wsum(cb, c, 3600) + wsum(cs, c, 3600)) / 12)
    if v != "bn":
        gap = 1e4 * np.log(close / V["bn"][0]); F[f"gap_{v}"] = gap[c]; F[f"dgap_{v}_60"] = gap[c] - gap[np.maximum(c - 60, 0)]
        for k in (10, 30): F[f"lead_{v}_{k}"] = F[f"ret_{v}_{k}"] - F[f"ret_bn_{k}"]
        for k in (10, 60): F[f"dimb_{v}_{k}"] = F[f"imb_{v}_{k}"] - F[f"imb_bn_{k}"]
if len(V) == 3:
    for k in (30, 60): F[f"xv_ret_{k}"] = (F[f"ret_bn_{k}"] + F[f"ret_by_{k}"] + F[f"ret_ok_{k}"]) / 3; F[f"xv_agree_{k}"] = np.sign(F[f"ret_bn_{k}"]) + np.sign(F[f"ret_by_{k}"]) + np.sign(F[f"ret_ok_{k}"])
    F["xv_imb_60"] = (F.imb_bn_60 + F.imb_by_60 + F.imb_ok_60) / 3
XV = [col for col in F.columns if not col.startswith(("ret_bn", "imb_bn", "vol_bn"))]      # Binance-only columns are already in the models
BN = [col for col in F.columns if col.startswith(("ret_bn", "imb_bn", "vol_bn"))]

# ------------------------------------------------------------------ book depth (30-s snapshots) and metrics (5-min)
def asof(tab, when):
    idx = tab.index.values; j = np.searchsorted(idx, when, side="right") - 1; ok = j >= 0
    out = pd.DataFrame(np.where(ok[:, None], tab.values[np.maximum(j, 0)], np.nan), columns=tab.columns); out["age"] = np.where(ok, when - idx[np.maximum(j, 0)], np.nan); return out
BOOK, MET = [], []
bd_files = sorted(glob.glob(f"{EXTRA}/bookdepth/*.parquet"))
if bd_files:
    bd = pd.concat([pd.read_parquet(f) for f in bd_files]).sort_index(); bd = bd[~bd.index.duplicated()]
    for lv in ("0.2", "1.0", "2.0", "5.0"): bd[f"imb{lv}"] = (bd[f"n-{lv}"] - bd[f"n+{lv}"]) / (bd[f"n-{lv}"] + bd[f"n+{lv}"])
    bd["ldepth1"] = np.log(bd["n-1.0"] + bd["n+1.0"]); bd["ldepth1_rel"] = bd.ldepth1 - bd.ldepth1.rolling(2880, min_periods=200).median()
    bd["imb_outer"] = ((bd["n-1.0"] - bd["n-0.2"]) - (bd["n+1.0"] - bd["n+0.2"])) / ((bd["n-1.0"] - bd["n-0.2"]) + (bd["n+1.0"] - bd["n+0.2"]))
    now = asof(bd[["imb0.2", "imb1.0", "imb2.0", "imb5.0", "ldepth1_rel", "imb_outer"]], T - LAT); ago = asof(bd[["imb0.2", "imb1.0"]], T - LAT - 300)
    for col in ("imb0.2", "imb1.0", "imb2.0", "imb5.0", "ldepth1_rel", "imb_outer"): F[f"bk_{col}"] = now[col].values
    F["bk_dimb0.2_300"] = now["imb0.2"].values - ago["imb0.2"].values; F["bk_dimb1.0_300"] = now["imb1.0"].values - ago["imb1.0"].values; F["bk_age"] = now.age.values
    F.loc[F.bk_age > 120, [col for col in F.columns if col.startswith("bk_")]] = np.nan
    BOOK = [col for col in F.columns if col.startswith("bk_") and col != "bk_age"]
mt_files = sorted(glob.glob(f"{EXTRA}/metrics/*.parquet"))
if mt_files:
    mt = pd.concat([pd.read_parquet(f) for f in mt_files]).sort_index(); mt = mt[~mt.index.duplicated()]
    mt["loi"] = np.log(mt.sum_open_interest); mt["doi_5"] = mt.loi.diff(1); mt["doi_15"] = mt.loi.diff(3); mt["doi_60"] = mt.loi.diff(12)
    mt["ltaker"] = np.log(mt.sum_taker_long_short_vol_ratio); mt["ltop_cnt"] = np.log(mt.count_toptrader_long_short_ratio); mt["ltop_sum"] = np.log(mt.sum_toptrader_long_short_ratio)
    mt["lls_cnt"] = np.log(mt.count_long_short_ratio); mt["dtop_cnt_60"] = mt.ltop_cnt.diff(12); mt["dlls_60"] = mt.lls_cnt.diff(12); mt["dtaker_15"] = mt.ltaker.diff(3)
    cols = ["doi_5", "doi_15", "doi_60", "ltaker", "ltop_cnt", "ltop_sum", "lls_cnt", "dtop_cnt_60", "dlls_60", "dtaker_15"]
    now = asof(mt[cols], T - LAT)
    for col in cols: F[f"mt_{col}"] = now[col].values
    F.loc[now.age.values > 900, [f"mt_{c_}" for c_ in cols]] = np.nan
    MET = [f"mt_{c_}" for c_ in cols]
print("features: cross-venue", len(XV), "book", len(BOOK), "metrics", len(MET), " coverage:", {k: f"{np.isfinite(F[v[0]]).mean():.0%}" for k, v in (("xv", XV), ("book", BOOK), ("met", MET)) if v}, flush=True)
F.to_parquet(f"{OUT}/../exports/new_inputs_features.parquet")

# ------------------------------------------------------------------ models, walk-forward by month
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
base = pd.DataFrame({"l_m2": logit(W["m2_-45_close_p"].values), "z5": W["bp_-60_z_s5"].values, "z1": W["bp_-60_z_s1"].values}, index=W.index)
SETS = {"heads": list(base.columns), "heads+xv": list(base.columns) + XV, "heads+book": list(base.columns) + BOOK, "heads+metrics": list(base.columns) + MET,
        "heads+all": list(base.columns) + XV + BOOK + MET, "heads+all+raw": list(base.columns) + XV + BOOK + MET + RAW}
X_all = pd.concat([base, F, W[RAW]], axis=1); y = W.up.values.astype(int); mon = W.month.values
LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=20.0, verbose=-1, num_threads=4)
def fit_predict(cols, learner):
    pred = np.full(len(W), np.nan)
    for mo in MONTHS:
        te = mon == mo
        tr = np.isin(mon, [m_ for m_ in MONTHS if MONTHS.index(m_) < MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
        X = X_all[cols].values; okx = np.isfinite(X).all(1); tr &= okx; te_ok = te & okx
        if tr.sum() < 2000 or te_ok.sum() == 0: continue
        if learner == "logit":
            sc = StandardScaler().fit(X[tr]); lr = LogisticRegression(C=0.05, max_iter=500).fit(sc.transform(X[tr]), y[tr]); pred[te_ok] = lr.predict_proba(sc.transform(X[te_ok]))[:, 1]
        else:
            b = lgb.train(LGB, lgb.Dataset(X[tr], y[tr]), num_boost_round=300); pred[te_ok] = b.predict(X[te_ok])
    return pred
P = {"frozen m2 head": W["m2_-45_close_p"].values}
for sn, cols in SETS.items():
    for ln in ("logit", "lgb"):
        if sn == "heads" and ln == "lgb": continue
        P[f"{sn} [{ln}]"] = fit_predict(cols, ln); print("fitted", sn, ln, flush=True)
rows = []
for nm, p in P.items():
    for mo in MONTHS:
        s = (mon == mo) & np.isfinite(p) & np.isfinite(P["frozen m2 head"])
        if s.sum() < 300: continue
        rows.append(dict(model=nm, month=mo, n=int(s.sum()), auc=roc_auc_score(y[s], p[s]), auc_frozen=roc_auc_score(y[s], P["frozen m2 head"][s]),
                         acc=float(((p[s] > 0.5) == (y[s] == 1)).mean()), acc_conf=float(((p[s] > 0.5) == (y[s] == 1))[np.abs(p[s] - 0.5) >= 0.05].mean()) if (np.abs(p[s] - 0.5) >= 0.05).sum() > 50 else np.nan,
                         n_conf=int((np.abs(p[s] - 0.5) >= 0.05).sum())))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/A_new_inputs_auc.csv", index=False)
print("\n== AUC vs the Polymarket outcome, walk-forward by month (Jun/Jul predicted backwards from Aug-Nov) ==")
print(R.pivot(index="model", columns="month", values="auc")[MONTHS].round(3).to_string())
print("\n== directional accuracy on confident windows (|p-0.5| >= 0.05), and their count ==")
print(R.pivot(index="model", columns="month", values="acc_conf")[MONTHS].round(3).to_string()); print(R.pivot(index="model", columns="month", values="n_conf")[MONTHS].to_string())

# ------------------------------------------------------------------ the A trade rule with the new signal (polysim, live exits)
def period_sim(markets, days):
    polysim.MARKETS = markets; mk, g = polysim.load_windows(days=days)
    sim = polysim.simulate(mk, g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=(0.85,), sls=(0.0,), tstops=(300,), slip=0.01)
    idx = np.arange(len(sim["W"]))
    for sd in (0, 1):
        k = sim["kind"][idx, sd, 0, 0, 0]; sim["pnl"][idx, sd, 0, 0, 0] = np.where(k == 0, 0.35, sim["settle"][idx, sd])
    return pd.DataFrame({"T": sim["W"], "pnl_up": sim["pnl"][idx, 0, 0, 0, 0], "pnl_dn": sim["pnl"][idx, 1, 0, 0, 0]}).set_index("T")
PER = {"Jun": (SRC[0][2], [f"2026-06-{d:02d}" for d in range(1, 31)]), "Jul": (SRC[1][2], [f"2026-07-{d:02d}" for d in range(1, 32)]),
       "tune": (SRC[2][2], [str(d.date()) for d in pd.date_range("2026-08-20", "2026-09-30")]), "test": (SRC[2][2], [str(d.date()) for d in pd.date_range("2026-10-01", "2026-11-03")]),
       "Nov4-7": (SRC[3][2], [f"2026-11-{d:02d}" for d in range(4, 8)])}
SIM = {k: period_sim(*v) for k, v in PER.items()}
def rule(p, thr, agree, s):
    side_up = p > 0.5; sel = np.isfinite(p) & (np.abs(p - 0.5) >= thr) & s
    if agree is not None: sel &= np.isfinite(agree) & (np.sign(agree) == np.sign(p - 0.5))
    return side_up, sel
rows = []
z5 = W["bp_-60_z_s5"].values
for per, sim in SIM.items():
    j = W.index.get_indexer(sim.index); ok = j >= 0; simv = sim[ok]; j = j[ok]
    pnl_up, pnl_dn = simv.pnl_up.values, simv.pnl_dn.values
    def evalrule(name, p, thr, agree):
        side_up, sel = rule(p[j], thr, None if agree is None else agree[j], np.ones(len(j), bool))
        pnl = np.where(side_up, pnl_up, pnl_dn)[sel]
        if len(pnl) == 0: return
        rows.append(dict(period=per, rule=name, thr=thr, n=int(len(pnl)), win=float((pnl > 0).mean()), edge_c=float(pnl.mean() * 100), usd_at_100=float(pnl.sum() * 200),
                         usd_per_day=float(pnl.sum() * 200 / len(PER[per][1]))))
    evalrule("FROZEN A (m2 >= 0.05 & bp agree)", P["frozen m2 head"], 0.05, z5)
    for nm, p in P.items():
        if nm == "frozen m2 head": continue
        for thr in (0.03, 0.05, 0.07): evalrule(nm, p, thr, None)
        evalrule(nm + " & bp agree", p, 0.05, z5)
    # the new model as a FILTER on frozen A's trades
    side_up, sel = rule(P["frozen m2 head"][j], 0.05, z5[j], np.ones(len(j), bool)); pnl = np.where(side_up, pnl_up, pnl_dn)
    for nm in ("heads+all [logit]", "heads+all [lgb]", "heads+xv [logit]", "heads+book [logit]"):
        if nm not in P: continue
        pn = P[nm][j]; ag = sel & np.isfinite(pn) & ((pn > 0.5) == side_up); dis = sel & np.isfinite(pn) & ((pn > 0.5) != side_up)
        for lab, msk in (("agrees", ag), ("disagrees", dis)):
            if msk.sum(): rows.append(dict(period=per, rule=f"FROZEN A where {nm} {lab}", thr=0.05, n=int(msk.sum()), win=float((pnl[msk] > 0).mean()), edge_c=float(pnl[msk].mean() * 100), usd_at_100=float(pnl[msk].sum() * 200), usd_per_day=np.nan))
E = pd.DataFrame(rows); E.to_csv(f"{OUT}/A_new_inputs_trades.csv", index=False)
print("\n== A rule, $100 per trade, live exits (0.85 sell else settle): frozen vs new-signal variants ==")
for per in PER:
    e = E[(E.period == per)]; print(f"\n--- {per} ---")
    print(e[~e.rule.str.startswith("FROZEN A where")].sort_values(["rule", "thr"])[["rule", "thr", "n", "win", "edge_c", "usd_at_100", "usd_per_day"]].round(3).to_string(index=False))
    print(e[e.rule.str.startswith("FROZEN A where")][["rule", "n", "win", "edge_c", "usd_at_100"]].round(3).to_string(index=False))
