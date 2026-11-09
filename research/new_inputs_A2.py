"""Follow-up to new_inputs_A.py (reads exports/new_inputs_features.parquet, refits in seconds).
1. equal trade count: per period, the top-n most confident windows of each model, n = frozen A's count -> win / edge / $.
2. group ablation of heads+all [logit] (drop cross-venue / book / metrics) per month.
3. the logistic coefficients of the Oct and Nov models (what carries the gain).
4. weekly stability of the new rule vs frozen A in Oct-Nov.
5. availability delay: book and metrics features taken 60 s older (as of T-107) -> does the gain survive?
Saves exports/new_inputs_preds.parquet (walk-forward probabilities per model)."""
import os, sys, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
sys.path.insert(0, "/home/user/polybtcmodel/polybtc"); import polysim  # noqa: E402
EXTRA = "/home/user/data/extra"; LAT = 47; OUT = "/home/user/polybtcmodel/reports"; EXP = "/home/user/polybtcmodel/exports"
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
HEADS = ["m2_-45_close_p", "bp_-60_z_s5", "bp_-60_z_s1"]
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
F = pd.read_parquet(f"{EXP}/new_inputs_features.parquet").reindex(W.index)
T = W.index.values.astype(np.int64)
XV = [c for c in F.columns if not c.startswith(("ret_bn", "imb_bn", "vol_bn", "bk_", "mt_"))]
BOOK = [c for c in F.columns if c.startswith("bk_") and c != "bk_age"]; MET = [c for c in F.columns if c.startswith("mt_")]
print("features xv", len(XV), "book", len(BOOK), "met", len(MET), flush=True)

# ---- delayed book / metrics (as of T-107)
def asof(tab, when):
    idx = tab.index.values; j = np.searchsorted(idx, when, side="right") - 1; ok = j >= 0
    out = pd.DataFrame(np.where(ok[:, None], tab.values[np.maximum(j, 0)], np.nan), columns=tab.columns); out["age"] = np.where(ok, when - idx[np.maximum(j, 0)], np.nan); return out
bd = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{EXTRA}/bookdepth/*.parquet"))]).sort_index(); bd = bd[~bd.index.duplicated()]
for lv in ("0.2", "1.0", "2.0", "5.0"): bd[f"imb{lv}"] = (bd[f"n-{lv}"] - bd[f"n+{lv}"]) / (bd[f"n-{lv}"] + bd[f"n+{lv}"])
bd["ldepth1"] = np.log(bd["n-1.0"] + bd["n+1.0"]); bd["ldepth1_rel"] = bd.ldepth1 - bd.ldepth1.rolling(2880, min_periods=200).median()
bd["imb_outer"] = ((bd["n-1.0"] - bd["n-0.2"]) - (bd["n+1.0"] - bd["n+0.2"])) / ((bd["n-1.0"] - bd["n-0.2"]) + (bd["n+1.0"] - bd["n+0.2"]))
mt = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{EXTRA}/metrics/*.parquet"))]).sort_index(); mt = mt[~mt.index.duplicated()]
mt["loi"] = np.log(mt.sum_open_interest); mt["doi_5"] = mt.loi.diff(1); mt["doi_15"] = mt.loi.diff(3); mt["doi_60"] = mt.loi.diff(12)
mt["ltaker"] = np.log(mt.sum_taker_long_short_vol_ratio); mt["ltop_cnt"] = np.log(mt.count_toptrader_long_short_ratio); mt["ltop_sum"] = np.log(mt.sum_toptrader_long_short_ratio)
mt["lls_cnt"] = np.log(mt.count_long_short_ratio); mt["dtop_cnt_60"] = mt.ltop_cnt.diff(12); mt["dlls_60"] = mt.lls_cnt.diff(12); mt["dtaker_15"] = mt.ltaker.diff(3)
MCOLS = ["doi_5", "doi_15", "doi_60", "ltaker", "ltop_cnt", "ltop_sum", "lls_cnt", "dtop_cnt_60", "dlls_60", "dtaker_15"]
def delayed(extra):
    G = pd.DataFrame(index=W.index); when = T - LAT - extra
    now = asof(bd[["imb0.2", "imb1.0", "imb2.0", "imb5.0", "ldepth1_rel", "imb_outer"]], when); ago = asof(bd[["imb0.2", "imb1.0"]], when - 300)
    for col in ("imb0.2", "imb1.0", "imb2.0", "imb5.0", "ldepth1_rel", "imb_outer"): G[f"bk_{col}"] = now[col].values
    G["bk_dimb0.2_300"] = now["imb0.2"].values - ago["imb0.2"].values; G["bk_dimb1.0_300"] = now["imb1.0"].values - ago["imb1.0"].values
    G.loc[now.age.values > 120 + extra, [c for c in G.columns if c.startswith("bk_")]] = np.nan
    nm_ = asof(mt[MCOLS], when)
    for col in MCOLS: G[f"mt_{col}"] = nm_[col].values
    G.loc[nm_.age.values > 900 + extra, [f"mt_{c_}" for c_ in MCOLS]] = np.nan
    return G
G60 = delayed(60)

MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; mon = W.month.values; y = W.up.values.astype(int)
base = pd.DataFrame({"l_m2": logit(W["m2_-45_close_p"].values), "z5": W["bp_-60_z_s5"].values, "z1": W["bp_-60_z_s1"].values}, index=W.index)
LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=20.0, verbose=-1, num_threads=4)
def fit_predict(X_all, cols, learner, keep_models=False):
    pred = np.full(len(W), np.nan); models = {}
    for mo in MONTHS:
        te = mon == mo
        tr = np.isin(mon, [m_ for m_ in MONTHS if MONTHS.index(m_) < MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
        X = X_all[cols].values; okx = np.isfinite(X).all(1); tr &= okx; te_ok = te & okx
        if tr.sum() < 2000 or te_ok.sum() == 0: continue
        if learner == "logit":
            sc = StandardScaler().fit(X[tr]); lr = LogisticRegression(C=0.05, max_iter=500).fit(sc.transform(X[tr]), y[tr]); pred[te_ok] = lr.predict_proba(sc.transform(X[te_ok]))[:, 1]
            if keep_models: models[mo] = pd.Series(lr.coef_[0], index=cols)
        else:
            b = lgb.train(LGB, lgb.Dataset(X[tr], y[tr]), num_boost_round=300); pred[te_ok] = b.predict(X[te_ok])
    return (pred, models) if keep_models else pred
X_all = pd.concat([base, F, W[RAW]], axis=1); X_del = pd.concat([base, F[XV], G60, W[RAW]], axis=1)
ALL = list(base.columns) + XV + BOOK + MET
P = {"frozen m2 head": W["m2_-45_close_p"].values}
P["heads [logit]"] = fit_predict(X_all, list(base.columns), "logit")
P["heads+all [logit]"], COEF = fit_predict(X_all, ALL, "logit", keep_models=True)
P["heads+all [lgb]"] = fit_predict(X_all, ALL, "lgb")
P["heads+all+raw [logit]"] = fit_predict(X_all, ALL + RAW, "logit")
P["heads+all -xv [logit]"] = fit_predict(X_all, list(base.columns) + BOOK + MET, "logit")
P["heads+all -book [logit]"] = fit_predict(X_all, list(base.columns) + XV + MET, "logit")
P["heads+all -metrics [logit]"] = fit_predict(X_all, list(base.columns) + XV + BOOK, "logit")
P["heads+all DELAY60 [logit]"] = fit_predict(X_del, ALL, "logit")
P["heads+all DELAY60 [lgb]"] = fit_predict(X_del, ALL, "lgb")
print("fitted", flush=True)
pd.DataFrame(P, index=W.index).to_parquet(f"{EXP}/new_inputs_preds.parquet")

rows = []
for nm, p in P.items():
    for mo in MONTHS:
        s = (mon == mo) & np.isfinite(p) & np.isfinite(P["frozen m2 head"])
        rows.append(dict(model=nm, month=mo, auc=roc_auc_score(y[s], p[s]), n=int(s.sum())))
R = pd.DataFrame(rows); print("\n== AUC walk-forward (ablation and 60-s delay) =="); print(R.pivot(index="model", columns="month", values="auc")[MONTHS].round(3).to_string())
print("\n== logistic coefficients (standardized) of heads+all, Oct and Nov models: top 15 by |coef| ==")
for mo in ("Oct", "Nov", "Sep"):
    cf = COEF[mo]; print(mo, cf.reindex(cf.abs().sort_values(ascending=False).index[:15]).round(3).to_dict())

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
z5 = W["bp_-60_z_s5"].values
print("\n== EQUAL TRADE COUNT: each model's n most confident windows, n = frozen A's trades in the period (live exits, $100/trade) ==")
rows = []; weekly = []
for per, sim in SIM.items():
    j = W.index.get_indexer(sim.index); ok = j >= 0; simv = sim[ok]; j = j[ok]; pnl_up, pnl_dn = simv.pnl_up.values, simv.pnl_dn.values
    pf = P["frozen m2 head"][j]; sel_f = np.isfinite(pf) & (np.abs(pf - 0.5) >= 0.05) & np.isfinite(z5[j]) & (np.sign(z5[j]) == np.sign(pf - 0.5))
    n_f = int(sel_f.sum()); pnl_f = np.where(pf > 0.5, pnl_up, pnl_dn)[sel_f]
    rows.append(dict(period=per, model="FROZEN A", n=n_f, win=(pnl_f > 0).mean(), edge_c=pnl_f.mean() * 100, usd=pnl_f.sum() * 200))
    for nm, p in P.items():
        if nm == "frozen m2 head": continue
        pj = p[j]; conf = np.where(np.isfinite(pj), np.abs(pj - 0.5), -1); order = np.argsort(-conf)[:n_f]; sel = np.zeros(len(j), bool); sel[order[conf[order] > 0]] = True
        pnl = np.where(pj > 0.5, pnl_up, pnl_dn)[sel]; thr_eff = conf[order].min()
        rows.append(dict(period=per, model=nm, n=int(sel.sum()), win=(pnl > 0).mean(), edge_c=pnl.mean() * 100, usd=pnl.sum() * 200, thr_eff=thr_eff))
        if per in ("test", "Nov4-7") and nm in ("heads+all [logit]", "heads+all+raw [logit]"):
            dates = W.date.values[j]; wk = pd.to_datetime(dates).isocalendar().week.values
            for w in np.unique(wk):
                m_ = sel & (wk == w); mf = sel_f & (wk == w)
                weekly.append(dict(week=int(w), model=nm, n=int(m_.sum()), win=float((np.where(pj > 0.5, pnl_up, pnl_dn)[m_] > 0).mean()) if m_.sum() else np.nan,
                                   n_frozen=int(mf.sum()), win_frozen=float((np.where(pf > 0.5, pnl_up, pnl_dn)[mf] > 0).mean()) if mf.sum() else np.nan))
E = pd.DataFrame(rows); E.to_csv(f"{OUT}/A_new_inputs_equal_count.csv", index=False)
for per in PER: print(f"--- {per} ---"); print(E[E.period == per].drop(columns="period").round(3).to_string(index=False))
print("\n== weekly win rate, equal-count rule vs frozen A, Oct-Nov ==")
print(pd.DataFrame(weekly).round(3).to_string(index=False))
