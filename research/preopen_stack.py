"""The user's question: using every pre-open forecast we have, is there a high-win-rate pre-open rule?
Decision at T-45 (entry 0.50 before T-40, frozen exits). Features available at T-45 with no look-ahead:
btcmovement2 at -120/-90/-75/-60/-50/-45 (close head, horizon heads, drift/scale, regime, microstructure),
btc-predict z at -120/-60 (all horizons), btc-movement at -45, and the last pre-open print.
Two learners fitted on TUNE only (logistic, gradient boosting), scored on TEST and Aug 1-19; results by score
decile and for 'top-x%' selections, each compared with the frozen rule with a paired day bootstrap."""
import os, sys, numpy as np, pandas as pd, warnings
from scipy.special import logit
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
master, mk, g = ex.master, ex.mk, ex.g
sim = polysim.simulate(mk, g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=(0.80,), sls=(0.05,), tstops=(240,), slip=0.01)
m, sim = ex.align(master, sim); idx = np.arange(len(m))
pnl_up = sim["pnl"][idx, 0, 0, 0, 0]; pnl_dn = sim["pnl"][idx, 1, 0, 0, 0]; pre = sim["pre_last"][:, 0]
y = m["up"].values; dates = m.date.values
# ---- features (all strictly before T-45)
feats = []
for d in (-120, -90, -75, -60, -50, -45):
    for c in ("close_p", "h10_pup_now", "h30_pup_now", "h60_pup_now", "close_pup_now", "h10_p", "h30_p", "h60_p"):
        k = f"m2_{d}_{c}"
        if k in m: feats.append(k)
for d in (-120, -60):
    for c in ("z_s10s", "z_s15s", "z_s1", "z_s2", "z_s5", "z_f1", "z_f10s"):
        k = f"bp_{d}_{c}"
        if k in m: feats.append(k)
for c in ("close_p", "h10_pup_now", "h30_pup_now", "h60_pup_now", "close_pup_now"):
    k = f"bm_-45_{c}"
    if k in m: feats.append(k)
X = pd.DataFrame(index=m.index)
for k in feats:
    v = m[k].values.astype(float)
    X[k] = logit(np.clip(v, 1e-4, 1 - 1e-4)) if ("_p" in k.split("_")[-1] or k.endswith("_p") or "pup_now" in k) and not k.startswith("bp_") else v
X["pre_last"] = np.where(np.isfinite(pre), pre - 0.5, 0.0); X["pre_known"] = np.isfinite(pre).astype(float)
X["m2_45_abs"] = np.abs(m["m2_-45_close_p"].values - 0.5)
for c in ("lrv_300", "seas_now", "reg_trend_1d", "reg_ll_beta_1h", "reg_catchup_1h", "x_bps", "r_30", "pimb_10"):
    k = f"m2_-45_{c}"
    if k in m: X[k] = m[k].values
X = X.replace([np.inf, -np.inf], np.nan)
ok = X.notna().all(1).values
tune = ok & (dates >= se.TUNE[0]) & (dates <= se.TUNE[1]); test = ok & (dates >= se.TEST[0]) & (dates <= se.TEST[1]); old = ok & (dates < "2026-08-20")
print(f"features {X.shape[1]}, rows ok {ok.sum()}: tune {tune.sum()}, test {test.sum()}, aug1-19 {old.sum()}")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X[tune]); Xs = sc.transform(X.fillna(0))
models = {"logistic": LogisticRegression(C=0.3, max_iter=2000).fit(Xs[tune], y[tune]),
          "gbm": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(X.fillna(0).values[tune], y[tune])}
# frozen rule for comparison
p45 = m["m2_-45_close_p"].values; z5 = m["bp_-60_z_s5"].values
frozen = np.isfinite(p45) & (np.abs(p45 - 0.5) >= 0.05) & np.isfinite(z5) & (np.sign(z5) == np.sign(p45 - 0.5))
frozen_pnl = np.where(frozen, np.where(p45 > 0.5, pnl_up, pnl_dn), np.nan)
def summarize(label, sel, side_up, mask):
    pnl = np.where(sel & mask, np.where(side_up, pnl_up, pnl_dn), np.nan); mt = se.metrics(pnl, dates)
    hit = np.mean((y[sel & mask] == 1) == side_up[sel & mask]) if (sel & mask).any() else np.nan
    base = np.where(mask, frozen_pnl, np.nan); d = pd.Series(np.nan_to_num(pnl) - np.nan_to_num(base)).groupby(dates).sum(); d = d[pd.Series(mask).groupby(dates).any().values]
    rng = np.random.default_rng(0); b = [np.mean(d.values[rng.integers(0, len(d), len(d))]) for _ in range(1000)] if len(d) > 2 else [0]
    return dict(rule=label, n=mt.get("n", 0), per_day=mt.get("n", 0) / max(mt.get("days", 1), 1), hit=hit, win=mt.get("win", np.nan), mean_c=100 * mt.get("mean", np.nan), usd=float(np.nansum(pnl) * 200), dpd_vs_frozen=100 * d.mean(), p_no_gain=float(np.mean(np.array(b) <= 0)))
for nm, mdl in models.items():
    P = mdl.predict_proba(Xs if nm == "logistic" else X.fillna(0).values)[:, 1]; conf = np.abs(P - 0.5); side_up = P > 0.5
    print(f"\n==== {nm}: P(up) from all pre-open forecasts, fitted on tune ====")
    for per, mask in (("tune (in-sample)", tune), ("test", test), ("aug1-19", old)):
        rows = [summarize("frozen rule", frozen, p45 > 0.5, mask)]
        q = np.quantile(conf[tune], [0.5, 0.7, 0.8, 0.9, 0.95])
        for lab, thr in zip(("top 50%", "top 30%", "top 20%", "top 10%", "top 5%"), q):
            rows.append(summarize(f"|P-0.5| {lab} (>= {thr:.3f})", conf >= thr, side_up, mask))
        rows.append(summarize("frozen AND stack agrees", frozen & (side_up == (p45 > 0.5)), p45 > 0.5, mask))
        rows.append(summarize("frozen AND stack top 50% agrees", frozen & (side_up == (p45 > 0.5)) & (conf >= q[0]), p45 > 0.5, mask))
        df = pd.DataFrame(rows); print(f"-- {per} --"); print(df.round(3).to_string(index=False))
    # decile view on test
    dec = pd.qcut(conf[test], 10, labels=False, duplicates="drop"); pnl = np.where(side_up, pnl_up, pnl_dn)[test]; hit = ((y[test] == 1) == side_up[test])
    t = pd.DataFrame({"decile": dec, "hit": hit, "pnl": pnl, "win": pnl > 0}).groupby("decile").agg(n=("hit", "size"), hit=("hit", "mean"), win=("win", "mean"), mean_c=("pnl", lambda x: 100 * np.nanmean(x)))
    print("test by confidence decile (0 = least confident):"); print(t.round(3).to_string())
    if nm == "logistic":
        co = pd.Series(mdl.coef_[0], index=X.columns).sort_values(key=np.abs, ascending=False).head(12); print("largest standardized coefficients:"); print(co.round(3).to_string())
