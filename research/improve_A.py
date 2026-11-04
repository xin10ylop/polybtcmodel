"""Improvement pass for strategy A (pre-open, filled at 0.50 by T-45). Selection on TUNE only."""
import os, sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
TUNE, TEST, OLD = se.TUNE, se.TEST, se.OLD
master, mk, g = ex.master, ex.mk, ex.g
m = master
paths = np.load("/home/user/data/truth/paths.npy"); tw = pd.read_parquet("/home/user/data/truth/windows.parquet")
ti = {t: i for i, t in enumerate(tw["T"].values)}; rows = np.array([ti[t] for t in m.index]); lp = np.log(paths[rows])   # col 120 = T
# ---------------------------------------------------------------- TWAP label and partial-TWAP features at T-45
twap_s = lp[:, 120 - 60:120].mean(1); twap_e = lp[:, 120 + 240:120 + 300].mean(1)
m["y_twap"] = (twap_e >= twap_s).astype(int)
print("TWAP label agreement with Polymarket outcome:", (m.y_twap == m.up).mean().round(4), " vs point label:", (m.up_binance_eff == m.up).mean().round(4))
vol = np.maximum(m.vol_prev5m_bp.values, 1.0) / 1e4
known = lp[:, 120 - 60:120 - 45].mean(1)                         # known 15 s of the start TWAP
m["ptw_disp"] = (lp[:, 120 - 46] - known) / (vol * np.sqrt(45 / 300))          # current price vs known part of the reference
m["r15"] = (lp[:, 120 - 46] - lp[:, 120 - 61]) / (vol * np.sqrt(15 / 300))
m["r60"] = (lp[:, 120 - 46] - lp[:, 120 - 106]) / (vol * np.sqrt(60 / 300))
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def per(name): a, b = {"TUNE": TUNE, "TEST": TEST, "OLD": OLD}[name]; return (m.date.values >= a) & (m.date.values <= b)
FEATS = {
    "m2 close_p (current)": ["m2_-45_close_p"],
    "m2 + ptw_disp": ["m2_-45_close_p", "ptw_disp"],
    "m2 + bm + bp": ["m2_-45_close_p", "bm_-45_close_p", "bp_-60_z_s5"],
    "m2 + bm + bp + ptw_disp": ["m2_-45_close_p", "bm_-45_close_p", "bp_-60_z_s5", "ptw_disp"],
    "m2 heads + bm + bp + ptw_disp + r15": ["m2_-45_close_p", "m2_-45_h60_pup_now", "m2_-45_h30_pup_now", "bm_-45_close_p", "bm_-45_h30_pup_now", "bp_-60_z_s5", "bp_-60_z_s1", "ptw_disp", "r15"],
}
def X_of(cols):
    return np.column_stack([m[c].values if (c.startswith("bp_") or c in ("ptw_disp", "r15", "r60")) else logit(m[c].values) for c in cols])
res = []; sigs = {}
for nm, cols in FEATS.items():
    X = X_of(cols); ok = np.isfinite(X).all(1)
    for label in ("y_twap", "up_binance_eff"):
        tr = per("TUNE") & ok
        lr = LogisticRegression(C=1.0).fit(X[tr], m[label].values[tr])
        p = np.full(len(m), np.nan); p[ok] = lr.predict_proba(X[ok])[:, 1]
        # out-of-fold on tune (alternate days) for an honest tune AUC
        days = m.date.values; ud = np.unique(days[tr]); f1 = np.isin(days, ud[::2])
        poof = np.full(len(m), np.nan)
        for a_, b_ in ((tr & f1, tr & ~f1), (tr & ~f1, tr & f1)):
            l2 = LogisticRegression(C=1.0).fit(X[a_], m[label].values[a_]); poof[b_] = l2.predict_proba(X[b_])[:, 1]
        r = {"signal": nm, "label": label, "auc_tune_oof": roc_auc_score(m.up.values[tr], poof[tr]), "auc_test": roc_auc_score(m.up.values[per("TEST") & ok], p[per("TEST") & ok]),
             "auc_old": roc_auc_score(m.up.values[per("OLD") & ok], p[per("OLD") & ok]), "coef": np.round(lr.coef_[0], 3).tolist()}
        res.append(r); sigs[(nm, label)] = p
print(pd.DataFrame(res).round(4).to_string(index=False))
# ---------------------------------------------------------------- simulation with the user's execution
sim = polysim.simulate(mk, g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=(0.70, 0.75, 0.80, 0.90, 0.97), sls=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30), tstops=(180, 240, 270, 290))
sims_trail = {tr_: polysim.simulate(mk, g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=(0.80, 0.90, 0.97), sls=(0.10, 0.20), tstops=(240, 290), trail=tr_)
              for tr_ in ((0.65, 0.50), (0.70, 0.55), (0.75, 0.60))}
mm, sim = ex.align(master.copy(), sim)
for k in sims_trail: _, sims_trail[k] = ex.align(master.copy(), sims_trail[k])
keep = np.isin(m.index.values, mm.index.values)
idx = np.arange(len(mm))
def run_sig(p, name, thr_grid=(0.03, 0.04, 0.05, 0.06, 0.08)):
    p = p[keep]; side = np.where(p > 0.5, 0, 1); rows = []
    for thr in thr_grid:
        base = np.isfinite(p) & (np.abs(p - 0.5) >= thr)
        for a, tp in enumerate(sim["tps"]):
            for b, sl in enumerate(sim["sls"]):
                for c, ts in enumerate(sim["tstops"]):
                    pnl = sim["pnl"][idx, side, a, b, c]; r = {"signal": name, "thr": thr, "tp": tp, "sl": sl, "tstop": ts, "trail": "-"}
                    for split, nm in ((TUNE, "tune"), (TEST, "test")):
                        s = base & (mm.date.values >= split[0]) & (mm.date.values <= split[1]); mt = se.metrics(np.where(s, pnl, np.nan), mm.date.values)
                        for k_ in ("n", "mean", "win", "sharpe", "maxdd", "tstat"): r[f"{k_}_{nm}"] = mt.get(k_, np.nan)
                    rows.append(r)
        for tr_, st in sims_trail.items():
            for a, tp in enumerate(st["tps"]):
                for b, sl in enumerate(st["sls"]):
                    for c, ts in enumerate(st["tstops"]):
                        pnl = st["pnl"][idx, side, a, b, c]; r = {"signal": name, "thr": thr, "tp": tp, "sl": sl, "tstop": ts, "trail": f"{tr_[0]}->{tr_[1]}"}
                        for split, nm in ((TUNE, "tune"), (TEST, "test")):
                            s = base & (mm.date.values >= split[0]) & (mm.date.values <= split[1]); mt = se.metrics(np.where(s, pnl, np.nan), mm.date.values)
                            for k_ in ("n", "mean", "win", "sharpe", "maxdd", "tstat"): r[f"{k_}_{nm}"] = mt.get(k_, np.nan)
                        rows.append(r)
    return pd.DataFrame(rows)
allr = []
cands = {"current m2": m["m2_-45_close_p"].values}
for (nm, label), p in sigs.items():
    if label == "y_twap" and nm != "m2 close_p (current)": cands[f"stack[{nm}]"] = p
for nm, p in cands.items():
    allr.append(run_sig(p, nm))
R = pd.concat(allr, ignore_index=True); R.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "A_improve_grid.csv"), index=False)
Rk = R[R.n_tune >= 400].copy(); Rk["score"] = Rk.mean_tune * np.sqrt(Rk.n_tune)
cols = ["signal", "thr", "tp", "sl", "tstop", "trail", "n_tune", "mean_tune", "win_tune", "n_test", "mean_test", "win_test", "sharpe_test", "maxdd_test", "tstat_test"]
print("\n== top 25 by tune score (n_tune >= 400) ==")
print(Rk.sort_values("score", ascending=False).head(25)[cols].round(4).to_string(index=False))
print("\n== best per signal (tune score), exits with sl >= 0.10 ==")
sub = Rk[Rk.sl >= 0.10]
print(sub.sort_values("score", ascending=False).groupby("signal").head(2)[cols].round(4).to_string(index=False))
