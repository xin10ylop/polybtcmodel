"""H1-final: user's execution (filled at 0.50 before T-45, TP as resting limit, SL/time-stop taker).
Grid over decision offsets, signals, thresholds, exits; selection on TUNE, report TEST. Also the
'tight TP/SL' reference (0.60/0.45) and regime filters."""
import os
import numpy as np, pandas as pd, sys, warnings, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
TUNE, TEST = se.TUNE, se.TEST
master, mk, g = ex.master, ex.mk, ex.g
TPS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.97); SLS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.45); TSS = (180, 240, 270, 290)
rows = []
sims = {}
for d in (-60, -45, -35):
    sim = polysim.simulate(mk, g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=TPS, sls=SLS, tstops=TSS)
    m, sim = ex.align(master, sim); sims[d] = (m, sim)
    sigs = {f"m2_{d}_close_p": m[f"m2_{d}_close_p"].values, f"bm_{d}_close_p": m[f"bm_{d}_close_p"].values}
    for nm, cols in {f"stack2@{d}": [f"m2_{d}_close_p", f"bm_{d}_close_p"], f"stack3@{d}": [f"m2_{d}_close_p", f"bm_{d}_close_p", "bp_-60_z_s5"]}.items():
        p, coef = ex.stacked_signal(m, d, cols=cols); sigs[nm] = p; print(nm, coef)
    idx = np.arange(len(m))
    for nm, sig in sigs.items():
        side = np.where(sig > 0.5, 0, 1)
        for thr in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
            base = np.isfinite(sig) & (np.abs(sig - 0.5) >= thr)
            for a, tp in enumerate(TPS):
                for b, sl in enumerate(SLS):
                    for c, ts in enumerate(TSS):
                        pnl = sim["pnl"][idx, side, a, b, c]
                        for split, spn in ((TUNE, "tune"), (TEST, "test")):
                            sel = base & (m.date.values >= split[0]) & (m.date.values <= split[1])
                            r = se.metrics(np.where(sel, pnl, np.nan), m.date.values); r.update(d=d, signal=nm, thr=thr, tp=tp, sl=sl, tstop=ts, split=spn)
                            rows.append(r)
            for split, spn in ((TUNE, "tune"), (TEST, "test")):
                sel = base & (m.date.values >= split[0]) & (m.date.values <= split[1])
                r = se.metrics(np.where(sel, sim["settle"][idx, side], np.nan), m.date.values); r.update(d=d, signal=nm, thr=thr, tp=np.nan, sl=np.nan, tstop="settle", split=spn); rows.append(r)
df = pd.DataFrame(rows); df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "h1_final_results.csv"), index=False)
key = ["d", "signal", "thr", "tp", "sl", "tstop"]
tune = df[(df.split == "tune") & (df.n >= 300)]; test = df[df.split == "test"]
mg = tune.merge(test, on=key, suffixes=("_tune", "_test"))
mg["score"] = mg.mean_tune * np.sqrt(mg.n_tune)        # t-like selection score on tune
print("\n== top 25 by tune score (TP/SL exits only) ==")
cols = key + ["n_tune", "mean_tune", "win_tune", "sharpe_tune", "n_test", "mean_test", "win_test", "sharpe_test", "maxdd_test", "tstat_test"]
print(mg[mg.tstop != "settle"].sort_values("score", ascending=False).head(25)[cols].round(4).to_string(index=False))
print("\n== top 10 by tune mean (n>=300), TP/SL exits ==")
print(mg[mg.tstop != "settle"].sort_values("mean_tune", ascending=False).head(10)[cols].round(4).to_string(index=False))
print("\n== user's tight exits (0.60/0.45, t290) for the stack3@-45 signal ==")
print(mg[(mg.signal == "stack3@-45") & (mg.tp == 0.60) & (mg.sl == 0.45) & (mg.tstop == 290)][cols].round(4).to_string(index=False))
print("\n== settlement reference ==")
print(mg[mg.tstop == "settle"].sort_values("score", ascending=False).head(8)[cols].round(4).to_string(index=False))
