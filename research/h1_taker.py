"""H1c: aggressive pre-open entry at 0.51 + taker fee (always filled): decide at d, rest a bid at `bid`, cancel at t_cancel; fill only if a print
of our side <= bid occurs. Exits: TP (maker) / SL / time stop. Signals: m2/bm/stack at the decision offset."""
import os
import numpy as np, pandas as pd, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
TUNE, TEST = se.TUNE, se.TEST
master, mk, g = ex.master, ex.mk, ex.g
rows = []
configs = [(-60, 0, 0.5275), (-50, 0, 0.5275), (-45, 0, 0.5275), (-40, 0, 0.5275), (-35, 0, 0.5275)]
for d, tc, bid in configs:
    sim = polysim.simulate(mk, g, t_entry=d, entry_mode="fixed", pe=0.5275,
                           tps=(0.70, 0.75, 0.80, 0.90, 0.97), sls=(0.05, 0.10, 0.15, 0.20, 0.30), tstops=(240, 270, 290))
    m, sim = ex.align(master, sim)
    sigs = {f"m2_{d}_close_p": m[f"m2_{d}_close_p"].values, f"bm_{d}_close_p": m[f"bm_{d}_close_p"].values}
    p, coef = ex.stacked_signal(m, d, cols=[c for c in (f"m2_{d}_close_p", f"bm_{d}_close_p", "bp_-60_z_s5") if c in m.columns]); sigs[f"stack@{d}"] = p
    for nm, sig in sigs.items():
        side = np.where(sig > 0.5, 0, 1); idx = np.arange(len(m))
        filled = np.isfinite(sim["entry_px"][idx, side])
        for thr in (0.02, 0.04, 0.06, 0.08):
            for ex_nm, (tp, sl, ts) in {"tp75_sl20_t240": (0.75, 0.20, 240), "tp80_sl10_t290": (0.80, 0.10, 290), "tp97_sl05_t290": (0.97, 0.05, 290), "tp75_sl10_t270": (0.75, 0.10, 270)}.items():
                a = list(sim["tps"]).index(tp); b = list(sim["sls"]).index(sl); c = list(sim["tstops"]).index(ts)
                pnl = sim["pnl"][idx, side, a, b, c]
                for split, spn in ((TUNE, "tune"), (TEST, "test")):
                    sel = np.isfinite(sig) & (np.abs(sig - 0.5) >= thr) & (m.date.values >= split[0]) & (m.date.values <= split[1])
                    r = se.metrics(np.where(sel, pnl, np.nan), m.date.values)
                    r.update(d=d, t_cancel=tc, bid=bid, signal=nm, thr=thr, exit=ex_nm, split=spn, fill_rate=filled[sel].mean() if sel.sum() else np.nan,
                             settle=np.nanmean(np.where(sel, sim["settle"][idx, side], np.nan)), n_sig=int(sel.sum()))
                    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "h1_taker_results.csv"), index=False)
piv = df.pivot_table(index=["d", "t_cancel", "bid", "signal", "thr", "exit"], columns="split", values=["n", "mean", "win", "fill_rate", "settle", "tstat"])
piv = piv.reindex(columns=[("n", "tune"), ("fill_rate", "tune"), ("mean", "tune"), ("settle", "tune"), ("win", "tune"), ("n", "test"), ("mean", "test"), ("settle", "test"), ("win", "test"), ("tstat", "test")])
print(piv.round(4).to_string())
