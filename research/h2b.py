"""H2b: post-reference disagreement with (a) maker entry at the last print, (b) agreement with the
pre-reference stack, (c) regime slices. TUNE period only (+TEST shown for the few pre-registered cells)."""
import os
import numpy as np, pandas as pd, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
TUNE, TEST = se.TUNE, se.TEST
master, mk, g = ex.master, ex.mk, ex.g
EX = {"tp97_sl05_t290": (0.97, 0.05, 290), "tp80_sl10_t240": (0.80, 0.10, 240), "tp75_sl20_t240": (0.75, 0.20, 240)}
pre_stack, coef = ex.stacked_signal(master, -45, cols=[c for c in ("m2_-45_close_p", "bm_-45_close_p", "bp_-60_z_s5") if c in master.columns])
master["pre_stack"] = pre_stack
rows = []
for d in (-10, -5, 0, 5):
    for mode, tc in (("taker", 0), ("maker_last", d + 10), ("maker_last", d + 30)):
        sim = polysim.simulate(mk, g, t_entry=d, entry_mode=mode, t_cancel=tc, tps=(0.75, 0.80, 0.97), sls=(0.05, 0.10, 0.20), tstops=(240, 290),
                               entry_fee=(mode == "taker"))
        m, sim = ex.align(master, sim)
        mod = m[f"m2_{d}_close_p"].values; mkt = sim["pre_last"][:, 0]
        dis = mod - mkt; idx = np.arange(len(m))
        vol = m.vol_prev5m_bp.values; vq = np.nanquantile(vol[(m.date.values >= TUNE[0]) & (m.date.values <= TUNE[1])], [1 / 3, 2 / 3])
        agree = np.sign(m.pre_stack.values - 0.5) == np.sign(dis)
        for delta in (0.05, 0.08, 0.12):
            side = np.where(dis > 0, 0, 1)
            base = np.isfinite(dis) & (np.abs(dis) >= delta)
            filt = {"all": np.ones(len(m), bool), "agree pre-stack": agree, "vol low": vol < vq[0], "vol mid": (vol >= vq[0]) & (vol < vq[1]),
                    "vol high": vol >= vq[1], "weekend": m.weekend.values == 1, "ET 0-8h": m.hour_et.values < 8, "ET 8-16h": (m.hour_et.values >= 8) & (m.hour_et.values < 16),
                    "ET 16-24h": m.hour_et.values >= 16}
            for ex_nm, (tp, sl, ts) in EX.items():
                a = list(sim["tps"]).index(tp); b = list(sim["sls"]).index(sl); c = list(sim["tstops"]).index(ts)
                pnl = sim["pnl"][idx, side, a, b, c]
                for fn, f in filt.items():
                    for split, spn in ((TUNE, "tune"), (TEST, "test")):
                        sel = base & f & (m.date.values >= split[0]) & (m.date.values <= split[1])
                        r = se.metrics(np.where(sel, pnl, np.nan), m.date.values)
                        r.update(d=d, mode=f"{mode}@{tc}", delta=delta, exit=ex_nm, filter=fn, split=spn, n_sig=int(sel.sum()),
                                 fill=float(np.isfinite(pnl[sel]).mean()) if sel.sum() else np.nan,
                                 settle=float(np.nanmean(np.where(sel, sim["settle"][idx, side], np.nan))))
                        rows.append(r)
df = pd.DataFrame(rows); df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "h2b_results.csv"), index=False)
t = df[(df.split == "tune") & (df.n >= 100)].sort_values("mean", ascending=False)
print("== TUNE: top 40 cells (n>=100) ==")
print(t.head(40)[["d", "mode", "delta", "exit", "filter", "n", "fill", "mean", "settle", "win", "sharpe", "tstat"]].round(4).to_string(index=False))
print("\n== TUNE: 'all' filter cells ==")
print(df[(df.split == "tune") & (df["filter"] == "all")][["d", "mode", "delta", "exit", "n", "fill", "mean", "settle", "win", "tstat"]].round(4).to_string(index=False))
