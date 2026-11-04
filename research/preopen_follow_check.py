"""Follow-up of the pre-open grid: the T-5 'follow' cell vs frozen B. (a) overlap and incremental value:
B = decide at T, delta 0.12, bid at last print 10 s, 0.80/0.10/240; C = decide at T-5, delta 0.08 (and 0.06), bid at
last print until the open, 0.80/0.05/240. Windows in both / only one; combined book (C first, B if C did not fire).
(b) neighbourhood of the cell on test. (c) runs on July with MASTER_PATH/POLY_* set (LABEL=july)."""
import os, sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
LABEL = os.environ.get("LABEL", "main"); master, mk, g = ex.master, ex.mk, ex.g
def trades(d, delta, cancel, tp, sl, regime="all"):
    sim = polysim.simulate(mk, g, t_entry=d, entry_mode="maker_last", t_cancel=cancel, tps=(tp,), sls=(sl,), tstops=(240,), slip=0.01); m, sim = ex.align(master, sim)
    mod = m[f"m2_{d}_close_p"].values; mkt = sim["pre_last"][:, 0]; dis = mod - mkt; side = np.where(dis > 0, 0, 1); idx = np.arange(len(m))
    sel = np.isfinite(dis) & (np.abs(dis) >= delta)
    if regime == "follow": sel &= (np.sign(dis) == np.sign(mkt - 0.5)) & (np.abs(mkt - 0.5) > 0)
    pnl = np.where(sel, sim["pnl"][idx, side, 0, 0, 0], np.nan); ent = sim["entry_px"][idx, side]
    return pd.DataFrame({"T": m.index.values, "date": m.date.values, "sel": sel, "pnl": pnl, "entry": ent, "side": side}).set_index("T")
B = trades(0, 0.12, 10, 0.80, 0.10); C8 = trades(-5, 0.08, 0, 0.80, 0.05); C6 = trades(-5, 0.06, 0, 0.80, 0.05); C8f = trades(-5, 0.08, 0, 0.80, 0.05, "follow")
pers = {"old": se.OLD, "tune": se.TUNE, "test": se.TEST} if LABEL == "main" else {LABEL: ("2026-07-01", "2026-07-31")}
def stats(df, per):
    s = (df.date >= per[0]) & (df.date <= per[1]) & df.pnl.notna(); x = df[s]
    if not len(x): return "n=0"
    d = (x.pnl / x.entry * 100).groupby(x.date).sum()
    return f"n={len(x):5d} mean {x.pnl.mean()*100:+.2f}c win {np.mean(x.pnl>0):.3f} ${d.sum():,.0f} dd ${(d.cumsum().cummax()-d.cumsum()).max():,.0f} sharpe {d.mean()/(d.std()+1e-9)*np.sqrt(365):.1f}"
for nm, per in pers.items():
    print(f"\n==== {nm} ====")
    for lab, df in (("B  (T, d0.12, 10s, .80/.10)", B), ("C8 (T-5, d0.08, to open, .80/.05)", C8), ("C6 (T-5, d0.06, to open, .80/.05)", C6), ("C8 follow only", C8f)):
        print(f"{lab:36s} {stats(df, per)}")
    s = (B.date >= per[0]) & (B.date <= per[1])
    both = B.pnl.notna() & C8.pnl.notna() & s; onlyB = B.pnl.notna() & C8.pnl.isna() & s; onlyC = C8.pnl.notna() & B.pnl.isna() & s
    same = (B.side == C8.side) & both
    print(f"windows: both {both.sum()} (same side {same.sum()}), only B {onlyB.sum()}, only C8 {onlyC.sum()}")
    print(f"  on shared windows: B {B.pnl[both].mean()*100:+.2f}c  C8 {C8.pnl[both].mean()*100:+.2f}c | only-B windows: B {B.pnl[onlyB].mean()*100:+.2f}c | only-C8 windows: C8 {C8.pnl[onlyC].mean()*100:+.2f}c (n={onlyC.sum()})")
    comb = C8.copy(); fill = comb.pnl.isna() & B.pnl.notna(); comb.loc[fill, ["pnl", "entry"]] = B.loc[fill, ["pnl", "entry"]].values
    print(f"combined (C8, else B){'':15s} {stats(comb, per)}")
    comb2 = B.copy(); fill = comb2.pnl.isna() & C8.pnl.notna(); comb2.loc[fill, ["pnl", "entry"]] = C8.loc[fill, ["pnl", "entry"]].values
    print(f"combined (B, else C8){'':15s} {stats(comb2, per)}")
    if LABEL == "main":
        dB = (B.pnl / B.entry * 100).fillna(0)[s].groupby(B.date[s]).sum(); dC = (comb.pnl / comb.entry * 100).fillna(0)[s].groupby(comb.date[s]).sum()
        dd = (dC - dB).values; rng = np.random.default_rng(0); b = [np.mean(dd[rng.integers(0, len(dd), len(dd))]) for _ in range(2000)]
        print(f"  paired day bootstrap, combined vs B alone: +${dd.mean():.0f}/day, P(no gain) {np.mean(np.array(b) <= 0):.3f}")
if LABEL == "main":
    grid = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "A_preopen_fade_grid.csv"))
    nb = grid[(grid.regime == "all") & (grid.min_tilt == 0.0) & (grid.tp == 0.80) & (grid.d.isin([-10, -5])) & (grid.delta.isin([0.06, 0.08, 0.10]))]
    print("\n== neighbourhood of the T-5 cell (regime all, tp 0.80) =="); print(nb[["d", "mode", "delta", "sl", "n_tune", "mean_tune", "win_tune", "n_test", "mean_test", "win_test", "usd_test", "tstat_test", "n_old", "mean_old"]].round(3).to_string(index=False))
