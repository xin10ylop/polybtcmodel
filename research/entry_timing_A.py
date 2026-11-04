"""Strategy A decided later than T-45: at T+d (d = -35, -25, -15, -10) with the close head and the
btc-predict agreement available then, buying (a) as a taker at the first print after T+d (+1 c slip, taker
fee), (b) with a resting bid at the last print (filled on a print at/below it within 10 s), or (c) a resting
bid at 0.50. Frozen exits. Compared with the frozen T-45 / 0.50 rule on tune and test."""
import os, sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
master, mk, g = ex.master, ex.mk, ex.g
TP, SL, TS = 0.80, 0.05, 240
rows = []
def row(label, m, sim, p, sel, side):
    idx = np.arange(len(m)); pnl = np.where(sel, sim["pnl"][idx, side, 0, 0, 0], np.nan); ent = sim["entry_px"][idx, side]
    r = {"rule": label}
    for nm, per in (("tune", se.TUNE), ("test", se.TEST)):
        s = (m.date.values >= per[0]) & (m.date.values <= per[1]); x = np.where(s, pnl, np.nan); mt = se.metrics(x, m.date.values)
        f = s & np.isfinite(pnl)
        usd = (pnl[f] / ent[f] * 100).sum() if f.any() else 0.0
        r.update({f"n_{nm}": mt.get("n", 0), f"signals_{nm}": int((sel & s).sum()), f"entry_{nm}": float(np.nanmean(ent[f])) if f.any() else np.nan, f"mean_{nm}": mt.get("mean", np.nan), f"win_{nm}": mt.get("win", np.nan), f"usd_{nm}": usd, f"maxdd_{nm}": mt.get("maxdd", np.nan)})
    rows.append(r)
z = master["bp_-60_z_s5"].values
# frozen reference
sim = polysim.simulate(mk, g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=(TP,), sls=(SL,), tstops=(TS,), slip=0.01); m, sim = ex.align(master, sim)
p = m["m2_-45_close_p"].values; zz = m["bp_-60_z_s5"].values; side = np.where(p > 0.5, 0, 1)
sel = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05) & np.isfinite(zz) & (np.sign(zz) == np.sign(p - 0.5)); row("T-45 fixed 0.50 (frozen)", m, sim, p, sel, side)
for d in (-35, -25, -15, -10):
    col = f"m2_{d}_close_p"
    for mode, kw in (("taker", dict(entry_mode="taker")), ("bid at last print, 10 s", dict(entry_mode="maker_last", t_cancel=d + 10)), ("bid 0.50, until T", dict(entry_mode="maker", bid=0.50, t_cancel=0))):
        sim = polysim.simulate(mk, g, t_entry=d, tps=(TP,), sls=(SL,), tstops=(TS,), slip=0.01, **kw); m, sim = ex.align(master, sim)
        p = m[col].values; zz = m["bp_-60_z_s5"].values; side = np.where(p > 0.5, 0, 1)
        for thr in (0.05, 0.08):
            sel = np.isfinite(p) & (np.abs(p - 0.5) >= thr) & np.isfinite(zz) & (np.sign(zz) == np.sign(p - 0.5))
            row(f"T{d:+d} {mode} thr {thr}", m, sim, p, sel, side)
df = pd.DataFrame(rows); df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "A_entry_timing.csv"), index=False)
print("(entry = average fill price; usd at $100 per trade; signals = decisions, n = filled trades)")
print(df.round(3).to_string(index=False))
