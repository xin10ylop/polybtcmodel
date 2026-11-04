"""User request: A with the sell limit placed right after the 0.50 fill at 0.53 / 0.54 / 0.55 (instead of
0.80), entry at T-45 (fixed 0.50) or T-30 (fixed 0.50, and a resting 0.50 bid from T-30 to T), with the
catastrophe stop 0.05 and time-stop T+240; also the variant where the tight limit is cancelled at the open
and replaced by the frozen exits. Tune / test, cents per share and $ at $100 per trade."""
import os, sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
master, mk, g = ex.master, ex.mk, ex.g
rows = []
def report(label, m, sim, sel, side, a=0, b=0, c=0):
    idx = np.arange(len(m)); pnl = np.where(sel, sim["pnl"][idx, side, a, b, c], np.nan); ent = sim["entry_px"][idx, side]; k = sim["kind"][idx, side, a, b, c]
    r = {"rule": label}
    for nm, per in (("tune", se.TUNE), ("test", se.TEST)):
        s = (m.date.values >= per[0]) & (m.date.values <= per[1]); mt = se.metrics(np.where(s, pnl, np.nan), m.date.values); f = s & np.isfinite(pnl)
        r.update({f"n_{nm}": mt.get("n", 0), f"mean_{nm}": mt.get("mean", np.nan), f"win_{nm}": mt.get("win", np.nan), f"p_tp_{nm}": float(np.mean(k[f] == 0)) if f.any() else np.nan,
                  f"usd_{nm}": float((pnl[f] / ent[f] * 100).sum()), f"maxdd_{nm}": mt.get("maxdd", np.nan) * 200, f"sharpe_{nm}": mt.get("sharpe", np.nan)})
    rows.append(r)
TPS = (0.53, 0.54, 0.55, 0.80); SLS = (0.05,); TSS = (240,)
for d, col, mode, kw in ((-45, "m2_-45_close_p", "fixed 0.50", dict(entry_mode="fixed", pe=0.50)), (-30, "m2_-35_close_p", "fixed 0.50", dict(entry_mode="fixed", pe=0.50)), (-30, "m2_-35_close_p", "bid 0.50 until T", dict(entry_mode="maker", bid=0.50, t_cancel=0))):
    sim = polysim.simulate(mk, g, t_entry=d, tps=TPS, sls=SLS, tstops=TSS, slip=0.01, **kw); m, sim = ex.align(master, sim)
    p = m[col].values; z = m["bp_-60_z_s5"].values; side = np.where(p > 0.5, 0, 1)
    sel = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05) & np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5))
    for a, tp in enumerate(TPS):
        report(f"entry T{d:+d} {mode}, sell limit {tp:.2f} (held to T+240), stop 0.05", m, sim, sel, side, a=a)
df = pd.DataFrame(rows); df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "A_tight_tp.csv"), index=False)
print("(mean = cents per share /100; p_tp = share of trades closed at the limit; usd and maxdd at $100 per trade)")
print(df.round(4).to_string(index=False))
