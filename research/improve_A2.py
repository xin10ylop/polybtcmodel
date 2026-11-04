"""Improvement pass 2 for A with the current signal (m2 close head at T-45): win-rate/profit trade-off of
exits, when-not-to-trade filters (consistency on tune AND test), confidence-scaled stakes, decision offset."""
import os, sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
TUNE, TEST = se.TUNE, se.TEST
master, mk, g = ex.master, ex.mk, ex.g
TPS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.97); SLS = (0.05, 0.10, 0.20, 0.30); TSS = (240, 290)
sim = polysim.simulate(mk, g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=TPS, sls=SLS, tstops=TSS)
m, sim = ex.align(master.copy(), sim); idx = np.arange(len(m))
tune = (m.date.values >= TUNE[0]) & (m.date.values <= TUNE[1]); test = (m.date.values >= TEST[0]) & (m.date.values <= TEST[1])
def cell(tp, sl, ts): return list(TPS).index(tp), list(SLS).index(sl), list(TSS).index(ts)
def report(pnl, sel, label, w=None):
    out = {"config": label}
    for s, nm in ((tune, "tune"), (test, "test")):
        x = np.where(sel & s, pnl, np.nan) if w is None else np.where(sel & s, pnl * w, np.nan)
        r = se.metrics(x, m.date.values); out.update({f"{k}_{nm}": r.get(k, np.nan) for k in ("n", "mean", "win", "sharpe", "maxdd", "tstat")})
    return out
for d in (-45, -40):
    p = m[f"m2_{d}_close_p"].values; side = np.where(p > 0.5, 0, 1)
    print(f"\n######## decision offset {d} ########")
    print("== exits trade-off (thr 0.05): win rate vs mean ==")
    rows = []
    base = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05)
    for tp in TPS:
        for sl in (0.10, 0.20, 0.30):
            for ts in TSS:
                a, b, c = cell(tp, sl, ts); rows.append(report(sim["pnl"][idx, side, a, b, c], base, f"tp{tp}/sl{sl}/t{ts}"))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\n== thresholds for tp0.80/sl0.20/t240 and tp0.97/sl0.10/t290 ==")
    rows = []
    for thr in (0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10):
        b_ = np.isfinite(p) & (np.abs(p - 0.5) >= thr)
        for ex_ in ((0.80, 0.20, 240), (0.97, 0.10, 290)):
            a, b, c = cell(*ex_); rows.append(report(sim["pnl"][idx, side, a, b, c], b_, f"thr{thr} tp{ex_[0]}/sl{ex_[1]}/t{ex_[2]}"))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\n== filters (thr 0.05, tp0.80/sl0.20/t240): keep only if positive on BOTH tune and test ==")
    a, b, c = cell(0.80, 0.20, 240); pnl = sim["pnl"][idx, side, a, b, c]
    vol = m.vol_prev5m_bp.values; vq = np.nanquantile(vol[tune], [1 / 3, 2 / 3]); hour = m.hour_et.values
    pre = sim["pre_last"][idx, side]                               # last pre-open print of our side (NaN if none)
    filt = {"all": np.ones(len(m), bool), "vol low": vol < vq[0], "vol mid": (vol >= vq[0]) & (vol < vq[1]), "vol high": vol >= vq[1],
            "weekend": m.weekend.values == 1, "weekday": m.weekend.values == 0, "US session 9-16 ET": m.us_session.values == 1,
            "ET 0-4": hour < 4, "ET 4-8": (hour >= 4) & (hour < 8), "ET 8-12": (hour >= 8) & (hour < 12), "ET 12-16": (hour >= 12) & (hour < 16),
            "ET 16-20": (hour >= 16) & (hour < 20), "ET 20-24": hour >= 20,
            "pre-open print of our side <= 0.50": pre <= 0.50, "pre-open print of our side <= 0.52": pre <= 0.52, "pre-open print > 0.52 (skip?)": pre > 0.52,
            "no pre-open print": ~np.isfinite(pre), "|p-0.5|>=0.05 & 10s head agrees": np.sign(m[f"m2_{d}_h10_pup_now"].values - 0.5) == np.sign(p - 0.5),
            "bm agrees": np.sign(m[f"bm_{d}_close_p"].values - 0.5) == np.sign(p - 0.5), "bp 5m agrees": np.sign(m["bp_-60_z_s5"].values) == np.sign(p - 0.5)}
    rows = [report(pnl, base & f, k) for k, f in filt.items()]
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\n== confidence-scaled stake (thr 0.05, tp0.80/sl0.20/t240): weight = |p-0.5|/0.05 capped at 3 ==")
    w = np.clip(np.abs(p - 0.5) / 0.05, 0, 3)
    for nm, ww in (("flat", None), ("scaled", w)):
        r = report(pnl, base, nm, ww); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
