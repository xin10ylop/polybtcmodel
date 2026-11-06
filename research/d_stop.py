"""D with a stop-loss, and D with stricter entries: can the win rate / P&L be improved? D buys at ~0.86 and holds to settlement;
a loser costs the whole entry. Variants: sell at the market when our side prints <= stop (taker fee, 1 c slippage), else settle;
and higher model confidence q / lower maxpx. LABEL=main (Aug 1-19 / tune / test) or june / july with MASTER_PATH, POLY_* set."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "polybtc"))
LABEL = os.environ.get("LABEL", "main")
import polysim, strategy_eval as se, experiments as ex, final_strategy as fs   # noqa: E402
pd.set_option("display.width", 230); master, mk, g = ex.master, ex.mk, ex.g
PERS = {"Aug1-19": se.OLD, "tune": se.TUNE, "test": se.TEST} if LABEL == "main" else {LABEL: (master.date.min(), master.date.max())}
def run(d, q, maxpx, stop):
    cfg = {"kind": "D", "decision_offset": d, "cancel": 10, "q": q, "maxpx": maxpx, "tp": None, "sl": stop, "tstop": 300, "settle": stop is None}
    sim = polysim.simulate(mk, g, t_entry=d, entry_mode="maker_last", t_cancel=d + 10, tps=(1.01,), sls=(0.0 if stop is None else stop,), tstops=(300,), slip=0.01)
    m, sim = ex.align(master, sim); idx = np.arange(len(m))
    for sd in (0, 1):     # stop then settle: keep the stop exits (kind 1), settle everything else at 1.00 / 0.00
        k = sim["kind"][idx, sd, 0, 0, 0]
        sim["pnl"][idx, sd, 0, 0, 0] = np.where(k == 1, sim["pnl"][idx, sd, 0, 0, 0], sim["settle"][idx, sd]); sim["kind"][idx, sd, 0, 0, 0] = np.where(k == 1, 1, 2)
    mod = m[f"m2_{d}_close_p"].values; mkt = sim["pre_last"][:, 0]
    side = np.where(mod >= 0.5, 0, 1); pm = np.where(side == 0, mod, 1 - mod); pk = np.where(side == 0, mkt, 1 - mkt)
    sel = np.isfinite(pm) & np.isfinite(pk) & (pm >= q) & (pk <= maxpx)
    pnl = np.where(sel, sim["pnl"][idx, side, 0, 0, 0], np.nan); entry = sim["entry_px"][idx, side]; kind = sim["kind"][idx, side, 0, 0, 0]
    out = []
    for per, (a, b) in PERS.items():
        s = (m.date.values >= a) & (m.date.values <= b); f = s & np.isfinite(pnl)
        if f.sum() < 5: continue
        usd = pd.Series(pnl[f] / entry[f] * 100).groupby(m.date.values[f]).sum(); days = len(set(m.date.values[s]))
        out.append(dict(period=per, q=q, maxpx=maxpx, stop=stop, n=int(f.sum()), entry=round(float(entry[f].mean()), 3), win=round(float((pnl[f] > 0).mean()), 3), stopped=round(float((kind[f] == 1).mean()), 3),
                        cents=round(100 * float(np.nanmean(pnl[f])), 2), usd_per_day=round(float(usd.sum() / days)), worst=round(float(usd.min())), maxdd=round(float((usd.cumsum().cummax() - usd.cumsum()).max()))))
    return out
rows = []
for q, maxpx in ((0.90, 0.90), (0.92, 0.90), (0.95, 0.90), (0.90, 0.85), (0.95, 0.85)):
    for stop in (None, 0.75, 0.70, 0.60, 0.50):
        rows += run(90, q, maxpx, stop)
df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, "..", "reports", f"D_stop_{LABEL}.csv"), index=False)
for per in PERS:
    print(f"\n== {per} ==  (D at T+90; stop = sell at market when our side prints <= stop, else settle)")
    print(df[df.period == per].drop(columns="period").to_string(index=False))
