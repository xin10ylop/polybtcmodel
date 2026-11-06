"""'Wait for the last minute, the direction is clearer': decide at T+180 / 210 / 240 / 260 from the model's close head at that
instant, bid at the last print on the model's side for 10 s (maker), hold to settlement (1.00 / 0.00, no fee).
Cells: model probability for its side >= q while that side still trades <= maxpx (otherwise there is nothing left to earn).
  LABEL=main python research/last_minute.py          (Aug 1-19 / tune / test)
  LABEL=june MASTER_PATH=/home/user/data/master_june.parquet POLY_MARKETS=... POLY_DAYS=... python research/last_minute.py
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "polybtc"))
LABEL = os.environ.get("LABEL", "main")
import polysim, strategy_eval as se, experiments as ex, final_strategy as fs   # noqa: E402
pd.set_option("display.width", 240)
master, mk, g = ex.master, ex.mk, ex.g
PERS = {"Aug1-19": se.OLD, "tune": se.TUNE, "test": se.TEST} if LABEL == "main" else {LABEL: (master.date.min(), master.date.max())}
rows = []
for d in (180, 210, 240, 260):
    for q in (0.75, 0.80, 0.85, 0.90, 0.95):
        for maxpx in (0.80, 0.85, 0.90, 0.95):
            cfg = {"kind": "D", "decision_offset": d, "cancel": 10, "q": q, "maxpx": maxpx, "tp": None, "sl": None, "tstop": None, "settle": True}
            m, sim, p, side, sel, _ = fs.trades_B(cfg, master, mk, g); idx = np.arange(len(m))
            pnl = np.where(sel, sim["pnl"][idx, side, 0, 0, 0], np.nan); entry = sim["entry_px"][idx, side]
            for per, (a, b) in PERS.items():
                s = (m.date.values >= a) & (m.date.values <= b); f = s & np.isfinite(pnl)
                if f.sum() < 5: rows.append(dict(decide=d, q=q, maxpx=maxpx, period=per, n=int(f.sum()), signals=int((sel & s).sum()))); continue
                usd = pd.Series(pnl[f] / entry[f] * 100).groupby(m.date.values[f]).sum(); days = len(set(m.date.values[s]))
                rows.append(dict(decide=d, q=q, maxpx=maxpx, period=per, signals=int((sel & s).sum()), n=int(f.sum()), fill=round(f.sum() / max((sel & s).sum(), 1), 2), avg_entry=round(float(np.nanmean(entry[f])), 3),
                                 win=round(float(np.mean(pnl[f] > 0)), 3), cents=round(100 * float(np.nanmean(pnl[f])), 2), usd_per_day=round(float(usd.sum() / days)), worst_day=round(float(usd.min())), maxdd=round(float((usd.cumsum().cummax() - usd.cumsum()).max()))))
df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, "..", "reports", f"last_minute_{LABEL}.csv"), index=False)
for per in PERS:
    x = df[(df.period == per) & (df.n >= 5)].sort_values("usd_per_day", ascending=False)
    print(f"\n== {per}: top cells by $/day at $100 per trade ==")
    print(x.head(12).to_string(index=False))
print("\n== the running D (T+90, q 0.90, maxpx 0.90) for reference ==")
res, rec = fs.evaluate(fs.CFG_D, verbose=False)
if LABEL == "main": print(res[["n", "win", "mean", "sharpe"]].round(3).to_string())
print("saved reports/last_minute_%s.csv" % LABEL)
