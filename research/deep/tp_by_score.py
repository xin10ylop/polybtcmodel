"""Take-profit or hold, by signal strength.  Live setup (trading hours, queue-aware fills): for each strength group, the profit per share of holding to
settlement against a resting sell at 0.70 .. 0.95 (maker, no fee).  'through' = the sell counts only when the price trades ABOVE it (certain fill);
'touch' = it counts as soon as the price reaches it (optimistic).  Also split by what the T-45 model said.     python research/deep/tp_by_score.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from policy_lab import run, SC, THR, Tix, date
good = np.asarray((date.dayofweek >= 5) | (pd.to_datetime(Tix, unit="s").hour >= 16)); TPS = (None, 0.70, 0.80, 0.85, 0.90, 0.95); R = {}
for tp in TPS:
    for thr_ in ((True, False) if tp else (True,)):
        r = run(dec=-120, tier=0.70, confirm=dict(cancel=True), tp=tp, cross="skip", fresh=5, queue=50.0, tp_through=thr_); m = good[r["i"]]; R[tp, thr_] = pd.Series(r["v"][m], index=r["i"][m]); 
        if tp is None: q = pd.Series(r["rel_score"][m], index=r["i"][m])
I = R[None, True].index.values; hold = R[None, True].values; per = np.where(date[I] < "2026-08-20", 0, np.where(date[I] < "2026-10-01", 1, 2)); day = date.values.astype("datetime64[D]")[I]; rng = np.random.default_rng(9)
s45 = SC[-45][I]; s120 = SC[-120][I]; ag = np.isfinite(s45) & ((s45 > 0) == (s120 > 0)); a45 = np.abs(s45) / THR[-45, 0.70][I]
G = [("all trades", np.ones(len(I), bool)), ("strength 1.00-1.35x", (q.values < 1.35)), ("strength 1.35-1.7x", (q.values >= 1.35) & (q.values < 1.7)), ("strength 1.7-2.2x", (q.values >= 1.7) & (q.values < 2.2)), ("strength 2.2x + (doubled)", q.values >= 2.2),
     ("T-45 agrees strongly", ag & (a45 >= 1)), ("T-45 agrees weakly", ag & (a45 < 1)), ("T-45 disagrees", ~ag)]
pd.set_option("display.width", 250)
for mode, lbl in ((True, "SELL COUNTS ONLY WHEN THE PRICE TRADES THROUGH IT (certain)"), (False, "SELL COUNTS ON A TOUCH (optimistic)")):
    rows = []
    for nm, k in G:
        row = {"group": nm, "trades": int(k.sum()), "win %": round(100 * (hold[k] > 0).mean(), 1), "hold ¢/share": round(100 * hold[k].mean(), 2)}
        for tp in TPS[1:]:
            v = R[tp, mode].reindex(I).values; dlt = (v - hold)[k]; g = pd.Series(dlt).groupby(day[k]).sum().values; b = g[rng.integers(0, len(g), (1000, len(g)))].sum(1) / k.sum()
            row[f"TP {tp:.2f} vs hold"] = f"{100*dlt.mean():+.2f} [{100*np.quantile(b,.025):+.1f},{100*np.quantile(b,.975):+.1f}]"
        rows.append(row)
    print("\n" + lbl + ": change in cents per share against holding [95 % range]; + = the take-profit is better"); print(pd.DataFrame(rows).to_string(index=False))
v85 = R[0.85, True].reindex(I).values; print("\nTP 0.85 (certain fill) against holding, by period, cents per share: " + "; ".join(f"{nm}: " + " / ".join(f"{100*(v85 - hold)[k & (per == p)].mean():+.2f}" for p in (0, 1, 2)) for nm, k in G[:5]))
h = R[0.85, True].reindex(I).values != hold; print(f"how often the price trades through 0.85 after our fill: {100*h.mean():.0f} % of trades; of those, {100*(hold[h] > 0).mean():.0f} % go on to win (so selling at 0.85 gives up {100*(hold[h] > 0).mean() - 85:.0f} points)")
