"""Robustness of 'buy at the offer before the open' on the 15-minute BTC market (trading hours, top 30 % T-120 signals, hold to settlement, fee included).
The entry price is not observed directly, only what other buyers paid, so it is varied: first print, first print + 1 cent, average of all buyers at or below the cap;
and the buying window: within 30 s of the decision, or any time until T-15.  Also: only when the T-45 model agrees (then buy from T-43).
  python research/deep/pm15_taker.py"""
import sys, io, contextlib, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
with contextlib.redirect_stdout(io.StringIO()): import pm15_hist as H
T, P, s120, s45, thr, mon, good, up, day, ND, rng = H.T, H.P, H.s120, H.s45, H.thr, H.mon, H.good, H.up, H.day, H.ND, H.rng
def run(cap, t0, t1, price, confirm=False, hours=True):
    o = []
    for j, w in enumerate(T):
        s = s120[w]
        if abs(s) < thr[mon[j]] or (hours and not good[j]): continue
        if confirm and not (w in s45.index and np.isfinite(s45[w]) and (s45[w] > 0) == (s > 0)): continue
        sd = 0 if s > 0 else 1; won = bool(up[w]) if sd == 0 else not bool(up[w]); rel, pu, lu, usd = P.get(int(w), (np.zeros(0), np.zeros(0), np.zeros(0, bool), np.zeros(0))); po = pu if sd == 0 else 1 - pu; tb = lu if sd == 0 else ~lu
        k = np.flatnonzero(tb & (rel >= t0) & (rel <= t1) & (po <= cap + 1e-6))
        if not len(k): continue
        p = float(po[k[0]]) if price == "first" else float(po[k[0]]) + 0.01 if price == "first+1c" else float((po[k] * usd[k]).sum() / usd[k].sum()); o.append((j, won, p + 0.07 * p * (1 - p), float(usd[k].sum())))
    o = pd.DataFrame(o, columns=["j", "won", "cost", "usd"]); jj = o.j.values; e = (o.won - o.cost).values; te = day[jj] >= np.datetime64("2026-10-01"); tu = (day[jj] >= np.datetime64("2026-08-20")) & ~te; out = {}
    for S in (20, 50, 100):
        real = np.minimum(S, o.usd.values) / o.cost.values * e; g = pd.Series(real).groupby(day[jj]).sum(); bs = g.values[rng.integers(0, len(g), (2000, len(g)))].sum(1); c = np.cumsum(real); out[f"${S}: total"] = round(real.sum()); out[f"${S}: 95% lo"] = round(np.quantile(bs, .025))
        if S == 100: out["$100: max DD"] = round(float((np.maximum.accumulate(np.maximum(c, 0)) - c).max())); out["$100/day"] = round(real.sum() / ND, 1)
    return {"trades/day": round(len(o) / ND, 1), "win %": round(100 * o.won.mean(), 1), "paid ¢": round(100 * o.cost.mean(), 1), "¢/share": round(100 * e.mean(), 2), "Jun1-Aug19": round(100 * e[~te & ~tu].mean(), 2), "Aug20-Sep30": round(100 * e[tu].mean(), 2), "Oct1-Nov7": round(100 * e[te].mean(), 2), "median $ offered": round(o.usd.median()), **out}
pd.set_option("display.width", 250); rows = []
for lbl, kw in (("any time T-118..T-15, first print", dict(t0=-118, t1=-15, price="first")), ("any time, first print + 1¢", dict(t0=-118, t1=-15, price="first+1c")), ("any time, average paid by all buyers", dict(t0=-118, t1=-15, price="vwap")),
                ("within 30 s of the decision, first print", dict(t0=-118, t1=-88, price="first")), ("within 30 s, first print + 1¢", dict(t0=-118, t1=-88, price="first+1c")), ("within 30 s, average paid", dict(t0=-118, t1=-88, price="vwap")),
                ("T-45 model agrees, buy T-43..T-15, first print", dict(t0=-43, t1=-15, price="first", confirm=True)), ("T-45 agrees, first print + 1¢", dict(t0=-43, t1=-15, price="first+1c", confirm=True)), ("T-45 agrees, average paid", dict(t0=-43, t1=-15, price="vwap", confirm=True))):
    for cap in (0.53, 0.55, 0.58):
        rows.append({"entry": lbl, "cap": cap, **run(cap, **kw)})
print(pd.DataFrame(rows).to_string(index=False))
print("\nall hours, any time, first print, cap 0.55:", run(0.55, -118, -15, "first", hours=False))
