"""'Buy the dip' with a third condition: how far Bitcoin (Binance spot) is from the window's reference price, in dollars in OUR direction, at the moment
our side's shares get down to x cents (z < 0 = Bitcoin against us).  Everything is tested: price x, move z, time of the dip.
Cells are CHOSEN on Jun 1 - Sep 30 and then judged on Oct 1 - Nov 7 only.  Model's side, top 30 % signals, $100 a trade, hold to settlement.
  python research/deep/dip_buy_z.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); sys.path.insert(0, "/home/user/btcmovement2")
from policy_lab import Tix, SC, THR, date, upwin, ok
from btcmodel.timeline import load_timeline
Z = np.load("/home/user/data/deep/dip_buy_BTC.npz"); XS = Z["XS"]; pos = pd.Series(np.arange(len(Z["W"])), index=Z["W"]).reindex(Tix).values; have = np.isfinite(pos); A = np.full((len(Tix), 2, len(XS), 5), np.nan, np.float32); A[have] = Z["A"][pos[have].astype(int)]
s = SC[-120]; base = ok & np.isfinite(s) & have; side = np.where(s > 0, 0, 1); good = np.asarray((date.dayofweek >= 5) | (pd.to_datetime(Tix, unit="s").hour >= 16)); day = date.values.astype("datetime64[D]")
T30 = base & (np.abs(s) >= THR[-120, 0.70]); test = np.asarray(date >= "2026-10-01"); rng = np.random.default_rng(0)
tl = load_timeline(); sc_ = tl["s_close"]; t0 = int(tl.t0_epoch); TW = pd.Timestamp("2026-09-01").timestamp(); I = np.where(T30)[0]; PX = np.full((len(Tix), 361), np.nan)
for i in I:
    w = int(Tix[i]); a = pd.Series(np.asarray(sc_[w - 60 - t0: w + 301 - t0], dtype=float)).ffill().bfill().values
    if len(a) == 361: PX[i] = (a - (a[:60].mean() if w >= TW else a[59])) * (1 if side[i] == 0 else -1)       # dollars in our direction, index = seconds since T-60
ZB = [(-1e9, -80), (-80, -50), (-50, -30), (-30, -20), (-20, -10), (-10, 0), (0, 1e9)]; ZL = ["< -80", "-80..-50", "-50..-30", "-30..-20", "-20..-10", "-10..0", "> 0"]; TB = [(0, 60), (60, 120), (120, 180), (180, 270)]
def events(how, sel):      # one row per (window, x) where the dip happened
    out = []
    for k, x in enumerate(XS):
        i = np.where(sel)[0]; a = A[i, side[i], k]; t = a[:, {"thru": 0, "take": 2}[how]]; f = np.isfinite(t) & np.isfinite(PX[i, 60]); i = i[f]; a = a[f]; t = t[f]
        cost = np.full(len(i), x) if how == "thru" else a[:, 3] + 0.07 * a[:, 3] * (1 - a[:, 3]); win = np.where(side[i] == 0, upwin[i], ~upwin[i])
        out.append(pd.DataFrame({"i": i, "x": int(round(100 * x)), "t": t, "z": PX[i, 60 + np.minimum(t.astype(int), 299)], "cost": cost, "win": win, "pnl": 100 / cost * win - 100, "day": day[i], "test": test[i]}))
    E = pd.concat(out, ignore_index=True); E["zb"] = pd.cut(E.z, [b[0] for b in ZB] + [1e9], labels=ZL, right=False); E["tb"] = pd.cut(E.t, [0, 60, 120, 180, 271], labels=["T+0-60", "T+60-120", "T+120-180", "T+180-270"], right=False); return E
def ci(e):
    d = e.groupby("day").pnl.sum().values; b = d[rng.integers(0, len(d), (2000, len(d)))].sum(1); return round(np.quantile(b, 0.025)), round(np.quantile(b, 0.975)), (b > 0).mean()
pd.set_option("display.width", 250)
for how, ttl, sel in (("thru", "RESTING BID certainly filled (maker), all hours", T30), ("thru", "RESTING BID certainly filled (maker), trading hours", T30 & good), ("take", "BUY WHAT IS ON OFFER (taker, fee included), all hours", T30)):
    E = events(how, sel); print(f"\n================ {ttl} ================")
    g = E.groupby(["x", "zb"], observed=False).apply(lambda e: f"{100 * (e.win - e.cost).mean():+.1f} ({len(e)})" if len(e) >= 40 else "-").unstack().sort_index(ascending=False)
    print("cents per share (number of trades), whole period.  Rows = price paid, columns = Bitcoin's move in our direction ($) when the price got there"); print(g.to_string())
    for dims, lbl in ((["x", "zb"], "price x move"), (["x", "zb", "tb"], "price x move x time")):
        pick = []
        for key, e in E[~E.test].groupby(dims, observed=True):
            if len(e) >= 150 and e.pnl.sum() > 0 and ci(e)[2] >= 0.90: pick.append(key)
        print(f"\n-- {lbl}: {E[~E.test].groupby(dims, observed=True).ngroups} cells tested, {len(pick)} look profitable on Jun 1 - Sep 30 (at least 150 trades, 90 % of resamples positive)")
        if not pick: continue
        key = E[dims].astype(str).agg("|".join, axis=1); ch = key.isin(["|".join(map(str, p)) for p in pick]); r = []
        for nm, m in (("Jun 1 - Sep 30 (chosen here)", ~E.test), ("Oct 1 - Nov 7 (unseen)", E.test)):
            e = E[ch & m]; lo, hi, p = ci(e); r.append({"period": nm, "trades": len(e), "win %": round(100 * e.win.mean(), 1), "paid ¢": round(100 * e.cost.mean(), 1), "¢/share": round(100 * (e.win - e.cost).mean(), 2), "total $": round(e.pnl.sum()), "95% lo": lo, "95% hi": hi})
        print(pd.DataFrame(r).to_string(index=False)); print("   chosen cells:", "; ".join(" / ".join(map(str, p)) for p in pick[:12]) + (" ..." if len(pick) > 12 else ""))
        e = E[ch & E.test]; print("   unseen period, each chosen cell: " + "; ".join(f"{k}: {round(v)}" for k, v in e.groupby(key[ch & E.test]).pnl.sum().items()))
    # does the Bitcoin move add anything once the price is known?  win rate minus price, by move, pooled over all price levels
    q = E.groupby("zb", observed=False).apply(lambda e: pd.Series({"trades": len(e), "win % minus price paid": round(100 * (e.win - e.cost).mean(), 2), "unseen period": round(100 * (e.win - e.cost)[e.test].mean(), 2)})); print("\npooled over all prices:"); print(q.T.to_string())
