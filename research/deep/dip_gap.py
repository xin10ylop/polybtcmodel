"""Evaluate 'buy the cheap side while Bitcoin is on that side' (data: dip_gap_data.py).  One trade per window and rule, $100 a trade, hold to settlement.
The key column is the LAG: how many seconds before the print's (block) timestamp Bitcoin was read.  A real, tradable effect survives a few seconds of lag.
  python research/deep/dip_gap.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from policy_lab import Tix, SC, THR, date, upwin, ok
Z = np.load("/home/user/data/deep/dip_gap_BTC.npz"); XS, ZM, LG = Z["XS"], Z["ZM"], Z["LG"]; pos = pd.Series(np.arange(len(Z["W"])), index=Z["W"]).reindex(Tix).values; have = np.isfinite(pos)
A = np.full((len(Tix),) + Z["A"].shape[1:], np.nan, np.float32); A[have] = Z["A"][pos[have].astype(int)]; s = SC[-120]; base = ok & np.isfinite(s) & have; side = np.where(s > 0, 0, 1)
T30 = base & (np.abs(s) >= THR[-120, 0.70]); test = np.asarray(date >= "2026-10-01"); day = date.values.astype("datetime64[D]"); rng = np.random.default_rng(0)
def ev(sel, sd, how, k, zi, li):
    i = np.where(sel)[0]; a = A[i, sd[i], how, k, zi, li]; f = np.isfinite(a[:, 0]); i = i[f]; p = a[f, 1].astype(float); cost = p + (0.07 * p * (1 - p) if how == 0 else 0.0); win = np.where(sd[i] == 0, upwin[i], ~upwin[i])
    return pd.DataFrame({"i": i, "cost": cost, "win": win, "pnl": 100 / cost * win - 100, "day": day[i], "test": test[i]})
def row(lbl, e):
    if len(e) < 30: return {"": lbl, "trades": len(e)}
    d = e.groupby("day").pnl.sum().values; b = d[rng.integers(0, len(d), (2000, len(d)))].sum(1); u = e[e.test]
    return {"": lbl, "trades": len(e), "win %": round(100 * e.win.mean(), 1), "paid ¢": round(100 * e.cost.mean(), 1), "¢/share": round(100 * (e.win - e.cost).mean(), 2), "total $": round(e.pnl.sum()), "95% lo": round(np.quantile(b, 0.025)), "95% hi": round(np.quantile(b, 0.975)),
            "Jun-Sep ¢/sh": round(100 * (e.win - e.cost)[~e.test].mean(), 2), "Oct-Nov7 ¢/sh": round(100 * (u.win - u.cost).mean(), 2) if len(u) else np.nan, "Oct-Nov7 $": round(u.pnl.sum())}
pd.set_option("display.width", 250); both = lambda sel, how, k, zi, li: pd.concat([ev(sel, np.zeros(len(Tix), int), how, k, zi, li), ev(sel, np.ones(len(Tix), int), how, k, zi, li)])
for how, ttl in ((0, "BUY WHAT IS ON OFFER (taker, fee included)"), (1, "RESTING BID, certainly filled (maker, no fee)")):
    print(f"\n================ {ttl} ================")
    for k, zi in ((0, 0), (0, 1), (1, 0), (1, 1), (3, 0), (3, 1)):
        print(f"\n-- shares at <= {int(100*XS[k])}¢ while Bitcoin is more than ${int(ZM[zi])} on that side.  By lag of the Bitcoin reading:")
        print("   model's side, top 30 % signals"); print(pd.DataFrame([row(f"lag {L}s", ev(T30, side, how, k, zi, li)) for li, L in enumerate(LG)]).to_string(index=False))
        print("   side the model did NOT pick, same windows"); print(pd.DataFrame([row(f"lag {L}s", ev(T30, 1 - side, how, k, zi, li)) for li, L in enumerate(LG)]).to_string(index=False))
        print("   either side, every window, no model"); print(pd.DataFrame([row(f"lag {L}s", both(ok & have, how, k, zi, li)) for li, L in enumerate(LG)]).to_string(index=False))
