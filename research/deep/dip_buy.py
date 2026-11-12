"""'Buy the dip': the T-120 model picks a side before the open; if that side's shares get down to x cents during the window, buy and hold to settlement.
Three fills (see dip_buy_data.py): thru = resting bid certainly filled (maker, pays x), touch = optimistic maker, take = buy what is on offer (pays price + fee).
Control = the same rule on the side the model did NOT pick: if the model adds nothing, both sides win about as often as the price says.
$100 a trade (shares = 100 / cost).  Periods: Jun 1 - Aug 19, Aug 20 - Sep 30 (tune), Oct 1 - Nov 7 (test).     python research/deep/dip_buy.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from policy_lab import Tix, SC, THR, date, upwin, ok
Z = np.load("/home/user/data/deep/dip_buy_BTC.npz"); XS = Z["XS"]; pos = pd.Series(np.arange(len(Z["W"])), index=Z["W"]).reindex(Tix).values; have = np.isfinite(pos); A = np.full((len(Tix), 2, len(XS), 5), np.nan, np.float32); A[have] = Z["A"][pos[have].astype(int)]
s = SC[-120]; base = ok & np.isfinite(s) & have; side = np.where(s > 0, 0, 1); good = (date.dayofweek >= 5) | (pd.to_datetime(Tix, unit="s").hour >= 16); day = date.values.astype("datetime64[D]")
PER = {"Jun 1-Aug 19": date < "2026-08-20", "Aug 20-Sep 30": (date >= "2026-08-20") & (date < "2026-10-01"), "Oct 1-Nov 7": date >= "2026-10-01"}; rng = np.random.default_rng(0)
def trades(sel, sd, k, how, tmin=0, tmax=270):
    i = np.where(sel)[0]; a = A[i, sd[i], k]; t = a[:, {"thru": 0, "touch": 1, "take": 2}[how]]; f = np.isfinite(t) & (t >= tmin) & (t <= tmax); i = i[f]; a = a[f]
    cost = np.full(len(i), XS[k]) if how != "take" else a[:, 3] + 0.07 * a[:, 3] * (1 - a[:, 3]); win = np.where(sd[i] == 0, upwin[i], ~upwin[i]); return i, cost, win, 100 / cost * win - 100
def line(lbl, sel, sd, k, how, **kw):
    i, cost, win, pnl = trades(sel, sd, k, how, **kw)
    if len(i) < 30: return {"rule": lbl, "trades": len(i)}
    d = pd.Series(pnl).groupby(day[i]).sum().values; bs = d[rng.integers(0, len(d), (2000, len(d)))].sum(1); r = {"rule": lbl, "trades": len(i), "win %": round(100 * win.mean(), 1), "paid ¢": round(100 * cost.mean(), 1), "¢/share": round(100 * (win - cost).mean(), 2), "total $": round(pnl.sum()), "95% lo": round(np.quantile(bs, 0.025)), "95% hi": round(np.quantile(bs, 0.975))}
    for nm, m in PER.items(): r[nm] = round(pnl[m[i]].sum())
    return r
pd.set_option("display.width", 250); T30 = base & (np.abs(s) >= THR[-120, 0.70]); T15 = base & (np.abs(s) >= THR[-120, 0.85])
for how, ttl in (("thru", "RESTING BID at x on the model's side, certainly filled (maker, no fee)"), ("take", "BUY WHAT IS ON OFFER at x or lower (taker, fee included)"), ("touch", "RESTING BID at x, optimistic fill (every touch counts)")):
    print(f"\n=== {ttl} ===\n-- model's side, top 30 % signals, all hours");  print(pd.DataFrame([line(f"x = {int(100*x)}¢", T30, side, k, how) for k, x in enumerate(XS)]).to_string(index=False))
    print("-- CONTROL: the side the model did NOT pick, same windows"); print(pd.DataFrame([line(f"x = {int(100*x)}¢", T30, 1 - side, k, how) for k, x in enumerate(XS)]).to_string(index=False))
    if how == "touch": continue
    print("-- model's side, every signal (no threshold)"); print(pd.DataFrame([line(f"x = {int(100*x)}¢", base, side, k, how) for k, x in enumerate(XS)]).to_string(index=False))
    print("-- model's side, top 15 %"); print(pd.DataFrame([line(f"x = {int(100*x)}¢", T15, side, k, how) for k, x in enumerate(XS)]).to_string(index=False))
    print("-- model's side, top 30 %, trading hours only"); print(pd.DataFrame([line(f"x = {int(100*x)}¢", T30 & good, side, k, how) for k, x in enumerate(XS)]).to_string(index=False))
print("\n=== by WHEN the price first got there (resting bid certainly filled, model's side, top 30 %) ===")
for k in (0, 1, 3, 5):
    print(f"-- x = {int(100*XS[k])}¢"); print(pd.DataFrame([line(f"T+{a}..{b}", T30, side, k, "thru", tmin=a, tmax=b) for a, b in ((0, 30), (30, 60), (60, 120), (120, 180), (180, 240), (240, 270))]).to_string(index=False))
i, _, _, _ = trades(T30, side, 3, "thru"); print("\ndollars sold at 30¢ or lower after the first touch, per window: median", round(float(np.nanmedian(A[i, side[i], 3, 4]))), " 25th pct", round(float(np.nanquantile(A[i, side[i], 3, 4], 0.25))))
