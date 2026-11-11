"""In-window exits that use information the share price may not contain.  For every simulated fill of the live setup, at each 30-s mark:
  m = what the market pays for our share (after the taker fee),  q = the close model's updated probability that our side wins (m2_<e>_close_p),
  plus the recent direction of Bitcoin and the distance to the reference in units of the remaining volatility.
1. does q predict the outcome beyond the price?   2. sell when the market pays more than the model thinks the share is worth (m - q >= d)
3. momentum and volatility-scaled distance.          $100 a trade = 200 shares.     python research/deep/stoploss_model.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); sys.path.insert(0, "/home/user/btcmovement2")
from common import load
from btcmodel.timeline import load_timeline
Z = np.load("/home/user/data/deep/stoploss_BTC.npz"); W = Z["W"]; side = Z["side"]; win = Z["win"]; mv = Z["move"]; d = pd.to_datetime(Z["date"]); MARKS = list(Z["marks"]); ref = Z["ref"]
sp = np.clip(np.where(np.isfinite(Z["sell"]), Z["sell"], Z["bid"] - 0.01), 0.0, 0.99); net = sp - 0.07 * sp * (1 - sp); mid = np.clip(Z["last"], 0.005, 0.995)
E = [30, 60, 90, 120, 150, 180, 210, 240]; M = load("BTC", [f"m2_{e}_close_p" for e in E] + [f"m2_{e}_lrv_300" for e in E]).reindex(W)
q = np.column_stack([np.where(side == 0, M[f"m2_{e}_close_p"].values, 1 - M[f"m2_{e}_close_p"].values) for e in E]); J = [MARKS.index(e) for e in E]
t = pd.to_datetime(W, unit="s"); good = (t.dayofweek.values >= 5) | (t.hour.values >= 16); SH = 200.0; hold = np.where(win, 0.5, -0.5); alld = pd.DatetimeIndex(sorted(set(d))); rng = np.random.default_rng(4)
PER = {"Jun-Aug19": d < "2026-08-20", "Aug20-Sep30": (d >= "2026-08-20") & (d <= "2026-09-30"), "Oct-Nov7": d >= "2026-10-01"}
print(f"{len(W)} fills ({good.sum()} in the trading hours); model forecast available: {np.isfinite(q).mean():.1%}")
print("\n1. DOES THE MODEL KNOW MORE THAN THE PRICE?  All fills. For each moment: trades grouped by (model's probability for our side) minus (market price of our share).")
print("   Each cell: number of trades | market price | share that actually won | won minus price (positive = the share was worth MORE than the market paid)")
print("   model - market    " + "".join(f"T+{e:<4d}                        " for e in (60, 120, 180, 240)))
for lo, hi in ((-1, -0.15), (-0.15, -0.07), (-0.07, -0.03), (-0.03, 0.03), (0.03, 0.07), (0.07, 0.15), (0.15, 1)):
    out = f"   {lo:+.2f} .. {hi:+.2f}     "
    for e in (60, 120, 180, 240):
        k = E.index(e); j = J[k]; dd = q[:, k] - mid[:, j]; m = (dd >= lo) & (dd < hi) & np.isfinite(dd)
        out += (f"{m.sum():>4d} | {mid[m, j].mean():.2f} | {win[m].mean():.2f} | {(win[m].mean() - mid[m, j].mean())*100:>+5.1f}c     " if m.sum() >= 30 else f"{m.sum():>4d} | too few                  ")
    print(out)
from numpy.linalg import lstsq
print("   Brier score (lower = better forecast of the outcome):   " + "   ".join(f"T+{e}: market {np.nanmean((mid[:, J[k]] - win)**2):.4f}, model {np.nanmean((q[:, k] - win)**2):.4f}, 50/50 blend {np.nanmean(((mid[:, J[k]] + q[:, k])/2 - win)**2):.4f}" for k, e in enumerate(E) if e in (60, 120, 180, 240)))
def detail(nm, p, mask=good):
    dl = (p - hold)[mask] * SH; s = pd.Series(dl).groupby(d[mask].values).sum().reindex(alld, fill_value=0.0); bs = [s.values[rng.integers(0, len(s), len(s))].sum() for _ in range(2000)]
    tot = pd.Series(p[mask] * SH).groupby(d[mask].values).sum().reindex(alld, fill_value=0.0); th = pd.Series(hold[mask] * SH).groupby(d[mask].values).sum().reindex(alld, fill_value=0.0)
    def ddn(x): c = np.concatenate([[0], np.cumsum(x)]); return (np.maximum.accumulate(c) - c).max()
    o = np.argsort(W[mask]); print(f"   {nm:64s} ${dl.sum():>+7,.0f} [{np.percentile(bs,2.5):>+7,.0f},{np.percentile(bs,97.5):>+7,.0f}] | periods " + " / ".join(f"{(p - hold)[mask & PER[k]].sum()*SH:>+6,.0f}" for k in PER) + f" | worst day {th.min():,.0f} -> {tot.min():,.0f} | drawdown {ddn((hold[mask]*SH)[o]):,.0f} -> {ddn((p[mask]*SH)[o]):,.0f}")
print("\n2. RULE: at this moment, sell if the market pays at least d more than the model thinks our share is worth.  Change in total P&L vs holding, trading hours, $100 a trade (trades sold | of those that would have won)")
print("   market - model >=  " + "".join(f"T+{e:<3d}                 " for e in E))
res = []
def rule(k, dlt, only_losing=False):
    j = J[k]; trig = (net[:, j] - q[:, k] >= dlt) & np.isfinite(q[:, k]) & np.isfinite(net[:, j]) & (net[:, j] > 0.01)
    if only_losing: trig &= mv[:, j] < 0
    return np.where(trig, net[:, j] - 0.5, hold), trig
for dlt in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20):
    out = f"   {dlt:.2f}               "
    for k in range(len(E)):
        p, trig = rule(k, dlt); tg = trig & good; dl = (p[good].sum() - hold[good].sum()) * SH; out += f"{dl:>+7,.0f} ({tg.sum():>4d}|{int(win[tg].sum()):>3d})  "; res.append((dl, E[k], dlt, k))
    print(out)
print("\n   same, checked at every mark from this moment on (sell the first time the condition holds)")
res2 = []
def rule_from(k0, dlt):
    p = hold.copy(); done = np.zeros(len(win), bool)
    for k in range(k0, len(E)):
        j = J[k]; trig = ~done & (net[:, j] - q[:, k] >= dlt) & np.isfinite(q[:, k]) & np.isfinite(net[:, j]) & (net[:, j] > 0.01); p[trig] = net[trig, j] - 0.5; done |= trig
    return p
for dlt in (0.04, 0.06, 0.08, 0.10, 0.15, 0.20):
    out = f"   {dlt:.2f}               "
    for k in range(len(E)):
        p = rule_from(k, dlt); dl = (p[good].sum() - hold[good].sum()) * SH; out += f"{dl:>+8,.0f}             "; res2.append((dl, E[k], dlt, k))
    print(out)
print("\n   best cells in detail (change vs holding [95 % range], by period, worst day, drawdown):")
for dl, e, dlt, k in sorted(res, reverse=True)[:5]: detail(f"single check at T+{e}: sell if market - model >= {dlt:.2f}", rule(k, dlt)[0])
for dl, e, dlt, k in sorted(res2, reverse=True)[:3]: detail(f"every mark from T+{e}: sell if market - model >= {dlt:.2f}", rule_from(k, dlt))
print("   the same best single-check cells on ALL hours:")
for dl, e, dlt, k in sorted(res, reverse=True)[:3]: detail(f"single check at T+{e}: sell if market - model >= {dlt:.2f}", rule(k, dlt)[0], np.ones(len(win), bool))
# 3. momentum and volatility-scaled distance
atl = load_timeline(); sc_ = atl["s_close"]; t0 = int(atl.t0_epoch)
print("\n3. MOMENTUM: trades that are against us by $30-100 at this moment, split by what Bitcoin did in the previous 30 s (in our direction or against). Share that still win | what the share sells for | sell minus hold per share")
for e in (120, 150, 180, 210):
    j = MARKS.index(e); r30 = np.array([(float(sc_[int(w) + e - 1 - t0]) - float(sc_[int(w) + e - 31 - t0])) * (1 if s == 0 else -1) for w, s in zip(W, side)]); b = (-mv[:, j] >= 30) & (-mv[:, j] < 100) & np.isfinite(net[:, j])
    print(f"   T+{e}: " + " | ".join(f"{nm}: n {(b & m).sum()}, win {win[b & m].mean():.1%}, sells {net[b & m, j].mean()*100:.1f}c, {(net[b & m, j].mean() - win[b & m].mean())*100:+.1f}c" for nm, m in (("still moving against us (30 s)", r30 < -5), ("flat", np.abs(r30) <= 5), ("bouncing our way", r30 > 5))))
