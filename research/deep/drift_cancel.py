"""Same family, applied BEFORE the fill: while the 0.50 bid rests (from T-118), cancel it if Bitcoin has moved Y dollars against our side
since the decision.  Fills that arrive after an adverse move are the ones that lose.  Also: the favourable side of the in-window calibration.
  python research/deep/drift_cancel.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); sys.path.insert(0, "/home/user/btcmovement2")
from btcmodel.timeline import load_timeline
Z = np.load("/home/user/data/deep/stoploss_BTC.npz"); W = Z["W"]; side = Z["side"]; win = Z["win"]; ft = Z["ft"]; d = pd.to_datetime(Z["date"]); mv = Z["move"]; MARKS = list(Z["marks"])
sp = np.clip(np.where(np.isfinite(Z["sell"]), Z["sell"], Z["bid"] - 0.01), 0.0, 0.99); net = sp - 0.07 * sp * (1 - sp)
atl = load_timeline(); sc_ = atl["s_close"]; t0 = int(atl.t0_epoch); n = len(W); drift_fill = np.full(n, np.nan); worst = np.full(n, np.nan); d30 = np.full(n, np.nan)
for k in range(n):
    w = int(W[k]); f = int(np.floor(ft[k])); a = pd.Series(np.asarray(sc_[w - 121 - t0: w + f + 1 - t0], dtype=float)).ffill().bfill().values; sg = 1.0 if side[k] == 0 else -1.0
    path = sg * (a - a[0]); drift_fill[k] = path[-1]; worst[k] = path.min(); d30[k] = sg * (a[-1] - a[max(len(a) - 31, 0)])       # a[0] = the price at the decision (T-121)
t = pd.to_datetime(W, unit="s"); good = (t.dayofweek.values >= 5) | (t.hour.values >= 16); SH = 200.0; hold = np.where(win, 0.5, -0.5); alld = pd.DatetimeIndex(sorted(set(d))); rng = np.random.default_rng(8)
PER = {"Jun-Aug19": d < "2026-08-20", "Aug20-Sep30": (d >= "2026-08-20") & (d <= "2026-09-30"), "Oct-Nov7": d >= "2026-10-01"}
print(f"{n} simulated fills ({good.sum()} in the trading hours). Bitcoin's move in our direction between the decision and the moment the bid fills: median ${np.median(drift_fill):+.1f}, 10th pct ${np.percentile(drift_fill,10):+.1f}, 90th ${np.percentile(drift_fill,90):+.1f}")
print("\n1. Win rate of the fill by where Bitcoin is when the bid fills, relative to the decision (all hours | trading hours)")
for lo, hi in ((-9999, -60), (-60, -40), (-40, -25), (-25, -15), (-15, -8), (-8, 0), (0, 8), (8, 15), (15, 25), (25, 40), (40, 9999)):
    m = (drift_fill >= lo) & (drift_fill < hi); g = m & good
    print(f"   {('below' if lo < -9000 else f'${lo:+d}'):>6} .. {('above' if hi > 9000 else f'${hi:+d}'):<6}  all: n {m.sum():>4d}, win {win[m].mean():.1%}, P&L ${hold[m].sum()*SH:>+8,.0f}   |  trading hours: n {g.sum():>4d}, win {win[g].mean() if g.sum() else float('nan'):.1%}, P&L ${hold[g].sum()*SH:>+8,.0f}")
print("\n2. RULE: cancel the resting bid as soon as Bitcoin is Y dollars against us since the decision (so those fills never happen). Trading hours, $100 a trade")
print("   cancel at   trades kept   win    total P&L   change vs no rule [95 % range]      by period (change)            worst day   max drawdown   losing days")
def dd(x): c = np.concatenate([[0], np.cumsum(x)]); return (np.maximum.accumulate(c) - c).max()
def line(nm, keep, mask=good):
    k = keep & mask; p = np.where(k, hold, 0.0) * SH; base = np.where(mask, hold, 0.0) * SH; dl = p - base
    s = pd.Series(dl).groupby(d.values).sum().reindex(alld, fill_value=0.0); bs = [s.values[rng.integers(0, len(s), len(s))].sum() for _ in range(2000)]; day = pd.Series(p).groupby(d.values).sum().reindex(alld, fill_value=0.0); o = np.argsort(W)
    print(f"   {nm:10s}  {k.sum():>5d}       {win[k].mean():.1%}   ${p.sum():>8,.0f}   ${dl.sum():>+7,.0f} [{np.percentile(bs,2.5):>+7,.0f},{np.percentile(bs,97.5):>+7,.0f}]     " + " / ".join(f"{dl[PER[q]].sum():>+6,.0f}" for q in PER) + f"      ${day.min():>6,.0f}     ${dd(p[o]):>6,.0f}        {(day < 0).sum()}")
line("no rule", np.ones(n, bool))
for Y in (5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80): line(f"-${Y}", worst > -Y)
print("   the same on ALL hours:")
line("no rule", np.ones(n, bool), np.ones(n, bool))
for Y in (10, 15, 20, 30, 40, 60): line(f"-${Y}", worst > -Y, np.ones(n, bool))
print("\n3. Variant: cancel only if Bitcoin fell Y dollars against us within the LAST 30 seconds before the fill (a fast move), trading hours")
line("no rule", np.ones(n, bool))
for Y in (5, 10, 15, 20, 30, 40): line(f"-${Y}/30s", d30 > -Y)
print("\n4. In-window calibration on the FAVOURABLE side (all hours): Bitcoin in OUR favour by this much at this moment -> share that win | what the share sells for after the fee | sell minus hold")
for lo, hi in ((10, 20), (20, 30), (30, 50), (50, 80), (80, 120), (120, 9999)):
    out = f"   ${lo:>3}-{hi if hi < 9999 else '+':<5} "
    for e in (60, 120, 180, 240):
        j = MARKS.index(e); m = (mv[:, j] >= lo) & (mv[:, j] < hi) & np.isfinite(net[:, j]); out += f"| T+{e}: n {m.sum():>4d}, win {win[m].mean():.0%}, sells {net[m, j].mean()*100:>4.1f}c, {(net[m, j].mean() - win[m].mean())*100:>+5.1f}c "
    print(out)
