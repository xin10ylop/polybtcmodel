"""Sell-out rule: at X seconds into the window, if Bitcoin has moved Y dollars against our side, sell the shares at the market (taker fee paid).
Every 30 s x every move size, on the simulated fills of the live setup.  $100 a trade (200 shares bought at 0.50), hold-to-settlement as the baseline.
  python research/deep/stoploss_grid.py"""
import numpy as np, pandas as pd
Z = np.load("/home/user/data/deep/stoploss_BTC.npz"); MARKS = Z["marks"]; win = Z["win"]; mv = Z["move"]; d = pd.to_datetime(Z["date"]); W = Z["W"]
sp = np.where(np.isfinite(Z["sell"]), Z["sell"], Z["bid"] - 0.01); sp = np.clip(sp, 0.0, 0.99); net = sp - 0.07 * sp * (1 - sp)      # what a share sells for at each mark, after the taker fee
t = pd.to_datetime(W, unit="s"); good = (t.dayofweek.values >= 5) | (t.hour.values >= 16); SH = 200.0; hold = np.where(win, 0.5, -0.5)
DS = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200]; rng = np.random.default_rng(2); alld = pd.DatetimeIndex(sorted(set(d)))
PER = {"Jun-Aug19": d < "2026-08-20", "Aug20-Sep30": (d >= "2026-08-20") & (d <= "2026-09-30"), "Oct-Nov7": d >= "2026-10-01"}
print(f"{len(win)} simulated fills (all hours), {good.sum()} in the trading hours; win rate {win[good].mean():.1%}; hold to settlement, trading hours: ${hold[good].sum()*SH:,.0f}")
print("\n1. WHAT THE MARKET PAYS vs WHAT THE SHARES ARE WORTH.  Trades where Bitcoin is AGAINST us by this much at this moment (all hours, for sample size)")
print("   For each cell: number of trades | share that still WIN in the end | what one share sells for after the fee | gain (+) or loss (-) per share from selling instead of holding, in cents")
print("   against us by     " + "".join(f"T+{m:<4d}                     " for m in MARKS[[1, 3, 5, 6, 7]]))
B = [(10, 20), (20, 30), (30, 50), (50, 80), (80, 120), (120, 9999)]
for lo, hi in B:
    out = f"   ${lo:>3}-{hi if hi < 9999 else '+':<5}      "
    for j in (1, 3, 5, 6, 7):
        m = (-mv[:, j] >= lo) & (-mv[:, j] < hi) & np.isfinite(net[:, j])
        out += (f"{m.sum():>4d} | {win[m].mean():>4.0%} | {net[m, j].mean()*100:>4.1f}c | {(net[m, j].mean() - win[m].mean())*100:>+5.1f}   " if m.sum() >= 25 else f"{m.sum():>4d} | too few              ")
    print(out)
print("\n2. THE RULE: at this moment, if Bitcoin is against us by at least this much, sell.  Change in total P&L vs holding, trading hours, $100 a trade  (number of trades sold | of those, how many would have won)")
print("   at least $      " + "".join(f"T+{m:<3d}                  " for m in MARKS))
def rule_pnl(j, D, mask):
    trig = (-mv[:, j] >= D) & np.isfinite(net[:, j]) & mask; return np.where(trig, net[:, j] - 0.5, hold), trig
best = []
for D in DS:
    out = f"   ${D:<4}           "
    for j in range(len(MARKS)):
        p, trig = rule_pnl(j, D, good); delta = (p[good].sum() - hold[good].sum()) * SH; out += f"{delta:>+7,.0f} ({trig.sum():>4d}|{int(win[trig].sum()):>3d})   "; best.append((delta, MARKS[j], D, j))
    print(out)
print("\n3. Same rule checked at EVERY 30-second mark from this moment on (sell the first time Bitcoin is against us by at least this much)")
print("   at least $      " + "".join(f"from T+{m:<3d}        " for m in MARKS))
best2 = []
def rule_from(j0, D, mask):
    p = hold.copy(); done = ~mask
    for j in range(j0, len(MARKS)):
        trig = ~done & (-mv[:, j] >= D) & np.isfinite(net[:, j]); p[trig] = net[trig, j] - 0.5; done |= trig
    return p
for D in DS:
    out = f"   ${D:<4}           "
    for j in range(len(MARKS)):
        p = rule_from(j, D, good); delta = (p[good].sum() - hold[good].sum()) * SH; out += f"{delta:>+8,.0f}          "; best2.append((delta, MARKS[j], D, j))
    print(out)
def detail(nm, p):
    dl = (p - hold)[good] * SH; s = pd.Series(dl).groupby(d[good].values).sum().reindex(alld, fill_value=0.0); bs = [s.values[rng.integers(0, len(s), len(s))].sum() for _ in range(2000)]
    tot = pd.Series(p[good] * SH).groupby(d[good].values).sum().reindex(alld, fill_value=0.0); th = pd.Series(hold[good] * SH).groupby(d[good].values).sum().reindex(alld, fill_value=0.0)
    def dd(x): c = np.concatenate([[0], np.cumsum(x)]); return (np.maximum.accumulate(c) - c).max()
    o = np.argsort(W[good]); print(f"   {nm:44s} change ${dl.sum():>+7,.0f} [{np.percentile(bs,2.5):>+7,.0f},{np.percentile(bs,97.5):>+7,.0f}] | by period " + " / ".join(f"{(p - hold)[good & PER[k]].sum()*SH:>+6,.0f}" for k in PER) + f" | worst day ${th.min():,.0f} -> ${tot.min():,.0f} | max drawdown ${dd((hold[good]*SH)[o]):,.0f} -> ${dd((p[good]*SH)[o]):,.0f} | losing days {(th<0).sum()} -> {(tot<0).sum()}")
print("\n4. The best cells, and your example, in detail (trading hours; change vs holding with its 95 % range)")
for delta, m, D, j in sorted(best, reverse=True)[:4]: detail(f"single check at T+{m}, against us by ${D}+", rule_pnl(j, D, good)[0])
for delta, m, D, j in sorted(best2, reverse=True)[:3]: detail(f"every 30 s from T+{m}, against us by ${D}+", rule_from(j, D, good))
detail("YOUR EXAMPLE: at T+180, against us by $50+", rule_pnl(5, 50, good)[0]); detail("at T+180, against us by $80+", rule_pnl(5, 80, good)[0]); detail("at T+240, against us by $50+", rule_pnl(7, 50, good)[0])
print("\n5. Same grid on ALL hours (6,366 trades) as a robustness check: change vs holding, single check")
for D in (30, 50, 80, 100, 150):
    print(f"   ${D:<4}           " + "".join(f"{(rule_pnl(j, D, np.ones(len(win), bool))[0].sum() - hold.sum())*SH:>+8,.0f}   " for j in range(len(MARKS))))
