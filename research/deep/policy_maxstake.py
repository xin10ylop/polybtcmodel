"""How large a stake is still dependable for the chosen BTC setup (T-120, top 30 %, T-45 confirmation, weekends + weekday 16-24 UTC, settle).
Profit is split into TRANCHES of stake (the first $50, the next $50, ...): a tranche is worth adding only if its own profit is clearly positive
- across queue assumptions, across the share of the post-queue volume that reaches us, and in each of the three periods.
  python research/deep/policy_maxstake.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from policy_lab import run, date, Tix, MONTHS
ND = 160; rng = np.random.default_rng(12); alld = pd.DatetimeIndex(sorted(set(date)))
PER = {"Jun-Aug19": lambda d: d < pd.Timestamp("2026-08-20"), "Aug20-Sep30": lambda d: (d >= pd.Timestamp("2026-08-20")) & (d <= pd.Timestamp("2026-09-30")), "Oct-Nov7": lambda d: d >= pd.Timestamp("2026-10-01")}
TR = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 750), (750, 1000)]
def data(Q):
    r = run(dec=-120, tier=0.70, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=Q); t = pd.to_datetime(Tix[r["i"]], unit="s"); k = (t.dayofweek.values >= 5) | (t.hour.values >= 16)
    o = np.argsort(Tix[r["i"]][k]); return r["v"][k][o], r["e"][k][o], r["vol_after"][k][o], pd.to_datetime(date[r["i"]][k][o])
def dd(x): c = np.concatenate([[0], np.cumsum(x)]); return (np.maximum.accumulate(c) - c).max()
D = {Q: data(Q) for Q in (25.0, 50.0, 100.0)}
for Q in (25.0, 50.0, 100.0):
    v, e, va, d = D[Q]
    for phi in (1.0, 0.5):
        av = va * phi; print(f"\n=== queue ahead ${Q:.0f}, share of the volume after the queue that reaches us {phi:.0%} ===  ({len(v)/ND:.1f} trades/day; dollars we could get per trade: median ${np.median(av):.0f}, 75th pct ${np.percentile(av,75):.0f})")
        print("  tranche of stake   $/day from this tranche [95% range]    c per $1 filled   by period ($/day)             cumulative $/day   share of tranche filled")
        cum = 0.0
        for lo, hi in TR:
            f = np.minimum(hi, av) - np.minimum(lo, av); pnl = v * f / e; s = pd.Series(pnl).groupby(d.values).sum().reindex(alld, fill_value=0.0); bs = [s.values[rng.integers(0, len(s), len(s))].sum() / ND for _ in range(1500)]
            per = " / ".join(f"{pnl[fn(d)].sum() / s.index[fn(s.index)].size:>6.1f}" for fn in PER.values()); cum += pnl.sum() / ND
            flag = "OK " if np.percentile(bs, 2.5) > 0 and all(pnl[fn(d)].sum() > 0 for fn in PER.values()) else "-- "
            print(f"  {flag}${lo:>4}-{hi:<5}      {pnl.sum()/ND:>7.1f} [{np.percentile(bs,2.5):>6.1f}, {np.percentile(bs,97.5):>6.1f}]              {pnl.sum()/max(f.sum(),1e-9)*100:>+6.2f}        {per}        {cum:>7.1f}            {f.sum()/((hi-lo)*len(f)):.0%}")
v, e, va, d = D[50.0]; mo_ = d.strftime("%b")
for S, phi in ((300.0, 1.0), (300.0, 0.5)):
    real = v * np.minimum(S, va * phi) / e; day = pd.Series(real).groupby(d.values).sum().reindex(alld, fill_value=0.0); bs = [day.values[rng.integers(0, len(day), len(day))].sum() for _ in range(2000)]
    print(f"\n$300 a trade, real size, queue $50, {phi:.0%} of the post-queue volume reaches us: average filled ${np.minimum(S, va*phi).mean():.0f}; total ${real.sum():,.0f} [{np.percentile(bs,2.5):,.0f}, {np.percentile(bs,97.5):,.0f}], ${real.sum()/ND:.0f}/day, max drawdown ${dd(real):,.0f}")
    print("month     | trades | win   | P&L      | winning | losing | flat | worst day | best day | max drawdown | avg filled")
    for m in MONTHS + ["total"]:
        md = np.ones(len(alld), bool) if m == "total" else (alld.strftime("%b") == m); mt = np.ones(len(v), bool) if m == "total" else (mo_ == m); x = day[md]
        print(f"{('Nov 1-7' if m=='Nov' else m):9s} | {mt.sum():>6d} | {(v[mt]>0).mean():.1%} | ${x.sum():>7,.0f} | {(x>0).sum():>7d} | {(x<0).sum():>6d} | {(x==0).sum():>4d} | ${x.min():>7,.0f}  | ${x.max():>6,.0f}  | ${dd(real[mt]):>6,.0f}       | ${np.minimum(S, va[mt]*phi).mean():.0f}")
print("\nstake    total real size (queue $50, 100% share)   max drawdown   total / drawdown   | with 50% share: total   max drawdown")
for S in (50, 100, 150, 200, 300, 400, 500, 750, 1000):
    a = v * np.minimum(S, va) / e; b = v * np.minimum(S, va * 0.5) / e; print(f"  ${S:<5}   ${a.sum():>8,.0f}                                 ${dd(a):>6,.0f}         {a.sum()/dd(a):>5.1f}            |   ${b.sum():>8,.0f}          ${dd(b):>6,.0f}")
