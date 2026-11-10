"""Combinations of the levers that helped in policy_grid.py (more signals, cancel when the T-45 model disagrees, higher sell), with day-bootstrap
ranges, drawdown and the monthly table of the chosen one.  python research/deep/policy_combo.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from policy_lab import run, summ, TABLE, PER, date, mon, MONTHS, STAKE, SC, THR, ok, S85
C = dict(cancel=True); K = {
 "K0 top 15 %, sell 0.85 (T-120 baseline)": dict(),
 "K1 top 15 %, cancel on disagreement, sell 0.90": dict(confirm=C, tp=0.90),
 "K2 top 30 %, cancel on disagreement, sell 0.90": dict(tier=0.70, confirm=C, tp=0.90),
 "K3 top 30 %, cancel on disagreement, settlement": dict(tier=0.70, confirm=C, tp=None),
 "K4 top 50 %, cancel on disagreement, sell 0.90": dict(tier=0.50, confirm=C, tp=0.90),
 "K5 top 50 %, cancel on disagreement, settlement": dict(tier=0.50, confirm=C, tp=None),
 "K6 top 30 %, no cancel, sell 0.90": dict(tier=0.70, tp=0.90),
 "K7 top 30 %, cancel, bid 0.51 from T-45, sell 0.90": dict(tier=0.70, confirm=C, tp=0.90, stages=((-118, -45, 0.50), (-45, -15, 0.51)))}
rng = np.random.default_rng(9); days = pd.DatetimeIndex(sorted(set(date))); R = {}
def dd(x): e = np.concatenate([[0], np.cumsum(x)]); return (np.maximum.accumulate(e) - e).max()
print(f"backtest table ${TABLE:,.0f} ($418/day).  $100 a trade.\n")
print(f"{'policy':52s} trades/day  win    c/share |  full: total [95% range]            /day  %table  maxDD |  real: total [95% range]          /day  %table  maxDD | full by period: Jun-Aug19 / tune / TEST | real by period")
for k, kw in K.items():
    r = run(**kw); R[k] = r; full = r["v"] * STAKE / r["e"]; real = r["v"] * r["cap"] / r["e"]; d = pd.to_datetime(date[r["i"]]); out = f"{k:52s} {len(full)/160:>6.1f}    {(r['v'] > 0).mean():.1%}  {r['v'].mean()*100:+.2f}  |"
    for x in (full, real):
        s = pd.Series(x).groupby(d.values).sum().reindex(days, fill_value=0.0).values; bs = [s[rng.integers(0, len(s), len(s))].sum() for _ in range(2000)]
        out += f"  ${x.sum():>7,.0f} [{np.percentile(bs,2.5):>7,.0f}, {np.percentile(bs,97.5):>7,.0f}]  {x.sum()/160:>5.0f}   {x.sum()/TABLE:>4.0%}  {dd(x):>6,.0f} |"
    p = {q: summ(r, q) for q in ("Jun1-Aug19", "tune", "TEST")}
    print(out + f"  {p['Jun1-Aug19']['full']:>7,.0f} / {p['tune']['full']:>6,.0f} / {p['TEST']['full']:>6,.0f} | {p['Jun1-Aug19']['real']:>6,.0f} / {p['tune']['real']:>6,.0f} / {p['TEST']['real']:>6,.0f}")
base_sel = ok & np.isfinite(SC[-45]) & (np.abs(SC[-45]) >= THR[-45, 0.85]); bp = np.where(SC[-45] > 0, S85.pnl_up.values, S85.pnl_dn.values)
for k in ("K2 top 30 %, cancel on disagreement, sell 0.90", "K4 top 50 %, cancel on disagreement, sell 0.90"):
    r = R[k]; full = r["v"] * STAKE / r["e"]; real = r["v"] * r["cap"] / r["e"]; m_ = mon[r["i"]]
    print(f"\n{k}\nmonth     | backtest table: trades / win / P&L   | this setup: trades / win / P&L full $100 / P&L real size")
    for mo in MONTHS + ["total"]:
        a = base_sel if mo == "total" else (base_sel & (mon == mo)); b = np.ones(len(full), bool) if mo == "total" else (m_ == mo)
        print(f"{('Nov 1-7' if mo == 'Nov' else mo):9s} | {a.sum():>6,d} / {(bp[a] > 0).mean():.1%} / ${bp[a].sum()*200:>8,.0f}        | {b.sum():>6,d} / {(r['v'][b] > 0).mean():.1%} / ${full[b].sum():>8,.0f} / ${real[b].sum():>8,.0f}")
