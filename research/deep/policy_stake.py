"""How much stake the market really fills, for the chosen setups.  For every filled trade: the dollars that traded at our price while the
bid rested, in total and after the queue assumed ahead of us.  P&L by stake = sum of (edge of the trade) x min(stake, dollars we could get).
  python research/deep/policy_stake.py                      (BTC: T-120, top 30 %, confirmation, good hours, settle)
  ASSET=ETH XSCORE=BTC python research/deep/policy_stake.py (ETH market on the BTC T-120 signal, top 15 %, confirmation, good hours, settle)"""
import os, sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
import policy_lab as L
from policy_lab import run, date, Tix, ASSET
TIER = 0.70 if ASSET == "BTC" else 0.85; QS = (25.0, 50.0, 100.0) if ASSET == "BTC" else (0.0, 10.0, 25.0); ND = pd.Series(date).nunique(); rng = np.random.default_rng(8)
STAKES = [5, 10, 25, 50, 100, 150, 200, 300, 500, 1000] if ASSET == "BTC" else [2.5, 5, 10, 15, 20, 25, 35, 50, 75, 100, 200]
def good(i): t = pd.to_datetime(Tix[i], unit="s"); return (t.dayofweek.values >= 5) | (t.hour.values >= 16)
print(f"{ASSET}: decision T-120{' (BTC signal)' if os.environ.get('XSCORE') else ''}, top {int(round((1-TIER)*100))} %, T-45 confirmation, weekends + weekdays 16-24 UTC, bid 0.50 to T-15, hold to settlement; {ND} days")
for Q in QS:
    r = run(dec=-120, tier=TIER, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=Q); m = good(r["i"]); v = r["v"][m]; e = r["e"][m]; i = r["i"][m]; va = r["vol_after"][m]; vt = r["vol"][m]; d_ = date[i]
    print(f"\n=== queue assumed ahead of us at 0.50: ${Q:.0f} ===   {len(v)/ND:.1f} filled trades a day, win {(v>0).mean():.1%}, {v.mean()*100:+.2f} c/share")
    print(f"   dollars we could get per filled trade (after the queue): 10th pct ${np.percentile(va,10):.0f}, 25th ${np.percentile(va,25):.0f}, median ${np.median(va):.0f}, 75th ${np.percentile(va,75):.0f}, 90th ${np.percentile(va,90):.0f}, mean ${va.mean():.0f}  (all that traded at our price: median ${np.median(vt):.0f})")
    print("   edge by size of what was available: " + " | ".join(f"${lo:.0f}-{hi:.0f}: {((va>=lo)&(va<hi)).mean():.0%} of trades, win {(v[(va>=lo)&(va<hi)]>0).mean():.1%}" for lo, hi in ((0, 5), (5, 15), (15, 40), (40, 100), (100, 1e9)) if ((va>=lo)&(va<hi)).sum() > 30))
    print("   stake   $ filled per trade (avg)  share of stake filled   $/day [95% range]        total over the period   c per $1 filled   extra $/day vs previous")
    prev = 0.0
    for S in STAKES:
        f = np.minimum(S, va); pnl = v * f / e; d = pd.Series(pnl).groupby(d_).sum(); bs = [d.values[k].sum() / ND for k in (rng.integers(0, len(d), len(d)) for _ in range(1000))]
        print(f"   {S:>6}      {f.mean():>7.2f}               {f.sum()/(S*len(f)):>6.1%}          {pnl.sum()/ND:>7.2f} [{np.percentile(bs,2.5):>6.2f},{np.percentile(bs,97.5):>7.2f}]      ${pnl.sum():>8,.0f}            {pnl.sum()/f.sum()*100:+.2f}            {pnl.sum()/ND - prev:+.2f}"); prev = pnl.sum() / ND
