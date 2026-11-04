"""One-day look at Kalshi's 15-minute BTC up/down market (KXBTC15M) with the same questions we asked of
Polymarket: liquidity, when the price leaves 0.50, agreement of settlement with the Binance TWAP, and
whether the market price lags the Binance displacement (the model-free version of strategy B's edge).
  python research/kalshi_day.py 2026-11-03
Inputs: /home/user/data/kalshi/{markets,trades}_<day>.parquet (research/kalshi_fetch.py), Binance 1 s
closes from /home/user/data/days/<day>.npz (+ previous day for the first reference window).
"""
import sys, os, datetime as dt
import numpy as np, pandas as pd
from scipy.stats import norm
pd.set_option("display.width", 220)
day = dt.date.fromisoformat(sys.argv[1]); K = "/home/user/data/kalshi"
mk = pd.read_parquet(f"{K}/markets_{day}.parquet"); tr = pd.read_parquet(f"{K}/trades_{day}.parquet")
mk["open"] = pd.to_datetime(mk.open_time, utc=True); mk["close"] = pd.to_datetime(mk.close_time, utc=True)
EP = pd.Timestamp(0, tz="UTC")
mk["T0"] = ((mk.open - EP) // pd.Timedelta("1s")).astype(int); mk["T1"] = ((mk.close - EP) // pd.Timedelta("1s")).astype(int)
mk = mk.sort_values("T0").reset_index(drop=True)
tr["t"] = (tr.ts - EP) / pd.Timedelta("1s"); tr = tr.sort_values("t", kind="stable").reset_index(drop=True)   # the API returns newest first
tr = tr.merge(mk[["ticker", "T0", "T1"]], on="ticker"); tr["rel"] = tr.t - tr.T0
# Binance 1 s closes for the day and the previous day
prev = day - dt.timedelta(days=1); base = int(dt.datetime(prev.year, prev.month, prev.day, tzinfo=dt.UTC).timestamp())
closes = np.concatenate([np.load(f"/home/user/data/days/{d}.npz")["s_close"] for d in (prev, day)])
closes = pd.Series(closes).ffill().values
def px(t): return closes[int(t) - base]
def twap(t): return np.nanmean(closes[int(t) - base - 60:int(t) - base])
mk["ref"] = mk.T0.map(twap); mk["end"] = mk.T1.map(twap); mk["bin_up"] = (mk.end >= mk.ref).astype(int); mk["res_up"] = (mk.result == "yes").astype(int)
print(f"== Kalshi KXBTC15M {day}: {len(mk)} markets, {len(tr)} trades, {tr['count'].sum():,.0f} contracts")
print(f"settlement (CF Benchmarks BRTI 60 s average) agrees with the Binance 60 s TWAP label in {np.mean(mk.bin_up == mk.res_up)*100:.1f}% of markets; UP share {mk.res_up.mean():.2f}")
print(f"per market: trades median {tr.groupby('ticker').size().median():.0f}, contracts median {tr.groupby('ticker')['count'].sum().median():.0f}, "
      f"first trade at open+{tr.groupby('ticker').rel.min().median():.0f}s (median), last at close-{(900 - tr.groupby('ticker').rel.max()).median():.0f}s")
# when does the price leave 0.50?  (first print outside [0.45, 0.55])
first_off = tr[(tr.yes_price < 0.45) | (tr.yes_price > 0.55)].groupby("ticker").rel.min()
print(f"first print outside [0.45, 0.55]: median open+{first_off.median():.0f}s (Polymarket 5-min: ~T-25 s, i.e. before the open)")
pre = tr[tr.rel < 0]; print(f"prints before the open: {len(pre)} ({len(pre)/len(tr)*100:.1f}% of trades); price range {pre.yes_price.min() if len(pre) else np.nan:.2f}-{pre.yes_price.max() if len(pre) else np.nan:.2f}")

# model-free displacement test: at several times into the window, last Kalshi price vs a fair value from Binance
def rv(t, n=3600):
    c = closes[int(t) - base - n:int(t) - base]; r = np.diff(np.log(c)); return np.sqrt(np.nanmean(r * r))
rows = []
for m in mk.itertuples():
    sig = rv(m.T0)
    for e in (30, 60, 120, 180, 300, 450, 600, 720, 840):
        t = m.T0 + e; tt = tr[(tr.ticker == m.ticker) & (tr.t <= t)]
        if not len(tt): continue
        last = tt.yes_price.iloc[-1]; d = np.log(px(t) / m.ref); tau = max(900 - e, 30)
        fair = norm.cdf(d / (sig * np.sqrt(tau) + 1e-12))
        rows.append(dict(ticker=m.ticker, e=e, mkt=last, fair=fair, d_bp=d * 1e4, up=m.res_up, bin_up=m.bin_up, age=t - tt.t.iloc[-1]))
df = pd.DataFrame(rows)
def brier(p, y): return np.mean((p - y) ** 2)
print("\nat elapsed e: n markets, median age of the last print, Brier of market price vs Binance-displacement fair value vs 50/50 blend; hit rate of sign(fair-mkt) when |fair-mkt| >= 0.10")
for e, g in df.groupby("e"):
    big = g[(g.fair - g.mkt).abs() >= 0.10]
    hit = np.mean(np.where(big.fair > big.mkt, big.up == 1, big.up == 0)) if len(big) else np.nan
    print(f"  e={e:4d}s  n={len(g):3d}  age {g.age.median():5.1f}s  brier mkt {brier(g.mkt, g.up):.4f}  fair {brier(g.fair, g.up):.4f}  blend {brier(0.5*(g.mkt+g.fair), g.up):.4f}  |  |fair-mkt|>=0.10: n={len(big):3d} hit {hit if np.isnan(hit) else round(hit,3)}")
# a crude B-type trade: at e=60 s buy the side the displacement favours when it disagrees with the market by >= delta, hold to settlement (no exit fee)
print("\nmodel-free B-type at e=60 s (buy at the last print, hold to settlement; Kalshi fee ~0.07 p(1-p) only if taker):")
g = df[df.e == 60]
for delta in (0.05, 0.10, 0.15, 0.20):
    s = g[(g.fair - g.mkt).abs() >= delta]
    if not len(s): print(f"  delta {delta}: no trades"); continue
    side_up = s.fair > s.mkt; entry = np.where(side_up, s.mkt, 1 - s.mkt); won = np.where(side_up, s.up == 1, s.up == 0)
    pnl = np.where(won, 1 - entry, -entry); fee = 0.07 * entry * (1 - entry)
    print(f"  delta {delta}: {len(s)} trades, win {won.mean():.3f}, mean pnl/share {pnl.mean():+.3f} (maker) / {(pnl - fee).mean():+.3f} (taker entry), return on stake {(pnl/entry).mean()*100:+.1f}%")
df.to_csv(f"/home/user/polybtcmodel/reports/kalshi_{day}.csv", index=False); print("saved reports/kalshi_<day>.csv")
