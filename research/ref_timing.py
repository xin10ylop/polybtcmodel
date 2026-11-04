"""Where (in Binance time) are the Polymarket start and end reference prices sampled?"""
import numpy as np, pandas as pd, datetime as dt, os
mk = pd.read_parquet("/home/user/data/polymarket/markets.parquet")
mk = mk[(mk.outcome_up_price.isin([0.0, 1.0]))].copy()
d0 = dt.date(2026, 7, 31); d1 = dt.date(2026, 11, 4)
days = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
cs = []
for d in days:
    f = f"/home/user/data/days/{d}.npz"
    cs.append(np.load(f)["s_close"] if os.path.exists(f) else np.full(86400, np.nan))
c = np.concatenate(cs)
m = np.isnan(c); idx = np.where(~m, np.arange(len(c)), 0); np.maximum.accumulate(idx, out=idx); c = c[idx]
t0 = int(dt.datetime(d0.year, d0.month, d0.day, tzinfo=dt.UTC).timestamp())
i = (mk.window_start.values - t0).astype(int)
ok = (i > 400) & (i + 700 < len(c)); i = i[ok]; mk = mk[ok]
ptb = mk.price_to_beat.values; up = (mk.outcome_up_price.values == 1.0)
lc = np.log(c)
print("n markets", len(i))
print("--- start reference: MAD of log(binance[T+k]/ptb) and of moving averages ---")
best = []
for k in range(-90, 31, 1):
    dev = 1e4 * (lc[i + k] - np.log(ptb)); best.append((np.median(np.abs(dev - np.median(dev))), k, np.median(dev)))
best.sort(); print("point samples, best 8 (MAD bp, k, median bp):", [(round(a, 2), b, round(cc, 1)) for a, b, cc in best[:8]])
for w in (5, 10, 20, 30, 45, 60):
    bw = []
    for k in range(-60, 11, 1):
        avg = np.mean([lc[i + k - j] for j in range(w)], axis=0)
        dev = 1e4 * (avg - np.log(ptb)); bw.append((np.median(np.abs(dev - np.median(dev))), k))
    bw.sort(); print(f"avg over {w}s ending at k: best", [(round(a, 2), b) for a, b in bw[:4]])
print("--- outcome agreement with sign(binance[T+300+k2] - binance[T+k1]) ---")
res = []
for k1 in range(-60, 11, 5):
    for k2 in range(-60, 11, 5):
        s = lc[i + 300 + k2] - lc[i + k1]
        agree = np.mean((s >= 0) == up)
        res.append((agree, k1, k2))
res.sort(reverse=True); print("best (agreement, k1, k2):", [(round(a, 4), b, cc) for a, b, cc in res[:10]])
print("agreement at (0,0):", [r for r in res if r[1] == 0 and r[2] == 0])
# agreement using ptb as start and binance end with the USDT offset removed (median offset)
for k2 in (-45, -30, -25, -20, -15, -10, -5, 0):
    off = np.median(lc[i - 25] - np.log(ptb))
    s = lc[i + 300 + k2] - off - np.log(ptb)
    print(f"start=ptb, end=binance[T+300+{k2}] minus median offset: agreement {np.mean((s >= 0) == up):.4f}")
