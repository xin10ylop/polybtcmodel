import numpy as np, pandas as pd, datetime as dt, os
mk = pd.read_parquet("/home/user/data/polymarket/markets.parquet")
mk = mk[(mk.outcome_up_price.isin([0.0, 1.0]))].copy()
d0 = dt.date(2026, 7, 31); d1 = dt.date(2026, 11, 4)
days = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
c = np.concatenate([np.load(f"/home/user/data/days/{d}.npz")["s_close"] if os.path.exists(f"/home/user/data/days/{d}.npz") else np.full(86400, np.nan) for d in days])
m = np.isnan(c); idx = np.where(~m, np.arange(len(c)), 0); np.maximum.accumulate(idx, out=idx); c = c[idx]; lc = np.log(c)
t0 = int(dt.datetime(d0.year, d0.month, d0.day, tzinfo=dt.UTC).timestamp())
i = (mk.window_start.values - t0).astype(int); ok = (i > 400) & (i + 700 < len(c)); i = i[ok]; mk = mk[ok]
up = (mk.outcome_up_price.values == 1.0); ptb = np.log(mk.price_to_beat.values)
day = pd.to_datetime(mk.window_start.values, unit="s").strftime("%m-%d")
rows = []
for dd in np.unique(day):
    s = day == dd
    if s.sum() < 100: continue
    best = max(((np.mean(((lc[i[s] + 300 + k2] - lc[i[s] + k1]) >= 0) == up[s]), k1, k2) for k1 in range(-45, 6, 3) for k2 in range(-45, 6, 3)))
    # start-ref lag via ptb (if available)
    f = np.isfinite(ptb[s])
    if f.sum() > 50:
        bb = min(((np.median(np.abs((lc[i[s][f] + k] - ptb[s][f]) - np.median(lc[i[s][f] + k] - ptb[s][f]))), k) for k in range(-50, 6)))
    else:
        bb = (np.nan, np.nan)
    rows.append((dd, s.sum(), round(best[0], 3), best[1], best[2], round(bb[0] * 1e4, 2), bb[1]))
print(pd.DataFrame(rows, columns=["day", "n", "agree", "k1", "k2", "ptb_mad_bp", "ptb_k"]).to_string())
