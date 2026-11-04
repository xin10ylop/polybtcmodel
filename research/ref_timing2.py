import numpy as np, pandas as pd, datetime as dt, os
mk = pd.read_parquet("/home/user/data/polymarket/markets.parquet")
mk = mk[(mk.outcome_up_price.isin([0.0, 1.0])) & np.isfinite(mk.price_to_beat)].copy()
d0 = dt.date(2026, 7, 31); d1 = dt.date(2026, 11, 4)
days = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
c = np.concatenate([np.load(f"/home/user/data/days/{d}.npz")["s_close"] if os.path.exists(f"/home/user/data/days/{d}.npz") else np.full(86400, np.nan) for d in days])
m = np.isnan(c); idx = np.where(~m, np.arange(len(c)), 0); np.maximum.accumulate(idx, out=idx); c = c[idx]; lc = np.log(c)
t0 = int(dt.datetime(d0.year, d0.month, d0.day, tzinfo=dt.UTC).timestamp())
i = (mk.window_start.values - t0).astype(int); ok = (i > 400) & (i + 700 < len(c)); i = i[ok]; mk = mk[ok]
ptb = np.log(mk.price_to_beat.values); up = (mk.outcome_up_price.values == 1.0)
month = pd.to_datetime(mk.window_start.values, unit="s").strftime("%Y-%m")
print("n", len(i))
best = []
for k in range(-60, 11):
    dev = 1e4 * (lc[i + k] - ptb); best.append((np.median(np.abs(dev - np.median(dev))), k, np.median(dev)))
best.sort(); print("start ref vs binance point: best (MAD bp, k, median offset bp):", [(round(a, 2), b, round(cc, 2)) for a, b, cc in best[:6]])
for mo in np.unique(month):
    sel = month == mo; bb = []
    for k in range(-60, 11):
        dev = 1e4 * (lc[i[sel] + k] - ptb[sel]); bb.append((np.median(np.abs(dev - np.median(dev))), k))
    bb.sort(); print("  ", mo, "best k:", bb[:3])
for w in (3, 5, 10, 20):
    bw = []
    for k in range(-50, -10):
        avg = np.mean([lc[i + k - j] for j in range(w)], axis=0); dev = 1e4 * (avg - ptb); bw.append((np.median(np.abs(dev - np.median(dev))), k))
    bw.sort(); print(f"  avg over {w}s ending at k: best", [(round(a, 2), b) for a, b in bw[:3]])
off = np.median(lc[i - 28] - ptb)
print("--- end reference timing: agreement of outcome with sign(binance[T+300+k2] - off - ptb) ---")
res = [(np.mean(((lc[i + 300 + k2] - off - ptb) >= 0) == up), k2) for k2 in range(-60, 11)]
res.sort(reverse=True); print([(round(a, 4), b) for a, b in res[:8]])
for mo in np.unique(month):
    sel = month == mo
    r = [(np.mean(((lc[i[sel] + 300 + k2] - off - ptb[sel]) >= 0) == up[sel]), k2) for k2 in range(-60, 11)]
    r.sort(reverse=True); print("  ", mo, [(round(a, 4), b) for a, b in r[:3]])
print("--- pure binance (k1,k2) fine grid ---")
res = [(np.mean(((lc[i + 300 + k2] - lc[i + k1]) >= 0) == up), k1, k2) for k1 in range(-40, -15) for k2 in range(-40, -15)]
res.sort(reverse=True); print([(round(a, 4), b, cc) for a, b, cc in res[:8]])
# how often is the outcome 'undetermined' by binance (|move| small)?
s = lc[i + 270] - lc[i - 30]
for th in (1, 2, 5, 10):
    sel = np.abs(s) * 1e4 > th
    print(f"|binance move T-30->T+270| > {th} bp: share {sel.mean():.3f}, agreement {np.mean((s[sel] >= 0) == up[sel]):.4f}")
