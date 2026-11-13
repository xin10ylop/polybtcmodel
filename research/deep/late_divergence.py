"""The last minute of the window, on the simulated fills of the live setup (trading hours).  Settlement compares the average of Bitcoin over the LAST 60 seconds with
the average over the 60 seconds before the open, so the '60-second average' line (what the settlement stream shows) lags the price by up to a minute.
States tested at T+240, T+255 and T+270:  A) the lagging average is still on our side but the price itself is already against us (looks like winning, is losing);
B) the price itself is on our side but our shares are cheap anyway.  For each: how often we still win, what the shares sell for (first sale stamped 5 s later,
fee off), and whether selling beats holding.     python research/deep/late_divergence.py"""
import sys, glob, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/btcmovement2")
from btcmodel.timeline import load_timeline
Z = np.load("/home/user/data/deep/stoploss_BTC.npz"); W = Z["W"]; side = Z["side"]; win = Z["win"]; d = pd.to_datetime(Z["date"]); n = len(W); pos = {int(w): k for k, w in enumerate(W)}; MK = np.array([210, 225, 240, 255, 270]); LAG = 5
tl = load_timeline(); t0 = int(tl.t0_epoch); sc_ = tl["s_close"]; TW = pd.Timestamp("2026-09-01").timestamp(); spot = np.full((n, len(MK)), np.nan); avg = np.full((n, len(MK)), np.nan); fin = np.full((n, len(MK)), np.nan)
for k, w in enumerate(W):
    w = int(w); a = pd.Series(np.asarray(sc_[w - 60 - t0: w + 300 - t0], dtype=float)).ffill().bfill().values; ref = a[:60].mean() if w >= TW else a[59]; sg = 1.0 if side[k] == 0 else -1.0
    for j, m in enumerate(MK): spot[k, j] = sg * (a[60 + m - 1] - ref); avg[k, j] = sg * (a[m:60 + m].mean() - ref); fin[k, j] = sg * (a[300:60 + m].mean() - ref) if m > 240 else np.nan      # price now / average of the last 60 s / part of the final minute already seen
sell = np.full((n, len(MK)), np.nan)
for f in sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t = t[t.window_start.isin(pos)]
    if not len(t): continue
    t["rel"] = t.ts - t.window_start; t = t[(t.rel >= 200) & (t.rel < 300)].sort_values(["window_start", "ts"], kind="stable")
    for w, g in t.groupby("window_start", sort=False):
        k = pos[int(w)]; rel = g.rel.values; o0 = g.outcome_index.values == 0; buy = g.side.values == "BUY"; ours = o0 if side[k] == 0 else ~o0; ps = np.where(ours, g.price.values, 1 - g.price.values); hit = (ours & ~buy) | (~ours & buy); hr, hp = rel[hit], ps[hit]
        q = np.searchsorted(hr, MK + LAG, side="left"); okq = q < len(hr); okq[okq] &= hr[q[okq]] < (MK + LAG)[okq] + 15; sell[k, okq] = hp[q[okq]]
tt = pd.to_datetime(W, unit="s"); good = (tt.dayofweek.values >= 5) | (tt.hour.values >= 16); N = np.clip(sell, 0.001, 0.999); N = N - 0.07 * N * (1 - N); day = d.values.astype("datetime64[D]"); rng = np.random.default_rng(5); late = np.asarray(d >= "2026-09-01")
def line(lbl, m, j):
    m = m & good & np.isfinite(N[:, j]) & late
    if m.sum() < 25: return {"state": lbl, "moment": f"T+{MK[j]}", "trades": int(m.sum())}
    e = N[m, j] - win[m]; g = pd.Series(e).groupby(day[m]).sum().values; b = g[rng.integers(0, len(g), (2000, len(g)))].sum(1) / m.sum()
    return {"state": lbl, "moment": f"T+{MK[j]}", "trades": int(m.sum()), "still win %": round(100 * win[m].mean(), 1), "shares sell for ¢ (fee off)": round(100 * N[m, j].mean(), 1), "selling minus holding ¢/share": round(100 * e.mean(), 2), "95% lo": round(100 * np.quantile(b, .025), 1), "95% hi": round(100 * np.quantile(b, .975), 1), "$ at $100 a trade": round(200 * e.sum())}
pd.set_option("display.width", 250); rows = []
for j in (2, 3, 4):
    rows += [line("A) average still $10+ on our side, price already $5+ against us", (avg[:, j] >= 10) & (spot[:, j] <= -5), j), line("A2) average on our side (any), price against us (any)", (avg[:, j] > 0) & (spot[:, j] < 0), j),
             line("B) price $10+ on our side, yet shares under 45c", (spot[:, j] >= 10) & (N[:, j] < 0.45), j), line("B2) price on our side (any), shares under 40c", (spot[:, j] > 0) & (N[:, j] < 0.40), j),
             line("C) shares 30-40c, whatever Bitcoin shows", (N[:, j] >= 0.30) & (N[:, j] < 0.40), j), line("D) price $10+ against us", spot[:, j] <= -10, j)]
print("since Sept 1 (the 60-second-average settlement rule), trading hours; + in 'selling minus holding' = selling is better"); print(pd.DataFrame(rows).to_string(index=False))
j = 3; m = good & late & np.isfinite(fin[:, j]); print(f"\nat T+255, the 15 seconds of the final minute already seen point the way the window ends in {100*((fin[m, j] > 0) == win[m]).mean():.1f} % of trades; the lagging 60-second average in {100*((avg[m, j] > 0) == win[m]).mean():.1f} %; the price itself in {100*((spot[m, j] > 0) == win[m]).mean():.1f} %")
