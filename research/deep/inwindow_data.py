"""Executable prices inside every BTC 5-minute window (Jun 1 - Nov 7): at each mark (30 s ... 240 s after the open), the price the next market
buyer of UP actually paid and the price the next market seller of UP actually got within the following seconds, and the dollars that traded.
  python research/deep/inwindow_data.py  ->  /home/user/data/deep/inwindow_BTC.parquet"""
import glob, numpy as np, pandas as pd
MARKS = [30, 60, 90, 120, 150, 180, 210, 240]; out = []
for f in sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t["rel"] = t.ts - t.window_start; t = t[(t.rel >= 0) & (t.rel < 262)]
    if not len(t): continue
    o0 = t.outcome_index.values == 0; buy = t.side.values == "BUY"; t["pu"] = np.where(o0, t.price.values, 1 - t.price.values); t["lu"] = (o0 & buy) | (~o0 & ~buy); t["usd"] = t.price.values * t["size"].values
    t = t.sort_values(["window_start", "ts"], kind="stable"); res = {}
    for m in MARKS:
        for lag, tag in ((1, "a"), (3, "b")):                       # a: prints from 1 s after the mark, b: from 3 s after (slower execution)
            w = t[(t.rel >= m + lag) & (t.rel < m + lag + 12)]
            g = w[w.lu].groupby("window_start"); res[f"ask{tag}_{m}"] = g.pu.first(); res[f"askusd{tag}_{m}"] = g.usd.sum()
            g = w[~w.lu].groupby("window_start"); res[f"bid{tag}_{m}"] = g.pu.first(); res[f"bidusd{tag}_{m}"] = g.usd.sum()
        res[f"last_{m}"] = t[t.rel < m].groupby("window_start").pu.last()
    out.append(pd.DataFrame(res))
D = pd.concat(out); D = D[~D.index.duplicated()]; D.index.name = "T"; D.to_parquet("/home/user/data/deep/inwindow_BTC.parquet")
print("windows", len(D), "| share with a market buy of UP within 12 s of the mark:", {m: round(float(D[f"aska_{m}"].notna().mean()), 2) for m in MARKS}, "| median spread at T+120:", round(float((D.aska_120 - D.bida_120).median()), 3))
