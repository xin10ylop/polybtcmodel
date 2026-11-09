"""Per window and decision time t (seconds after the open): the price a taker really paid for each token right after t (first taker BUY
of that token in [t, t+15), i.e. the ask that was lifted), the $ bought at or below that price in those 15 s, and the last UP-equivalent
print before t.  -> /home/user/data/deep/latequotes_<asset>.parquet     ASSET=BTC python research/deep/late_quotes.py"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import SRC, DEEP  # noqa: E402
ASSET = os.environ.get("ASSET", "BTC"); TS = [60, 90, 120, 150, 180, 210, 240, 260]; rows = []
tdir = "/home/user/data/polymarket/trades" if ASSET == "BTC" else "/home/user/data_eth/polymarket/trades"
for f in sorted(glob.glob(f"{tdir}/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t["rel"] = t.ts - t.window_start; t = t[(t.rel >= 30) & (t.rel < 300)].sort_values(["window_start", "ts"], kind="stable")
    t["pu"] = np.where(t.outcome_index == 0, t.price, 1 - t.price); t["usd"] = t.price * t["size"]
    for tt in TS:
        pre = t[t.rel < tt].groupby("window_start").pu.last().rename("last_up"); w = t[(t.rel >= tt) & (t.rel < tt + 15) & (t.side == "BUY")]
        d = pd.DataFrame(pre)
        for s, nm in ((0, "up"), (1, "dn")):
            ws = w[w.outcome_index == s]; g = ws.groupby("window_start"); first = g.price.first().rename(f"ask_{nm}"); d = d.join(first, how="outer")
            ws = ws.join(first, on="window_start"); d = d.join(ws[ws.price <= ws[f"ask_{nm}"] + 1e-9].groupby("window_start").usd.sum().rename(f"usd_{nm}"), how="left")
        d["t"] = tt; rows.append(d.reset_index())
    print(os.path.basename(f), flush=True)
Q = pd.concat(rows, ignore_index=True).rename(columns={"window_start": "T"}); Q.to_parquet(f"{DEEP}/latequotes_{ASSET}.parquet", index=False); print("saved", len(Q))
