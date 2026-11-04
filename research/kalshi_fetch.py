"""One day of Kalshi's 'BTC price up in next 15 mins?' markets (series KXBTC15M): market list + every trade.
  python research/kalshi_fetch.py 2026-11-03    -> /home/user/data/kalshi/markets_<day>.parquet, trades_<day>.parquet
Settlement: 60 s simple average of CF Benchmarks BRTI before the close vs before the open (same TWAP shape as Polymarket's 5-minute markets, different index)."""
import sys, time, datetime as dt, requests, pandas as pd, json
B = "https://api.elections.kalshi.com/trade-api/v2"; OUT = "/home/user/data/kalshi"
day = dt.date.fromisoformat(sys.argv[1]); t0 = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp()); t1 = t0 + 86400
def get(path, **p):
    for k in range(6):
        try:
            r = requests.get(f"{B}/{path}", params=p, timeout=30)
            if r.status_code == 200: return r.json()
            print("http", r.status_code, path, str(p)[:80], flush=True)
        except Exception as e: print("err", repr(e)[:80], flush=True)
        time.sleep(1 + k)
    raise RuntimeError(path)
s = get("series/KXBTC15M"); print("series:", json.dumps(s)[:400], flush=True)
mk, cur = [], None
while True:
    r = get("markets", series_ticker="KXBTC15M", status="settled", min_close_ts=t0 + 1, max_close_ts=t1, limit=200, **({"cursor": cur} if cur else {}))
    mk += r.get("markets", []); cur = r.get("cursor")
    if not cur or not r.get("markets"): break
mdf = pd.DataFrame(mk); print("markets:", len(mdf), flush=True)
keep = [c for c in ("ticker", "open_time", "close_time", "result", "floor_strike", "strike_type", "volume", "volume_fp", "open_interest", "liquidity", "last_price", "settlement_value", "settlement_ts") if c in mdf]
mdf[keep].to_parquet(f"{OUT}/markets_{day}.parquet", index=False)
rows = []
for i, m in enumerate(mdf.itertuples()):
    cur = None; n = 0
    while True:
        r = get("markets/trades", ticker=m.ticker, limit=1000, **({"cursor": cur} if cur else {}))
        for t in r.get("trades", []):
            rows.append((m.ticker, t["created_time"], float(t["yes_price_dollars"]), float(t.get("count_fp") or t.get("count") or 0), t.get("taker_side"), t.get("taker_book_side")))
        n += len(r.get("trades", [])); cur = r.get("cursor")
        if not cur or not r.get("trades"): break
    if (i + 1) % 16 == 0: print(f"{i+1}/{len(mdf)} markets, {len(rows)} trades", flush=True)
tdf = pd.DataFrame(rows, columns=["ticker", "created_time", "yes_price", "count", "taker_side", "taker_book_side"])
tdf["ts"] = pd.to_datetime(tdf.created_time, utc=True)
tdf.to_parquet(f"{OUT}/trades_{day}.parquet", index=False); print("saved", len(tdf), "trades", flush=True)
