"""Is the live perp feed complete? Runs the trader's own feed function (PERP_FEED=binance_rest | binance | okx) for N seconds
into a scratch timeline, then compares per-second trade count, volume, largest trade and taker-buy share with (a) Binance spot
1 s klines for the same seconds (REST, data-api.binance.vision) and (b) the archive's median for this UTC hour over the cached
day files. The model's pre-open edge (strategy A) rides on the perp lead-lag features: a feed that delivers only part of the
trades mutes A while B and D look normal.
  PERP_FEED=binance_rest /home/user/venv/bin/python3 scripts/perp_feed_check.py [seconds=90]
"""
import os, sys, time, threading, glob, datetime as dt
import numpy as np, requests
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import paper_trader as pt
N = int(sys.argv[1]) if len(sys.argv) > 1 else 90
feed = os.environ.get("PERP_FEED", "binance_rest"); fn = {"binance_rest": pt.binance_perp_rest, "binance": pt.binance_perp_ws, "okx": pt.okx_ws}[feed]
t0 = int(time.time()) // pt.DAY * pt.DAY; tl = pt.LiveTimeline(t0, days=2); stop = threading.Event()
th = threading.Thread(target=fn, args=(tl, stop), daemon=True); th.start()
print(f"collecting {N}s of {feed} perp trades ...", flush=True); t_start = int(time.time()) + 2; time.sleep(N); stop.set(); t_end = int(time.time()) - 2
secs = [s for s in range(t_start, t_end) if s in tl.perp_pending]
if not secs: print("no perp trades received"); sys.exit(1)
bars = [tl.perp_pending[s] for s in secs]; n = t_end - t_start
p_ntr = sum(b["p_ntr"] for b in bars) / n; p_vol = sum(b["vv"] for b in bars) / n; p_max = np.mean([b["p_maxq"] for b in bars]); p_buy = sum(b["p_buy"] for b in bars) / max(sum(b["vv"] for b in bars), 1e-9)
empty = n - len(secs)
# spot klines for the same seconds
r = requests.get("https://data-api.binance.vision/api/v3/klines", params=dict(symbol="BTCUSDT", interval="1s", startTime=t_start * 1000, endTime=t_end * 1000 - 1, limit=1000), timeout=20).json()
s_cnt = sum(int(k[8]) for k in r) / n; s_vol = sum(float(k[5]) for k in r) / n
print(f"\nlive, {n} s from {dt.datetime.fromtimestamp(t_start, dt.UTC):%H:%M:%S} UTC:")
print(f"  perp: {p_ntr:6.1f} trades/s  {p_vol:6.2f} BTC/s  largest trade/s {p_max:5.2f} BTC  taker-buy share {p_buy:.2f}  seconds without a trade {empty}/{n}")
print(f"  spot: {s_cnt:6.1f} trades/s  {s_vol:6.2f} BTC/s      ->  perp/spot ratio: trades {p_ntr / max(s_cnt, 1e-9):.2f}  volume {p_vol / max(s_vol, 1e-9):.2f}")
# archive reference for this UTC hour
h = dt.datetime.fromtimestamp(t_start, dt.UTC).hour; rows = []
for f in sorted(glob.glob("/home/user/data/days/2026-*.npz"))[-10:]:
    z = np.load(f)
    if "p_ntr" not in z.files: continue
    sl = slice(h * 3600, (h + 1) * 3600); rows.append((np.nanmean(z["p_ntr"][sl]), np.nanmean(z["p_buy"][sl] + z["p_sell"][sl]), np.nanmean(z["p_maxq"][sl]), np.nanmean(z["s_count"][sl]), np.nanmean(z["s_volume"][sl]), np.mean(~np.isfinite(z["p_close"][sl]))))
if rows:
    a = np.median(np.array(rows), axis=0)
    print(f"archive, same UTC hour {h:02d}, median of the last {len(rows)} days:")
    print(f"  perp: {a[0]:6.1f} trades/s  {a[1]:6.2f} BTC/s  largest trade/s {a[2]:5.2f} BTC  seconds without a trade {a[5]:.1%}")
    print(f"  spot: {a[3]:6.1f} trades/s  {a[4]:6.2f} BTC/s      ->  perp/spot ratio: trades {a[0] / a[3]:.2f}  volume {a[1] / a[4]:.2f}")
    ok = (p_ntr / max(s_cnt, 1e-9)) / (a[0] / a[3]) > 0.6 and (p_vol / max(s_vol, 1e-9)) / (a[1] / a[4]) > 0.6
    print("\nVERDICT:", "perp feed complete (live perp/spot ratios within the archive's)" if ok else "PERP FEED INCOMPLETE: live perp/spot ratios far below the archive's")
else:
    print("no archive day files with perp data to compare")
