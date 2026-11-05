"""Record raw perp trades from Hyperliquid (websocket) and OKX (REST polling, as the trader does) for N minutes, to
calibrate both against the Binance USDT-M archive of the same period once it is published (next day).
  python research/record_perp_feeds.py 60   -> /home/user/data/perp_feeds/<start>_{hl,okx}.parquet"""
import sys, os, json, time, threading, requests, websocket, pandas as pd
MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 60; OUT = "/home/user/data/perp_feeds"; os.makedirs(OUT, exist_ok=True)
t_start = int(time.time()); t_end = t_start + 60 * MIN; tag = time.strftime("%Y-%m-%d_%H%M", time.gmtime(t_start))
hl, okx = [], []
def rec_hl():
    while time.time() < t_end:
        try:
            ws = websocket.create_connection("wss://api.hyperliquid.xyz/ws", timeout=30)
            ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": "BTC"}}))
            while time.time() < t_end:
                m = json.loads(ws.recv())
                if m.get("channel") == "trades":
                    for t in m["data"]: hl.append((int(t["time"]) / 1000, float(t["px"]), float(t["sz"]), t["side"]))
        except Exception as e:
            print("hl error", repr(e)[:80], flush=True); time.sleep(2)
def rec_okx():
    last = None
    while time.time() < t_end:
        try:
            if last is None:
                rows = requests.get("https://www.okx.com/api/v5/market/trades", params=dict(instId="BTC-USDT-SWAP", limit=100), timeout=10).json().get("data", [])
            else:
                rows = []
                for _ in range(10):
                    d = requests.get("https://www.okx.com/api/v5/market/history-trades", params=dict(instId="BTC-USDT-SWAP", type="1", before=last, limit=100), timeout=10).json().get("data", [])
                    if not d: break
                    rows += d; last = max(int(x["tradeId"]) for x in d)
                    if len(d) < 100: break
            if rows:
                last = max(int(x["tradeId"]) for x in rows)
                for r in rows: okx.append((int(r["ts"]) / 1000, float(r["px"]), float(r["sz"]) * 0.01, r["side"], int(r["tradeId"])))
            time.sleep(0.4)
        except Exception as e:
            print("okx error", repr(e)[:80], flush=True); time.sleep(2)
th = [threading.Thread(target=rec_hl, daemon=True), threading.Thread(target=rec_okx, daemon=True)]; [t.start() for t in th]
while time.time() < t_end: time.sleep(30); print(f"{int(time.time()-t_start)}s: hl {len(hl)} okx {len(okx)}", flush=True)
time.sleep(3)
pd.DataFrame(hl, columns=["t", "px", "q", "side"]).drop_duplicates().to_parquet(f"{OUT}/{tag}_hl.parquet", index=False)
pd.DataFrame(okx, columns=["t", "px", "q", "side", "tid"]).drop_duplicates("tid").to_parquet(f"{OUT}/{tag}_okx.parquet", index=False)
print("saved", tag, len(hl), len(okx), flush=True)
