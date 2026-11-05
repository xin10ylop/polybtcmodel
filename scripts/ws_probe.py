"""Probe Binance market-data websockets from this host (20 s each) and the REST polling freshness."""
import json, time, socket, requests, websocket
_gai = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [x for x in _gai(*a, **k) if x[0] == socket.AF_INET] or _gai(*a, **k)
tests = [("futures aggTrade", "wss://fstream.binance.com/ws/btcusdt@aggTrade"),
         ("futures markPrice@1s", "wss://fstream.binance.com/ws/btcusdt@markPrice@1s"),
         ("futures !ticker@arr", "wss://fstream.binance.com/ws/!ticker@arr"),
         ("futures aggTrade, port 443 explicit", "wss://fstream.binance.com:443/ws/btcusdt@aggTrade"),
         ("futures mirror host", "wss://fstream.binance.vision/ws/btcusdt@aggTrade"),
         ("spot main stream", "wss://stream.binance.com:9443/ws/btcusdt@trade"),
         ("spot mirror stream", "wss://data-stream.binance.vision/ws/btcusdt@trade")]
for name, url in tests:
    t0 = time.time()
    try:
        ws = websocket.create_connection(url, timeout=20, suppress_origin=True)
        n = 0; first = None
        while time.time() - t0 < 20:
            m = ws.recv(); n += 1
            if first is None: first = m[:120]; print(f"{name}: first message after {time.time()-t0:.1f}s: {first}", flush=True)
            if n >= 20: break
        print(f"{name}: {n} messages in {time.time()-t0:.1f}s", flush=True); ws.close()
    except Exception as e:
        print(f"{name}: error after {time.time()-t0:.1f}s: {repr(e)[:120]}", flush=True)
print("REST polling freshness (fapi aggTrades, 10 polls):", flush=True)
for _ in range(10):
    t = time.time(); r = requests.get("https://fapi.binance.com/fapi/v1/aggTrades", params=dict(symbol="BTCUSDT", limit=1000), timeout=10)
    d = r.json(); age = (time.time() * 1000 - int(d[-1]["T"])) / 1000 if isinstance(d, list) else None
    print(f"  http {r.status_code}, request {time.time()-t:.2f}s, newest trade age {age if age is None else round(age, 2)}s, weight used {r.headers.get('x-mbx-used-weight-1m')}", flush=True); time.sleep(0.6)
