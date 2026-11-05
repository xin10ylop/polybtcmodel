"""Probe Binance's futures websocket from this host: three subscription styles, 20 s each, prints the first message or the error."""
import json, time, socket, websocket
_gai = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [x for x in _gai(*a, **k) if x[0] == socket.AF_INET] or _gai(*a, **k)
tests = [("raw stream", "wss://fstream.binance.com/ws/btcusdt@aggTrade", None),
         ("combined stream", "wss://fstream.binance.com/stream?streams=btcusdt@aggTrade", None),
         ("subscribe message", "wss://fstream.binance.com/ws", {"method": "SUBSCRIBE", "params": ["btcusdt@aggTrade"], "id": 1})]
for name, url, sub in tests:
    t0 = time.time()
    try:
        ws = websocket.create_connection(url, timeout=20, suppress_origin=True)
        print(f"{name}: connected in {time.time()-t0:.1f}s", flush=True)
        if sub: ws.send(json.dumps(sub))
        n = 0; first = None
        while time.time() - t0 < 20:
            m = ws.recv(); n += 1
            if first is None: first = m[:160]; print(f"  first message after {time.time()-t0:.1f}s: {first}", flush=True)
            if n >= 50: break
        print(f"  {n} messages in {time.time()-t0:.1f}s", flush=True); ws.close()
    except Exception as e:
        print(f"{name}: error after {time.time()-t0:.1f}s: {repr(e)[:160]}", flush=True)
