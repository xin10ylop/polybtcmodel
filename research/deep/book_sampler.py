"""Sample the live Polymarket order books of the next BTC and ETH 5-minute windows shortly before they open (T-120, -60, -45, -20):
$ offered at <= 0.50 / 0.51 / 0.52 and $ bid at >= 0.50 / 0.49 per token.  python research/deep/book_sampler.py <n_windows>  -> /home/user/data/deep/book_samples.csv"""
import requests, json, time, sys, csv
S = requests.Session(); N = int(sys.argv[1]) if len(sys.argv) > 1 else 4; OUT = "/home/user/data/deep/book_samples.csv"; rows = []
def snap(asset, T, rel):
    try:
        e = S.get(f"https://gamma-api.polymarket.com/events?slug={asset}-updown-5m-{T}", timeout=10).json(); toks = json.loads(e[0]["markets"][0]["clobTokenIds"])
        for nm, tok in zip(("UP", "DOWN"), toks):
            b = S.get("https://clob.polymarket.com/book", params=dict(token_id=tok), timeout=10).json()
            asks = [(float(x["price"]), float(x["size"])) for x in b.get("asks", [])]; bids = [(float(x["price"]), float(x["size"])) for x in b.get("bids", [])]
            rows.append(dict(asset=asset.upper(), T=T, rel=rel, token=nm, best_ask=min(p for p, _ in asks) if asks else None, best_bid=max(p for p, _ in bids) if bids else None,
                             ask_usd_050=sum(p * s for p, s in asks if p <= 0.505), ask_usd_051=sum(p * s for p, s in asks if p <= 0.515), ask_usd_052=sum(p * s for p, s in asks if p <= 0.525),
                             bid_usd_050=sum(p * s for p, s in bids if p >= 0.495), bid_usd_049=sum(p * s for p, s in bids if p >= 0.485)))
    except Exception as ex: rows.append(dict(asset=asset.upper(), T=T, rel=rel, token="ERR " + repr(ex)[:60]))
T = int(time.time()) // 300 * 300 + 300
for k in range(N):
    Tk = T + 300 * k
    for rel in (-120, -60, -45, -20):
        while time.time() < Tk + rel: time.sleep(0.5)
        if time.time() > Tk + rel + 8: continue
        for a in ("btc", "eth"): snap(a, Tk, rel)
    w = csv.DictWriter(open(OUT, "w", newline=""), fieldnames=["asset", "T", "rel", "token", "best_ask", "best_bid", "ask_usd_050", "ask_usd_051", "ask_usd_052", "bid_usd_050", "bid_usd_049"]); w.writeheader(); w.writerows(rows)
print("done", len(rows))
