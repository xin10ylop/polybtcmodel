"""Cross-venue check, Limitless vs Polymarket, BTC 5-minute Up/Down (both settle on the same Chainlink 60-second TWAP).  Every ~2 s: best bid / ask and size
of Up on both books, and the cost of holding BOTH outcomes across the two venues (Up on one, Down on the other); under $1.00 before fees would be an arbitrage.
Read-only, public endpoints, no account.   python research/deep/limitless_sampler.py [minutes] -> /home/user/data/deep/limitless_samples.csv"""
import sys, time, json, csv, requests
MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 11; S = requests.Session(); out = "/home/user/data/deep/limitless_samples.csv"; tok = {}; f = open(out, "a", newline=""); w = csv.writer(f)
if f.tell() == 0: w.writerow(["ts", "T", "rel", "l_bid", "l_bid_sz", "l_ask", "l_ask_sz", "l_mid", "p_bid", "p_bid_sz", "p_ask", "p_ask_sz"])
def poly_tok(T):
    if T not in tok:
        e = S.get("https://gamma-api.polymarket.com/events", params=dict(slug=f"btc-updown-5m-{T}"), timeout=10).json(); tok[T] = json.loads(e[0]["markets"][0]["clobTokenIds"])[0]
    return tok[T]
end = time.time() + MIN * 60
while time.time() < end:
    t = time.time(); T = int(t) // 300 * 300
    for Tw in (T, T + 300):                      # the running window and the next one (pre-open)
        try:
            lb = S.get(f"https://api.limitless.exchange/markets/btc-up-or-down-5-min-{Tw}/orderbook", timeout=8).json(); pb = S.get("https://clob.polymarket.com/book", params=dict(token_id=poly_tok(Tw)), timeout=8).json()
            lbid = max(lb.get("bids") or [], key=lambda x: x["price"], default=None); lask = min(lb.get("asks") or [], key=lambda x: x["price"], default=None)
            pbid = max(pb.get("bids") or [], key=lambda x: float(x["price"]), default=None); pask = min(pb.get("asks") or [], key=lambda x: float(x["price"]), default=None)
            w.writerow([round(time.time(), 2), Tw, round(time.time() - Tw, 1), lbid and lbid["price"], lbid and lbid["size"] / 1e6, lask and lask["price"], lask and lask["size"] / 1e6, lb.get("midpoint"), pbid and float(pbid["price"]), pbid and float(pbid["size"]), pask and float(pask["price"]), pask and float(pask["size"])]); f.flush()
        except Exception as ex: w.writerow([round(time.time(), 2), Tw, "", "ERR", repr(ex)[:80]]); f.flush()
    time.sleep(max(0.0, 2.0 - (time.time() - t)))
