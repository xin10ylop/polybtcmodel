"""Would the real fills also have filled at a bigger stake?  For every rule-A buy the live bot got filled, the public Polymarket tape of that market is
read and the shares that takers sold into our price on our side are counted, from our order's placement to its cancel time:
  after   = shares sold at our price AFTER our own fill was seen (plus our own shares): a bigger order at the same place in the queue gets these for sure
  total   = everything sold at our price while the order was resting (the most a bigger order could get, if it had been first in the queue)
  through = somebody sold BELOW our price while we were resting: the bids ahead of us were used up, so everything sold from then on reaches us
Reads the bot's logs and public data only; changes nothing.     python tests/fill_depth.py [/home/user/paper] [stake in $, default 100] [since ISO]"""
import os, sys, json, time, datetime as dt, requests
D = sys.argv[1] if len(sys.argv) > 1 else "/home/user/paper"; STAKE = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0; since = dt.datetime.fromisoformat(sys.argv[3] + "+00:00").timestamp() if len(sys.argv) > 3 else 0.0
CANCEL = int(os.environ.get("A_CANCEL_S", "-15")); LAG = 6          # a print's timestamp is its block time, a few seconds after the trade; the bot sees its fill within a second or two
def rd(f): p = os.path.join(D, f); return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
def get(url, params):
    for k in range(6):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429: time.sleep(2 + k); continue
            r.raise_for_status(); return r.json()
        except Exception as e: err = e; time.sleep(1 + k)
    raise RuntimeError(f"{url}: {err}")
orders = {o["T"]: o for o in rd("live_orders.jsonl") if o.get("event") == "placed" and o.get("kind") == "buy" and o.get("strategy") == "A" and o.get("ts", 0) >= since}
tr = rd("live_trades.jsonl"); fills = {}
for t in tr:
    if t.get("event") == "fill" and t.get("kind") == "buy" and t.get("strategy") == "A" and t.get("T") in orders:
        f = fills.setdefault(t["T"], dict(size=0.0, ts=t["ts"], side=t["side"], price=t["price"])); f["size"] += t.get("size", 0.0)
won = {t["T"]: t.get("won") for t in tr if t.get("event") == "settle" and t.get("strategy") == "A"}; rows = []
for T in sorted(fills):
    f = fills[T]; px = float(f["price"]); want = STAKE / px; s = 0 if str(f["side"]).lower().startswith("u") else 1
    try:
        e = get(f"https://gamma-api.polymarket.com/events", dict(slug=f"btc-updown-5m-{T}")); cid = e[0]["markets"][0]["conditionId"]; pr = []; cur = None
        while True:
            d = get("https://data-api.polymarket.com/v2/trades", dict(condition=cid, limit=1000, **({"cursor": cur} if cur else {}))); pr += d.get("data") or []; pg = d.get("pagination") or {}; cur = pg.get("next_cursor")
            if not pg.get("has_more") or not cur: break
    except Exception as ex: print(f"{T}: tape not available ({ex})"); continue
    aft = tot = tv = 0.0; thru = False
    for r in sorted(pr, key=lambda r: int(r["timestamp"])):
        ts = int(r["timestamp"]); o = int(r["outcome_index"]); p = float(r["price"]); po = p if o == s else 1 - p; sold = (o == s and r["side"] == "SELL") or (o != s and r["side"] == "BUY")
        if not sold or ts < orders[T]["ts"] or ts > T + CANCEL + LAG or po > px + 1e-6: continue
        tot += float(r["size"]); thru = thru or po < px - 0.005
        if thru: tv += float(r["size"])                       # from the first sale below our price on, the bids ahead of us are gone: all of it reaches us
        if ts >= f["ts"] + LAG: aft += float(r["size"])
    sure = min(want, max(f["size"] + aft, tv, f["size"])); may = min(want, max(tot, f["size"]))
    rows.append(dict(T=T, side=f["side"], won=won.get(T), got=f["size"], after=aft, total=tot, thru=thru, sure=sure, may=may, want=want, px=px))
if not rows: print("no filled rule-A buys found in", D); sys.exit(0)
print(f"stake ${STAKE:.0f} = {rows[0]['want']:.0f} shares at {rows[0]['px']:.2f}.  One line per real fill:")
print(f"{'window (UTC)':17s} {'side':5s} {'result':6s} {'we got':>7s} {'sold after our fill':>20s} {'sold while resting':>19s} {'through':>8s} {'filled for sure':>16s} {'at most':>8s}")
for r in rows:
    print(f"{dt.datetime.fromtimestamp(r['T'], dt.UTC):%m-%d %H:%M}       {str(r['side']):5s} {('won' if r['won'] else 'lost') if r['won'] is not None else 'open':6s} {r['got']:7.0f} {r['after']:20.0f} {r['total']:19.0f} {'yes' if r['thru'] else 'no':>8s} {100*r['sure']/r['want']:15.0f}% {100*r['may']/r['want']:7.0f}%")
n = len(rows); full = sum(r["sure"] >= r["want"] - 1e-6 for r in rows); fullm = sum(r["may"] >= r["want"] - 1e-6 for r in rows); done = [r for r in rows if r["won"] is not None]
pnl = lambda r, k: r[k] * ((1 - r["px"]) if r["won"] else -r["px"])
print(f"\n{n} real fills.  At ${STAKE:.0f}: filled completely for sure in {full} ({full/n:.0%}), at most in {fullm} ({fullm/n:.0%}); average share of the stake filled: {100*sum(r['sure']/r['want'] for r in rows)/n:.0f}% for sure, {100*sum(r['may']/r['want'] for r in rows)/n:.0f}% at most")
if done:
    w = [r for r in done if r["won"]]; l = [r for r in done if not r["won"]]
    print(f"winners: {len(w)}, average fill for sure {100*sum(r['sure']/r['want'] for r in w)/max(len(w),1):.0f}%   losers: {len(l)}, average fill for sure {100*sum(r['sure']/r['want'] for r in l)/max(len(l),1):.0f}%")
    print(f"profit on these trades at ${STAKE:.0f}: if every one had filled completely ${sum(pnl(r, 'want') for r in done):+.0f};  with the fill that is sure ${sum(pnl(r, 'sure') for r in done):+.0f};  with the most possible ${sum(pnl(r, 'may') for r in done):+.0f}")
