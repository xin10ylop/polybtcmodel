"""What the bot WOULD have made on the signals it logged while no real orders went through (no balance, halted, ...), up to now.
Signals, sides, stake multipliers and T-45 confirmations come from the bot's own log (PAPER_DIR/forecasts.jsonl); fills are simulated on the real Polymarket
tape of each window (bid 0.50 from T-118; refused when our side was last seen on offer at 0.50 or lower; filled once somebody sells below 0.50 or $50 has been
sold at 0.50; cancelled at T-44 when the T-45 model disagreed, else at T-15); results from Polymarket's resolution.  Windows with a REAL fill are reported
from the real log instead.  Reads logs and public data only.
  python tests/would_have.py [/home/user/paper] [since, e.g. 2026-11-15T04:20] [base stake $, default 10] [min multiple, default 1] [strong] [x2]
  With a min multiple above 1 only signals with |score| >= that multiple of the threshold count, at a flat stake; 'strong' keeps a bid past T-44 only
  when the T-45 model agrees AND |score45| >= its top-30 % threshold (0.0957).  Only trading hours (weekends + weekday 16-24 UTC) are counted, and every
  window is simulated the same way (real fills are counted too, at the same stake), so different rules compare like for like."""
import os, sys, json, time, datetime as dt, requests
from concurrent.futures import ThreadPoolExecutor
D = sys.argv[1] if len(sys.argv) > 1 else "/home/user/paper"; since = dt.datetime.fromisoformat(sys.argv[2] + "+00:00").timestamp() if len(sys.argv) > 2 else 0.0; STAKE = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
MINX = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0; STRONG = "strong" in sys.argv[5:]; X2 = "x2" in sys.argv[5:]; THR45 = 0.09569651720399566      # x2: with a min multiple, double the stake at 2.2x the threshold
def hours_ok(T): d = dt.datetime.fromtimestamp(T, dt.UTC); return d.weekday() >= 5 or d.hour >= 16
def rd(f): p = os.path.join(D, f); return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
fc = rd("forecasts.jsonl"); sig = {r["T"]: r for r in fc if r.get("strategy") == "A" and str(r.get("action", "")).startswith("BUY") and r["T"] >= since and r["T"] + 300 < time.time() - 420 and hours_ok(r["T"])
       and r.get("new_score") is not None and abs(r["new_score"]) >= MINX * float(r.get("new_thr") or 0.0700208716834402)}
conf = {r["T"]: r for r in fc if r.get("strategy") == "A_conf"}; tr = rd("live_trades.jsonl"); real = {}
for t in tr:
    if t.get("strategy") == "A" and t.get("event") == "fill" and t.get("kind") == "buy": real[t["T"]] = real.get(t["T"], 0.0) + t.get("size", 0.0)
settle = {t["T"]: t for t in tr if t.get("strategy") == "A" and t.get("event") == "settle"}
def get(url, params):
    for k in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429: time.sleep(2 + k); continue
            r.raise_for_status(); return r.json()
        except Exception as e: err = e; time.sleep(1 + k)
    raise RuntimeError(str(err)[:100])
def fetch(T):
    try:
        m = get("https://gamma-api.polymarket.com/events", dict(slug=f"btc-updown-5m-{T}"))[0]["markets"][0]; p = json.loads(m.get("outcomePrices") or "[]"); up = (float(p[0]) > 0.5) if p and (float(p[0]) > 0.99 or float(p[0]) < 0.01) else None; pr = []; cur = None
        while True:
            d = get("https://data-api.polymarket.com/v2/trades", dict(condition=m["conditionId"], limit=1000, **({"cursor": cur} if cur else {})))
            pr += [(int(r["timestamp"]) - T, r["side"], int(r["outcome_index"]), float(r["price"]), float(r["size"])) for r in d.get("data") or [] if -200 <= int(r["timestamp"]) - T <= 0]; pg = d.get("pagination") or {}; cur = pg.get("next_cursor")
            if not pg.get("has_more") or not cur: break
        return T, up, sorted(pr)
    except Exception: return T, None, []
with ThreadPoolExecutor(6) as ex: R = {T: (up, pr) for T, up, pr in ex.map(fetch, sorted(sig))}
rows = []
for T in sorted(sig):
    up, pr = R[T]; side = sig[T]["action"].split()[1]; s = 0 if side == "UP" else 1; mult = float(sig[T].get("stake_mult") or 1.0) if MINX <= 1.0 else (2.0 if X2 and abs(sig[T]["new_score"]) >= 2.2 * float(sig[T].get("new_thr") or 0.0700208716834402) else 1.0)
    if up is None: continue
    s45 = conf.get(T, {}).get("score45"); keep = True
    if s45 is not None and s45 == s45: keep = ((s45 > 0) == (side == "UP")) and (abs(s45) >= THR45 if STRONG else True)
    won = up if s == 0 else not up; stop = -15 if keep else -44; refused = False; last = None; cum = 0.0; filled = False
    for rel, sd, oi, px, sz in pr:
        po = px if oi == s else 1 - px; took = (oi == s and sd == "BUY") or (oi != s and sd == "SELL")       # took = somebody bought our side
        if -178 <= rel < -118: last = (took, po)
        elif -118 <= rel <= stop and not took and po <= 0.50 + 1e-6:
            cum += po * sz; filled = filled or po < 0.495 or cum >= 50.0
    if last is not None and ((last[0] and last[1] <= 0.50 + 1e-6) or (not last[0] and last[1] < 0.495)): refused = True; filled = False
    isreal = T in real; pnl = (1 if won else -1) * STAKE * mult if filled else 0.0
    rows.append(dict(T=T, day=dt.datetime.fromtimestamp(T, dt.UTC).strftime("%m-%d"), won=won, filled=filled, real=isreal and filled, refused=refused, mult=mult, pnl=pnl))
if not rows: print("no rule-A signals in", D, "since", sys.argv[2] if len(sys.argv) > 2 else "the start"); sys.exit(0)
def line(lbl, rs):
    f = [r for r in rs if r["filled"]]; w = sum(r["won"] for r in f)
    return f"{lbl:22s} signals {len(rs):4d} (right {100*sum(r['won'] for r in rs)/len(rs):5.1f} %) | fills {len(f):4d}: won {w:3d}, lost {len(f)-w:3d} -> {100*w/max(len(f),1):5.1f} % | profit ${sum(r['pnl'] for r in f):+8.2f} | real fills {sum(r['real'] for r in f)}, doubled {sum(r['mult'] > 1 for r in f)}"
print(f"RULE: {'every signal' if MINX <= 1 else f'signals from {MINX:g}x the threshold'}, T-45 check {'must agree strongly' if STRONG else 'must agree'}; trading hours since {sys.argv[2] if len(sys.argv) > 2 else 'the start'} (UTC); stake ${STAKE:g}" + (" (strongest doubled as logged)" if MINX <= 1 else (" doubled at 2.2x the threshold" if X2 else " flat")) + "; every window simulated on the Polymarket tape")
for d in sorted({r["day"] for r in rows}): print(line(d, [r for r in rows if r["day"] == d]))
print(line("TOTAL", rows)); sim = [r for r in rows if not r["real"]]; print(line("  of which simulated", sim)); rl = [r for r in rows if r["real"]]
if rl: print(line("  of which real", rl))
