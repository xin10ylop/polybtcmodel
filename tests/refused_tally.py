"""What happened to every rule-A signal the live bot produced, and how the REFUSED ones (post-only bid rejected as crossing) would have done
if the offer had been taken at 0.50 plus the taker fee.  Reads the bot's own logs; changes nothing.
  python tests/refused_tally.py [/home/user/paper] [since, e.g. 2026-11-11T16:00]"""
import os, sys, json, datetime as dt
D = sys.argv[1] if len(sys.argv) > 1 else "/home/user/paper"; since = dt.datetime.fromisoformat(sys.argv[2] + "+00:00").timestamp() if len(sys.argv) > 2 else 0.0
def rd(f): p = os.path.join(D, f); return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
sig = {r["T"]: r for r in rd("forecasts.jsonl") if r.get("strategy") == "A" and str(r.get("action", "")).startswith("BUY") and r.get("ts", 0) >= since}
orders = rd("live_orders.jsonl"); placed = {o["T"]: o for o in orders if o.get("event") == "placed" and o.get("kind") == "buy" and o.get("T") in sig}
trades = rd("live_trades.jsonl"); filled = {}
for t in trades:
    if t.get("event") == "fill" and t.get("kind") == "buy" and t.get("T") in sig: filled[t["T"]] = filled.get(t["T"], 0.0) + t.get("size", 0.0)
settle = {t["T"]: t for t in trades if t.get("event") == "settle" and t.get("T") in sig}
paper = {t["T"]: t for t in rd("paper_trades.jsonl") if t.get("strategy") == "A" and t.get("T") in sig and t.get("exit_kind") == "SETTLE"}
refused = [T for T in sig if T not in placed]; unfilled = [T for T in placed if T not in filled]
def wl(Ts): r = [paper[T]["exit"] > 0.5 for T in Ts if T in paper]; return sum(r), len(r) - sum(r), len(Ts) - len(r)
n = len(sig)
if not n: print("no rule-A signals in", D, "since", sys.argv[2] if len(sys.argv) > 2 else "the start of the log"); sys.exit(0)
print(f"signals: {n}   placed: {len(placed)} ({len(placed)/n:.0%})   refused: {len(refused)} ({len(refused)/n:.0%})   filled: {len(filled)} ({len(filled)/n:.0%} of signals)   placed but not filled: {len(unfilled)}")
w, l, u = wl(list(filled)); rw = sum(1 for T in filled if T in settle and settle[T].get("won")); rl = sum(1 for T in filled if T in settle and not settle[T].get("won"))
print(f"FILLED (real money): won {rw}, lost {rl}, still open {len(filled) - rw - rl}; real P&L ${sum(t.get('pnl', 0.0) for t in trades if t.get('T') in sig):+.2f}")
w, l, u = wl(refused); fee = 0.07 * 0.5 * 0.5
print(f"REFUSED: would have won {w}, lost {l}, not resolved yet {u}" + (f"  -> win rate {w/(w+l):.0%}; taken at 0.50 + fee, per 10 shares: ${(w * (5 - fee * 10) - l * (5 + fee * 10)):+.2f} in total (break-even win rate 51.75 %)" if w + l else ""))
w, l, u = wl(unfilled); print(f"PLACED BUT NOT FILLED: would have won {w}, lost {l}, not resolved yet {u}" + (f"  -> win rate {w/(w+l):.0%}" if w + l else ""))
