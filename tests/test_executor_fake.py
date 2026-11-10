"""Exercises the REAL-order code path of polybtc/live_executor.py (DRY_RUN=0) against a fake exchange: fills, partial fills, cancels,
rejections, restart recovery, shutdown.  No network, no keys.   python tests/test_executor_fake.py [polybtc|polyeth]"""
import os, sys, time, json, tempfile, importlib
PKG = sys.argv[1] if len(sys.argv) > 1 else "polybtc"; ROOT = {"polybtc": "/home/user/polybtcmodel/polybtc", "polyeth": "/home/user/polyethmodel/polyeth"}[PKG]
os.environ.update(PAPER_DIR=tempfile.mkdtemp(prefix="exec_test_"), DRY_RUN="0", STAKE_USD="5", MAX_OPEN="3", DAILY_LOSS_LIMIT="-25", ALLOW_TAKER="0", CANCEL_ALL_ON_START="0", POLYMARKET_RELAYER_API_KEY="x")
sys.path.insert(0, ROOT); import live_executor as LE
TPX = LE.CFG_TP["A"]

class Fake:
    """Shared fake exchange."""
    book = {}; n = 0; reject_sells = 0; crossing = False; fail_next = None; refuse_cancel = 0; ghost = None; redeemed = []; cancel_all_calls = 0
    def __init__(self): pass
    def limit(self, token, side, price, size, post_only=False):
        if Fake.ghost:      # the exchange accepts the order but the answer is lost (timeout): the order exists, the caller sees an error
            Fake.n += 1; Fake.book[f"o{Fake.n}"] = dict(token=token, side=side, price=price, size=size, matched=0.0, status="live"); m = Fake.ghost; Fake.ghost = None; raise TimeoutError(m)
        if Fake.fail_next: m = Fake.fail_next; Fake.fail_next = None; raise RuntimeError(m)
        if post_only and Fake.crossing: raise RuntimeError("invalid post-only order: order crosses book")
        if side == "SELL" and Fake.reject_sells > 0: Fake.reject_sells -= 1; raise RuntimeError("order rejected: not_enough_balance")
        Fake.n += 1; oid = f"o{Fake.n}"; Fake.book[oid] = dict(token=token, side=side, price=price, size=size, matched=0.0, status="live"); return dict(order_id=oid, status="live", trade_ids=[])
    def cancel(self, oid):
        if Fake.refuse_cancel > 0: Fake.refuse_cancel -= 1; raise RuntimeError("cancel refused: busy")
        o = Fake.book[oid]
        if o["status"] == "live": o["status"] = "canceled"
        return True
    def open_orders(self, token): return [k for k, o in Fake.book.items() if o["token"] == token and o["status"] == "live"]
    def cancel_all(self): Fake.cancel_all_calls += 1; [self.cancel(k) for k in Fake.book]; return True
    def status(self, oid): o = Fake.book[oid]; return o["status"], o["matched"]
    def redeem(self, cond): Fake.redeemed.append(cond); return "0xtx"
def fill(oid, qty):
    o = Fake.book[oid]; o["matched"] = min(o["size"], o["matched"] + qty)
    if o["matched"] >= o["size"] - 1e-9: o["status"] = "matched"
LE.SDK = Fake
def new_exec(): return LE.LiveExecutor()
def poll(ex, now, outcome=lambda T: None): ex._last_poll = 0; ex.poll(now, outcome)
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else "")); RES.append(cond)
RES = []; now = time.time(); T = int(now) // 300 * 300 + 600        # a window comfortably in the future; cancel time T-40
ex = new_exec()
# 1. full fill -> take-profit posted -> sold
oid = ex.buy("A", T, "UP", 0.50, time.time() + 30, "tokU", "tokD", "cond1", TPX); check("buy placed, 10 shares at 0.50", oid is not None and Fake.book[oid]["size"] == 10.0 and Fake.book[oid]["price"] == 0.50, str(Fake.book.get(oid)))
fill(oid, 10); poll(ex, time.time()); sells = [k for k, o in Fake.book.items() if o["side"] == "SELL"]
check("full fill books a position and rests the take-profit for all shares", len(sells) == 1 and Fake.book[sells[0]]["price"] == TPX and Fake.book[sells[0]]["size"] == 10.0 and oid not in ex.orders, str([Fake.book[k] for k in sells]))
fill(sells[0], 10); poll(ex, time.time()); p = ex.positions[("A", T)]
check("take-profit fill closes the position with the right P&L", p["settled"] and abs(p["pnl"] - (TPX - 0.5) * 10) < 1e-9 and abs(ex.day_pnl - (TPX - 0.5) * 10) < 1e-9, f"pnl {p['pnl']:.2f} day {ex.day_pnl:.2f}")
# 2. no fill -> cancelled at the deadline
T2 = T + 300; oid = ex.buy("A", T2, "DOWN", 0.50, time.time() + 30, "tokU2", "tokD2", "cond2", TPX); poll(ex, time.time()); check("unfilled bid still resting before its cancel time", Fake.book[oid]["status"] == "live")
poll(ex, time.time() + 31); poll(ex, time.time() + 32); check("unfilled bid cancelled at the deadline and forgotten, no position", Fake.book[oid]["status"] == "canceled" and oid not in ex.orders and ("A", T2) not in ex.positions)
# 3. partial fill of 3 shares (< 5): no take-profit, settles and queues the redeem
T3 = T + 600; oid = ex.buy("A", T3, "UP", 0.50, time.time() + 30, "tokU3", "tokD3", "cond3", TPX); fill(oid, 3); poll(ex, time.time()); poll(ex, time.time() + 31); poll(ex, time.time() + 32)
p = ex.positions.get(("A", T3)); check("3-share partial fill: position of 3, no sell below the 5-share minimum, rest cancelled", p is not None and p["size"] == 3 and p["sell_orders"] == [] and oid not in ex.orders, str(p))
poll(ex, T3 + 365, lambda t: 1); check("held to settlement: booked as a win and queued for redemption", p["settled"] and abs(p["pnl"] - 1.5) < 1e-9 and any(q["cond"] == "cond3" for q in ex.redeem_q), f"pnl {p['pnl']}")
# 4. take-profit rejected twice, then accepted on a retry
T4 = T + 900; Fake.reject_sells = 2; oid = ex.buy("A", T4, "UP", 0.50, time.time() + 30, "tokU4", "tokD4", "cond4", TPX); fill(oid, 10); poll(ex, time.time()); p = ex.positions[("A", T4)]
c0 = len(p["sell_orders"]); poll(ex, time.time() + 4); poll(ex, time.time() + 8); check("rejected take-profit is retried until it rests", c0 == 0 and len(p["sell_orders"]) == 1 and p["resting"] == 10.0, f"after first try {c0}, now {p['sell_orders']}")
# 5. take-profit vanishes (cancelled outside the bot) -> posted again
Fake.book[p["sell_orders"][0]]["status"] = "canceled"; poll(ex, time.time() + 12); poll(ex, time.time() + 16)
live_sells = [k for k in p["sell_orders"] if Fake.book[k]["status"] == "live"]; check("a take-profit that disappears unfilled is posted again", len(live_sells) == 1 and abs(p["resting"] - 10.0) < 1e-9, f"{p['sell_orders']} resting {p['resting']}")
# 6. late decision: refused
r = ex.buy("A", T + 1200, "UP", 0.50, time.time() + 0.4, "a", "b", "c", TPX); check("bid refused when less than 1 s is left before its cancel time", r is None)
# 7. restart: a bid left resting fills while the bot is down -> recovered, take-profit posted; other bot's order untouched
T7 = T + 1500; oid = ex.buy("A", T7, "DOWN", 0.50, time.time() + 30, "tokU7", "tokD7", "cond7", TPX); other = Fake().limit("otherbot", "BUY", 0.5, 10)["order_id"]
day_before = ex.day_pnl; del ex; fill(oid, 10); ex = new_exec(); poll(ex, time.time()); p = ex.positions.get(("A", T7))
check("restart: fill made while down is booked and its take-profit rests", p is not None and p["size"] == 10 and len(p["sell_orders"]) == 1, str(p))
check("restart: the other bot's resting order is not cancelled, no cancel_all", Fake.book[other]["status"] == "live" and Fake.cancel_all_calls == 0)
check("restart: today's realised P&L restored", abs(ex.day_pnl - day_before) < 1e-9, f"{ex.day_pnl} vs {day_before}")
# 8. shutdown from inside the lock (SIGTERM during poll) must not deadlock and cancels only own orders
with ex.lock: ex.shutdown()
check("shutdown inside the lock returns and cancels own orders only", Fake.book[other]["status"] == "live" and all(Fake.book[k]["status"] != "live" for k in ex.orders))
# cancel_now: the trader withdraws a resting bid before its deadline (T-45 model disagrees with an earlier decision); a filled part keeps its sell
exq = ex; TQ = T + 6000; oq = exq.buy("A", TQ, "UP", 0.50, time.time() + 100, "tokUq", "tokDq", "condq", TPX); poll(exq, time.time()); exq.cancel_now("A", TQ); poll(exq, time.time()); poll(exq, time.time() + 1)
check("cancel_now withdraws the resting bid long before its deadline, no position", Fake.book[oq]["status"] == "canceled" and oq not in exq.orders and ("A", TQ) not in exq.positions)
TQ2 = TQ + 300; oq = exq.buy("A", TQ2, "DOWN", 0.50, time.time() + 100, "tokUq2", "tokDq2", "condq2", TPX); fill(oq, 6); poll(exq, time.time()); exq.cancel_now("A", TQ2); poll(exq, time.time()); poll(exq, time.time() + 1); pq = exq.positions.get(("A", TQ2))
check("cancel_now after a partial fill: the rest is cancelled, the 6 filled shares keep their sell order", Fake.book[oq]["status"] == "canceled" and pq is not None and pq["size"] == 6 and len(pq["sell_orders"]) == 1, str(pq))
# ALLOW_TAKER: a post-only bid refused as crossing is re-sent once as a plain limit; any other failure is never re-sent (no duplicate order)
TK = T + 9000; n0 = len(Fake.book); Fake.crossing = True; LE.ALLOW_TAKER = False; mo = LE.MAX_OPEN; LE.MAX_OPEN = 99; r0 = ex.buy("A", TK, "UP", 0.50, time.time() + 100, "tU", "tD", "ck", TPX)
check("crossing bid with ALLOW_TAKER=0: nothing placed", r0 is None and len(Fake.book) == n0)
LE.ALLOW_TAKER = True; r1 = ex.buy("A", TK, "UP", 0.50, time.time() + 100, "tU", "tD", "ck", TPX)
check("crossing bid with ALLOW_TAKER=1: exactly one plain limit at 0.50, flagged taker", r1 is not None and len(Fake.book) == n0 + 1 and Fake.book[r1]["price"] == 0.50 and ex.orders[r1]["taker"] is True)
fill(r1, 10); poll(ex, time.time()); pk = ex.positions[("A", TK)]; check("taker fill books the fee", abs(pk["pnl"] + LE.taker_fee(0.50, 10)) < 1e-9, f"pnl {pk['pnl']:.4f}")
Fake.crossing = False; n1 = len(Fake.book); Fake.fail_next = "ReadTimeout: the request timed out"; r2 = ex.buy("A", TK + 300, "UP", 0.50, time.time() + 100, "tU2", "tD2", "ck2", TPX)
check("a timeout on the first attempt is NOT re-sent (no duplicate order), even with ALLOW_TAKER=1", r2 is None and len(Fake.book) == n1); LE.ALLOW_TAKER = False; LE.MAX_OPEN = mo
# refused cancel: retried until the order is off the book; lost answer on placement: the untracked order is found and cancelled
LE.MAX_OPEN = 99; TR = T + 12000; orf = ex.buy("A", TR, "UP", 0.50, time.time() + 30, "tR", "tRd", "cr", TPX); Fake.refuse_cancel = 1; poll(ex, time.time() + 31); st1 = Fake.book[orf]["status"]
ex.orders[orf]["cancel_ts"] -= 3; poll(ex, time.time() + 32); poll(ex, time.time() + 33)
check("a refused cancel is retried until the bid is off the book", st1 == "live" and Fake.book[orf]["status"] == "canceled" and orf not in ex.orders, f"first {st1}, then {Fake.book[orf]['status']}")
nb = len(Fake.book); Fake.ghost = "read timed out"; rg = ex.buy("A", TR + 300, "UP", 0.50, time.time() + 100, "tG", "tGd", "cg", TPX); gid = [k for k, o in Fake.book.items() if o["token"] == "tG"]
for sw in ex.sweeps: sw["at"] = [time.time() - 1]
poll(ex, time.time())
check("placement answer lost: the order that exists untracked on the book is found and cancelled", rg is None and len(gid) == 1 and Fake.book[gid[0]]["status"] == "canceled", str([Fake.book[g] for g in gid])); LE.MAX_OPEN = 3
# 9. risk: daily loss limit halts new trades
ex.day_pnl = -26.0; r = ex.buy("A", T + 1800, "UP", 0.50, time.time() + 30, "a", "b", "c", TPX); check("daily loss limit stops new orders", r is None and ex.halted)
print(f"\n{PKG}: {sum(RES)}/{len(RES)} checks passed"); sys.exit(0 if all(RES) else 1)
