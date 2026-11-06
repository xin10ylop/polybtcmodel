"""Live execution layer for the paper trader's decisions on Polymarket (official Python SDK `polymarket-client`).

The paper trader keeps deciding and keeps its own simulated book; when LIVE=1 it also hands every decision to this
executor, which places the real resting orders and tracks the real fills:
  A : BUY limit at 0.50 on the model's side at T-45, cancelled at T-40 if unfilled; on fill a SELL limit at 0.85 (GTC);
      whatever is unsold at the close is held to settlement and redeemed.
  B : BUY limit at the last print at T+1.6, cancelled after 10 s; on fill SELL limit at 0.85; else settle + redeem.
  D : BUY limit at the last print at T+90, cancelled after 10 s; no sell; settle + redeem.
Sizing: STAKE_USD per trade at the entry price, rounded to the market's size precision and at least MIN_SHARES.
Risk: MAX_OPEN positions, DAILY_LOSS_LIMIT (stop opening new trades for the UTC day once realised P&L is below it),
and a kill file (touch /home/user/paper/STOP) that cancels all open orders and stops new ones.
DRY_RUN=1 logs every order it would place without touching the account.
Fees: every order is post-only (maker), so no taker fee is ever paid; an order that would cross the book is not placed
(ALLOW_TAKER=1 re-sends it crossing and books the fee). The tally reports fees paid, which should read $0.00.

Environment: POLYMARKET_PRIVATE_KEY (signer key), POLYMARKET_WALLET_ADDRESS (account wallet), and for redemption
POLYMARKET_RELAYER_API_KEY + POLYMARKET_RELAYER_API_KEY_ADDRESS (polymarket.com -> Settings -> API Keys -> Relayer).
Everything is logged to PAPER_DIR/live_orders.jsonl and live_trades.jsonl, and live.log.
"""
import os, json, time, math, datetime as dt, threading, traceback

STAKE_USD = float(os.environ.get("STAKE_USD", "10"))
MIN_SHARES = 5.0
MAX_OPEN = int(os.environ.get("MAX_OPEN", "4"))
DAILY_LOSS_LIMIT = float(os.environ.get("DAILY_LOSS_LIMIT", "-100"))
DRY_RUN = os.environ.get("DRY_RUN", "1") == "1"
ALLOW_TAKER = os.environ.get("ALLOW_TAKER", "0") == "1"   # default: post-only everywhere, a fee can never be paid
PAPER = os.environ.get("PAPER_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")
STOP_FILE = os.path.join(PAPER, "STOP")
CFG_TP = {"A": 0.85, "B": 0.85, "D": None}   # sell limit per strategy (mirrors paper_trader CFG_*; used only when re-posting after a restart)
LOG = open(os.path.join(PAPER, "live.log"), "a")


def log(*a):
    s = f"{dt.datetime.now(dt.UTC):%H:%M:%S} [live{'/dry' if DRY_RUN else ''}] " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def taker_fee(price, size):
    """Polymarket taker fee on the 5-minute markets: 0.07 * p * (1 - p) per share (feeRateBps 700, exponent 1); makers pay nothing."""
    return 0.07 * price * (1.0 - price) * size


class SDK:
    """Thin wrapper over the official SDK so the executor can be exercised without it (DRY_RUN) and so that
    method names live in one place. Fill in from docs.polymarket.com/trading/manage-orders if the SDK differs."""
    def __init__(self):
        self.client = None
        if DRY_RUN: return
        from polymarket import SecureClient, RelayerApiKey   # pip install polymarket-client
        kw = dict(private_key=os.environ["POLYMARKET_PRIVATE_KEY"], wallet=os.environ["POLYMARKET_WALLET_ADDRESS"])
        if os.environ.get("POLYMARKET_RELAYER_API_KEY"):
            kw["api_key"] = RelayerApiKey(key=os.environ["POLYMARKET_RELAYER_API_KEY"], address=os.environ["POLYMARKET_RELAYER_API_KEY_ADDRESS"])
        self.client = SecureClient.create(**kw)
        log("sdk ready: wallet", os.environ["POLYMARKET_WALLET_ADDRESS"], "type", getattr(self.client, "wallet_type", "?"))
        try: self.client.setup_trading_approvals(); log("trading approvals ok (pUSD + outcome tokens)")
        except Exception as e: log("setup_trading_approvals:", repr(e)[:160])

    def limit(self, token_id, side, price, size, post_only=False):
        if DRY_RUN: return dict(order_id=f"dry-{int(time.time()*1000)}", status="live")
        r = self.client.place_limit_order(asset_id=token_id, side=side, price=f"{price:.2f}", size=f"{size:.2f}", post_only=post_only)
        if not getattr(r, "ok", True): raise RuntimeError(f"order rejected: {getattr(r, 'code', '')} {getattr(r, 'message', '')}")
        return dict(order_id=r.order_id, status=r.status, trade_ids=list(getattr(r, "trade_ids", []) or []))

    def cancel(self, order_id):
        if DRY_RUN: return True
        return self.client.cancel_order(order_id=order_id)

    def cancel_all(self):
        if DRY_RUN: return True
        return self.client.cancel_all()

    def status(self, order_id):
        """-> (status, size_matched). Field names per manage-orders docs; adjust there if the SDK names differ."""
        if DRY_RUN: return "live", 0.0
        o = self.client.get_order(order_id=order_id)
        return getattr(o, "status", "?"), float(getattr(o, "size_matched", 0) or 0)

    def redeem(self, condition_id):
        if DRY_RUN: return "dry"
        tx = self.client.redeem_positions(condition_id=condition_id); out = tx.wait()
        return getattr(out, "transaction_hash", "?")


class LiveExecutor:
    def __init__(self):
        self.sdk = SDK(); self.lock = threading.Lock()
        self.orders = {}          # order_id -> dict(strategy, T, side, token, price, size, kind buy|sell, cancel_at, filled)
        self.positions = {}       # (strategy, T) -> dict(side, token, cond, size, entry, sell_order, settled)
        self.day = None; self.day_pnl = 0.0; self.halted = False
        self.fo = open(os.path.join(PAPER, "live_orders.jsonl"), "a"); self.ft = open(os.path.join(PAPER, "live_trades.jsonl"), "a")
        self.pos_file = os.path.join(PAPER, "live_positions.json")
        log(f"executor up: stake ${STAKE_USD}, max open {MAX_OPEN}, daily loss limit ${DAILY_LOSS_LIMIT}, dry_run={DRY_RUN}")
        if not DRY_RUN:
            if not os.environ.get("POLYMARKET_RELAYER_API_KEY"): log("WARNING: no POLYMARKET_RELAYER_API_KEY: winning positions held to settlement will NOT be redeemed automatically")
            try: self.sdk.cancel_all(); log("startup: cancelled every resting order left by a previous run")
            except Exception as e: log(f"startup cancel_all FAILED: {repr(e)[:120]} - check the Polymarket UI for resting orders")
        self._load_positions()

    # ---------------------------------------------------------------- persistence (a restart must not forget shares it holds)
    def _save_positions(self):
        try:
            tmp = self.pos_file + ".tmp"
            json.dump({f"{k[0]}|{k[1]}": p for k, p in self.positions.items() if not p["settled"]}, open(tmp, "w")); os.replace(tmp, self.pos_file)
        except Exception as e: log(f"save positions failed: {repr(e)[:100]}")

    def _load_positions(self):
        if not os.path.exists(self.pos_file): return
        try: saved = json.load(open(self.pos_file))
        except Exception as e: log(f"load positions failed: {repr(e)[:100]}"); return
        now = time.time()
        for key, p in saved.items():
            st, T = key.split("|"); T = int(T); p["sell_order"] = None; p["sell_orders"] = []; p["next_check"] = 0.0
            self.positions[(st, T)] = p; held = p["size"] - p["sold"]
            log(f"[{st}] restored position from a previous run: {p['side']} x{held:.2f} @{p['entry']:.2f} (window {dt.datetime.fromtimestamp(T, dt.UTC):%H:%M})")
            tp = CFG_TP.get(st)
            if held >= 0.01 and tp is not None and now < T + 300 and not DRY_RUN:   # re-post the sell for the shares still held
                try:
                    r, taker = self._place(p["token"], "SELL", tp, held)
                    p["sell_order"] = r["order_id"]; p["sell_orders"] = [r["order_id"]]
                    self.orders[r["order_id"]] = dict(strategy=st, T=T, side=p["side"], token=p["token"], cond=p["cond"], price=tp, size=held, kind="sell", cancel_at=None, filled=0.0, tp=None, placed=now, taker=taker)
                    log(f"[{st}] re-posted SELL limit @{tp:.2f} x{held:.2f} order {r['order_id']}")
                except Exception as e: log(f"[{st}] re-post SELL failed ({repr(e)[:100]}); holding to settlement")
        self._save_positions()

    # ---------------------------------------------------------------- helpers
    def _rec(self, f, **d):
        d["ts"] = round(time.time(), 2); f.write(json.dumps(d) + "\n"); f.flush()

    def _roll_day(self):
        d = dt.datetime.now(dt.UTC).date().isoformat()
        if d != self.day: self.day, self.day_pnl, self.halted = d, 0.0, False

    def _can_open(self, strategy):
        self._roll_day()
        if os.path.exists(STOP_FILE): log("STOP file present: no new orders"); return False
        if self.halted: return False
        if self.day_pnl <= DAILY_LOSS_LIMIT: self.halted = True; log(f"daily loss limit reached ({self.day_pnl:+.2f}): no new trades today"); return False
        open_pos = len([p for p in self.positions.values() if not p["settled"]]) + len([o for o in self.orders.values() if o["kind"] == "buy" and not o["filled"]])
        if open_pos >= MAX_OPEN: log(f"max open positions ({MAX_OPEN}) reached: skip {strategy}"); return False
        return True

    @staticmethod
    def _size(price):
        return max(MIN_SHARES, math.floor(STAKE_USD / price * 100) / 100)

    def _place(self, token, side, price, size):
        """Every order is post-only: it rests on the book and can only ever be filled as a maker (no fee). If the exchange
        rejects it because the price would cross the book, the order is NOT placed (a BUY is skipped, a SELL falls back to
        holding to settlement) unless ALLOW_TAKER=1, in which case the same limit is re-sent crossing and the taker fee
        0.07 p(1-p) per share is booked on the fill. Returns (order, taker_flag)."""
        try:
            return self.sdk.limit(token, side, price, size, post_only=True), False
        except Exception as e:
            if not ALLOW_TAKER:
                log(f"post-only {side} @{price:.2f} rejected ({repr(e)[:100]}); not placed (ALLOW_TAKER=0: never pay a fee)"); raise
            log(f"post-only {side} @{price:.2f} rejected ({repr(e)[:100]}); placing as a crossing limit (ALLOW_TAKER=1)")
            return self.sdk.limit(token, side, price, size, post_only=False), True

    # ---------------------------------------------------------------- decisions from the paper trader
    def buy(self, strategy, T, side, price, cancel_at, token_up, token_down, condition_id, tp):
        """Rest a BUY limit for `strategy` on `side` at `price`; cancel at epoch `cancel_at` if unfilled; sell limit `tp` after the fill (None = hold)."""
        with self.lock:
            if not self._can_open(strategy): return None
            token = token_up if side == "UP" else token_down; size = self._size(price)
            try:
                r, taker = self._place(token, "BUY", price, size)
            except Exception as e:
                log(f"[{strategy}] BUY {side} @{price:.2f} x{size} FAILED: {repr(e)[:160]}"); return None
            oid = r["order_id"]; self.orders[oid] = dict(strategy=strategy, T=T, side=side, token=token, cond=condition_id, price=price, size=size, kind="buy", cancel_at=cancel_at, filled=0.0, tp=tp, placed=time.time(), taker=taker)
            self._rec(self.fo, event="placed", order_id=oid, strategy=strategy, T=T, side=side, price=price, size=size, kind="buy", status=r.get("status"), taker=taker)
            log(f"[{strategy}] BUY {side} @{price:.2f} x{size} order {oid} status {r.get('status')}{' TAKER' if taker else ''}")
            if r.get("status") == "matched": self._on_fill(oid, size)
            return oid

    def _on_fill(self, oid, matched):
        o = self.orders[oid]; new = matched - o["filled"]
        if new <= 0: return
        o["filled"] = matched
        key = (o["strategy"], o["T"])
        p = self.positions.setdefault(key, dict(side=o["side"], token=o["token"], cond=o["cond"], size=0.0, entry=o["price"], sell_order=None, sell_orders=[], settled=False, sold=0.0, pnl=0.0, next_check=0.0))
        p["size"] += new; fee = taker_fee(o["price"], new) if o.get("taker") else 0.0; p["pnl"] -= fee; self.day_pnl -= fee; self._save_positions()
        self._rec(self.ft, event="fill", order_id=oid, strategy=o["strategy"], T=o["T"], side=o["side"], price=o["price"], size=new, kind="buy", fee=round(fee, 4), pnl=round(-fee, 4))
        log(f"[{o['strategy']}] FILLED {o['side']} @{o['price']:.2f} x{new:.2f} (T{time.time()-o['T']:+.1f}s)" + (f" taker fee {fee:.2f}" if fee else ""))
        if o["tp"] is not None and new >= 0.01:          # a sell limit for exactly the shares just filled (a later partial fill gets its own)
            try:
                r, taker = self._place(o["token"], "SELL", o["tp"], new)
                p["sell_order"] = p["sell_order"] or r["order_id"]; p["sell_orders"].append(r["order_id"])
                self.orders[r["order_id"]] = dict(strategy=o["strategy"], T=o["T"], side=o["side"], token=o["token"], cond=o["cond"], price=o["tp"], size=new, kind="sell", cancel_at=None, filled=0.0, tp=None, placed=time.time(), taker=taker)
                self._rec(self.fo, event="placed", order_id=r["order_id"], strategy=o["strategy"], T=o["T"], side=o["side"], price=o["tp"], size=new, kind="sell", status=r.get("status"), taker=taker)
                log(f"[{o['strategy']}] SELL limit @{o['tp']:.2f} x{new:.2f} order {r['order_id']}{' TAKER' if taker else ''}"); self._save_positions()
            except Exception as e:
                log(f"[{o['strategy']}] SELL limit not placed ({repr(e)[:120]}); holding to settlement")

    def _on_sell_fill(self, oid, matched):
        o = self.orders[oid]; new = matched - o["filled"]
        if new <= 0: return
        o["filled"] = matched; key = (o["strategy"], o["T"]); p = self.positions[key]
        p["sold"] += new; fee = taker_fee(o["price"], new) if o.get("taker") else 0.0; pnl = (o["price"] - p["entry"]) * new - fee; p["pnl"] += pnl; self.day_pnl += pnl
        self._rec(self.ft, event="fill", order_id=oid, strategy=o["strategy"], T=o["T"], side=o["side"], price=o["price"], size=new, kind="sell", fee=round(fee, 4), pnl=round(pnl, 4))
        log(f"[{o['strategy']}] SOLD @{o['price']:.2f} x{new:.2f}  pnl {pnl:+.2f}  (day {self.day_pnl:+.2f})")
        if p["sold"] >= p["size"] - 1e-6: p["settled"] = True
        self._save_positions()

    # ---------------------------------------------------------------- periodic maintenance (called from the trader loop)
    def poll(self, now, outcome_fn):
        if now - getattr(self, "_last_poll", 0) < 0.5: return
        self._last_poll = now
        with self.lock:
            try:
                if os.path.exists(STOP_FILE):
                    for oid, o in list(self.orders.items()):
                        if o["kind"] == "buy" and o["filled"] < o["size"]:
                            self.sdk.cancel(oid); self._rec(self.fo, event="cancelled", order_id=oid, reason="stop"); log(f"cancelled {oid} (STOP)"); del self.orders[oid]
                for oid, o in list(self.orders.items()):
                    if DRY_RUN:
                        st, matched = "live", o["filled"]
                    else:
                        try:
                            st, matched = self.sdk.status(oid); o["err"] = 0
                        except Exception as e:                    # one order's lookup must never stall the others
                            o["err"] = o.get("err", 0) + 1
                            if o["err"] in (1, 10, 100): log(f"status {oid} failed x{o['err']}: {repr(e)[:100]}")
                            if o["kind"] == "buy" and o["cancel_at"] is not None and now >= o["cancel_at"]:
                                try: self.sdk.cancel(oid)
                                except Exception: pass
                            continue
                    st = str(st).lower()
                    if o["kind"] == "buy":
                        if matched > o["filled"]: self._on_fill(oid, matched)
                        if o["cancel_at"] is not None and now >= o["cancel_at"] and o["filled"] < o["size"]:
                            try: self.sdk.cancel(oid)
                            except Exception as e: log(f"cancel {oid} failed: {repr(e)[:100]}")
                            if not DRY_RUN:                          # a fill can land in the same instant as the cancel: look once more
                                try:
                                    st2, m2 = self.sdk.status(oid)
                                    if m2 > o["filled"]: self._on_fill(oid, m2)
                                except Exception: pass
                            self._rec(self.fo, event="cancelled", order_id=oid, filled=o["filled"]); log(f"[{o['strategy']}] cancelled unfilled BUY {o['side']} @{o['price']:.2f} ({o['filled']:.2f}/{o['size']:.2f} filled)")
                            del self.orders[oid]
                        elif st in ("cancelled", "canceled", "expired") or (o["filled"] >= o["size"] - 1e-6):
                            del self.orders[oid]
                    else:
                        if matched > o["filled"]: self._on_sell_fill(oid, matched)
                        if o["filled"] >= o["size"] - 1e-6 or st in ("cancelled", "canceled", "expired"): del self.orders[oid]
                # settlement: a position still held 60 s after the close resolves to 1 or 0; cancel its sell, book it, redeem
                for key, p in list(self.positions.items()):
                    if p["settled"] or now < key[1] + 300 + 60 or now < p.get("next_check", 0): continue
                    p["next_check"] = now + 15.0
                    out = outcome_fn(key[1])
                    if out is None: continue
                    for so in list(p.get("sell_orders", [])) + ([p["sell_order"]] if p.get("sell_order") else []):
                        if so in self.orders:
                            try: self.sdk.cancel(so)
                            except Exception as e: log(f"cancel sell {so} failed: {repr(e)[:100]}")
                            del self.orders[so]
                    held = p["size"] - p["sold"]; won = (out == 1) == (p["side"] == "UP"); pnl = ((1.0 if won else 0.0) - p["entry"]) * held
                    p["pnl"] += pnl; self.day_pnl += pnl; p["settled"] = True; self._save_positions()
                    self._rec(self.ft, event="settle", strategy=key[0], T=key[1], side=p["side"], size=held, entry=p["entry"], won=won, pnl=round(pnl, 4))
                    log(f"[{key[0]}] SETTLED {p['side']} x{held:.2f} {'WON' if won else 'LOST'}  pnl {pnl:+.2f}  (day {self.day_pnl:+.2f})")
                    if won and held > 0:
                        try: tx = self.sdk.redeem(p["cond"]); log(f"[{key[0]}] redeemed {p['cond'][:10]}... tx {tx}")
                        except Exception as e: log(f"[{key[0]}] redeem FAILED (retry later from the UI or a redeem sweep): {repr(e)[:160]}")
            except Exception:
                log("executor poll error", traceback.format_exc()[-400:])

    def summary(self):
        tr = [json.loads(l) for l in open(os.path.join(PAPER, "live_trades.jsonl")) if l.strip()] if os.path.exists(os.path.join(PAPER, "live_trades.jsonl")) else []
        pnl = sum(t.get("pnl", 0.0) for t in tr); n = len([t for t in tr if t["event"] in ("settle", "fill") and t.get("kind") != "buy"])
        fees = sum(t.get("fee", 0.0) for t in tr); nt = len([t for t in tr if t.get("fee", 0.0) > 0])
        return f"live: {len([t for t in tr if t['event']=='fill' and t.get('kind')=='buy'])} buys, {n} closes, realised pnl ${pnl:+.2f}, fees paid ${fees:.2f} on {nt} taker fills"
