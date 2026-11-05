"""Paper trader for the Polymarket BTC 5-minute markets (strategies A and B from the README).

Feeds (all public, no keys):
  * Binance spot BTCUSDT 1 s klines  - websocket mirror data-stream.binance.vision (+ REST backfill)
  * OKX BTC-USDT-SWAP trades          - stands in for the Binance perp (geo-blocked here); 1 s bars
  * Polymarket CLOB market websocket  - last prints and books of the UP/DOWN tokens of each window
History for the model's regime features comes from the local Binance archive timeline (btcmovement2
format) up to its last day; the perp gap between the archive and now is synthetic (spot x basis).

Everything is logged to paper/: forecasts.jsonl (every window), paper_trades.jsonl, paper.log.
Run:  python polybtc/paper_trader.py            (foreground; use nohup)
      python polybtc/paper_trader.py --summary  (P&L report from the logs)
"""
import os, sys, json, time, threading, datetime as dt, traceback
import numpy as np, pandas as pd, requests, websocket
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, HERE)
os.environ.setdefault("BTC_DATA_DIR", "/home/user/data")
sys.path.insert(0, "/home/user/btcmovement2")
from btcmodel.timeline import load_timeline, FIELDS
from btcmodel.predictor import Predictor
from btcmodel.regime import regime_features
import score_m2 as S

PAPER = os.environ.get("PAPER_DIR") or os.path.join(REPO, "paper"); os.makedirs(PAPER, exist_ok=True)   # PAPER_DIR=... keeps box logs out of git
LOG = open(os.path.join(PAPER, "paper.log"), "a")
DAY = 86400
CFG_B = dict(delta=0.12, cancel=10, tp=0.85, sl=None, tstop=None, settle=True, slip=0.01, fee=0.07)   # adopted 2026-11-04: sell 0.85 else settle (research/ideas10.py)
CFG_B0 = dict(delta=0.12, cancel=10, tp=0.80, sl=0.10, tstop=240, slip=0.01, fee=0.07)               # previous exits, shadow
CFG_D = dict(decide=90, q=0.90, maxpx=0.90, cancel=10, tp=None, sl=None, tstop=None, settle=True, slip=0.01, fee=0.07)   # high-confidence settlement capture (research/ideas10.py idea 6)
CFG_A = dict(thr=0.05, entry=0.50, tp=0.85, sl=None, tstop=None, settle=True, slip=0.01, fee=0.07)   # adopted 2026-11-04: TP 0.85, no stop, hold to settlement (research/settle_exit_A.py)
CFG_A0 = dict(thr=0.05, entry=0.50, tp=0.80, sl=0.05, tstop=240, slip=0.01, fee=0.07)               # the previous frozen exits, kept as a shadow
CFG_C = dict(delta=0.08, decide=-5, tp=0.80, sl=0.05, tstop=240, slip=0.01, fee=0.07)   # "B five seconds early": bid at the last print from T-5 until the open
LAG = 30
CFGS = {"A": CFG_A, "A0": CFG_A0, "B": CFG_B, "B0": CFG_B0, "C": CFG_C, "D": CFG_D}
ENABLED = {"A", "B", "D"}          # C switched off 2026-11-05 at the user's request (code kept; add "C" here to run it again)


def log(*a):
    s = f"{dt.datetime.now(dt.UTC):%H:%M:%S} " + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def fee(p, rate=0.07):
    return rate * p * (1 - p)


# ------------------------------------------------------------------------------------ live timeline
class LiveTimeline:
    """In-memory per-second arrays (btcmovement2 FIELDS) from t0_epoch (UTC midnight) onwards."""
    def __init__(self, t0_epoch, days=25):
        self.t0_epoch = t0_epoch; self.cap = days * DAY
        self.a = {k: np.zeros(self.cap, dtype=dt_) for k, dt_ in FIELDS.items()}
        for k in ("e_c", "e_v", "e_bv", "p_nb", "p_lbv", "p_lsv", "p_qv"):
            self.a[k] = np.full(self.cap, np.nan if k == "e_c" else 0.0)
        for k in ("s_open", "s_high", "s_low", "s_close", "p_open", "p_high", "p_low", "p_close", "p_vwap"):
            self.a[k][:] = np.nan
        self.known = 0                      # seconds [0, known) are final
        self.last_spot_sec = None; self.last_perp_sec = None   # last bar actually received from each feed
        self.lock = threading.Lock()
        self.perp_pending = {}              # second -> partial perp bar

    @property
    def n(self):
        return self.cap

    def __getitem__(self, k):
        return self.a[k]

    def set_spot(self, sec, o, h, l, c, v, q, n, tb):
        i = sec - self.t0_epoch
        if i < 0 or i >= self.cap: return
        a = self.a
        a["s_open"][i], a["s_high"][i], a["s_low"][i], a["s_close"][i] = o, h, l, c
        a["s_volume"][i], a["s_qvol"][i], a["s_tb_base"][i], a["s_count"][i] = v, q, tb, n

    def set_perp(self, sec, bar):
        i = sec - self.t0_epoch
        if i < 0 or i >= self.cap: return
        a = self.a
        for k in ("p_open", "p_high", "p_low", "p_close", "p_buy", "p_sell", "p_ntr", "p_maxq", "p_vwap"):
            a[k][i] = bar[k]
        for k in ("p_nb", "p_lbv", "p_lsv"):
            a[k][i] = bar.get(k, 0.0)
        a["p_qv"][i] = bar.get("pv", 0.0)

    def set_eth(self, sec, c, v, bv):
        i = sec - self.t0_epoch
        if 0 <= i < self.cap:
            self.a["e_c"][i], self.a["e_v"][i], self.a["e_bv"][i] = c, v, bv

    def finalize_to(self, sec):
        """Mark seconds < sec as known: forward-fill missing prices, synthesize missing perp from spot."""
        j = sec - self.t0_epoch
        if j <= self.known: return
        a = self.a; lo, hi = self.known, j
        # forward-fill the close first (gaps from feed outages), then derive open/high/low from it
        for t in range(lo, hi):
            if np.isnan(a["s_close"][t]) and t > 0: a["s_close"][t] = a["s_close"][t - 1]
        for k in ("s_open", "s_high", "s_low"):
            for t in range(lo, hi):
                if np.isnan(a[k][t]): a[k][t] = a["s_close"][t]
        for t in range(lo, hi):
            if np.isnan(a["p_close"][t]):
                basis = self.basis
                px = a["s_close"][t] * basis
                a["p_open"][t] = a["p_high"][t] = a["p_low"][t] = a["p_close"][t] = a["p_vwap"][t] = px
                a["p_buy"][t] = a["s_tb_base"][t]; a["p_sell"][t] = max(a["s_volume"][t] - a["s_tb_base"][t], 0); a["p_ntr"][t] = a["s_count"][t]
                a["p_nb"][t] = a["s_count"][t] / 2; a["p_qv"][t] = px * a["s_volume"][t]
            if np.isnan(a["e_c"][t]):
                a["e_c"][t] = a["e_c"][t - 1] if t > 0 else np.nan
        self.known = j

    def view(self, k, lo, hi):
        """Slice with unknown seconds forward-filled with the last known price (features never read them)."""
        x = self.a[k][lo:hi].copy()
        kn = self.known - lo
        if kn < len(x):
            if k in ("s_open", "s_high", "s_low", "s_close", "p_open", "p_high", "p_low", "p_close", "p_vwap", "e_c"):
                x[max(kn, 0):] = self.a[k][self.known - 1] if self.known > 0 else np.nan
            else:
                x[max(kn, 0):] = 0
        return x


class ViewTL:
    """Timeline-like view for score_m2.ShiftedTL: __getitem__ returns full-length arrays lazily sliced."""
    def __init__(self, tl):
        self.tl = tl; self.n = tl.cap; self.t0_epoch = tl.t0_epoch
    def __getitem__(self, k):
        return _LazyCol(self.tl, k)


class _LazyCol:
    def __init__(self, tl, k): self.tl, self.k = tl, k
    def __getitem__(self, sl):
        if isinstance(sl, slice):
            return self.tl.view(self.k, sl.start or 0, sl.stop if sl.stop is not None else self.tl.cap)
        return self.tl.a[self.k][sl]


# ------------------------------------------------------------------------------------ backfill
def load_archive(tl):
    """Archive part of the backfill (btcmovement2 timeline + btc-predict extras); returns the archive end (exclusive)."""
    atl = load_timeline()
    a_end = atl.t0_epoch + atl.n                           # archive end (exclusive)
    start = tl.t0_epoch
    lo = start - atl.t0_epoch; hi = min(atl.n, lo + tl.cap)
    for k in FIELDS:
        tl.a[k][: hi - lo] = np.asarray(atl[k][lo:hi])
    tl.known = hi - lo
    # btc-predict archive extras (eth, perp nb/lbv/lsv/qv) for the same days
    import pandas as _pd
    for i_d in range((hi - lo) // DAY):
        day = dt.datetime.fromtimestamp(start + i_d * DAY, dt.UTC).date(); j0 = i_d * DAY
        try:
            e = _pd.read_parquet(f"/home/user/btc-predict/data/sec/eth/{day}.parquet"); u = _pd.read_parquet(f"/home/user/btc-predict/data/sec/um/{day}.parquet")
            tl.a["e_c"][j0:j0 + DAY] = e["c"].to_numpy(float); tl.a["e_v"][j0:j0 + DAY] = e["v"].to_numpy(float); tl.a["e_bv"][j0:j0 + DAY] = e["bv"].to_numpy(float)
            for k, c in (("p_nb", "nb"), ("p_lbv", "lbv"), ("p_lsv", "lsv"), ("p_qv", "qv")): tl.a[k][j0:j0 + DAY] = u[c].to_numpy(float)
        except Exception as ex:
            log("archive extras missing for", day, repr(ex)[:60])
    pc, sc = tl.a["p_close"][:tl.known], tl.a["s_close"][:tl.known]
    ok = np.isfinite(pc) & np.isfinite(sc)
    tl.basis = float(np.nanmedian(pc[ok][-DAY:] / sc[ok][-DAY:])) if ok.any() else 1.0
    log(f"archive loaded through {dt.datetime.fromtimestamp(a_end, dt.UTC)}; perp/spot basis {tl.basis:.6f}")
    return a_end


def backfill(tl, a_end=None):
    a_end = load_archive(tl) if a_end is None else a_end
    # REST backfill of spot + ETH 1 s klines up to now (two threads; retries; progress lines)
    def klines(sym, t_from):
        for k in range(8):
            try:
                r = requests.get("https://data-api.binance.vision/api/v3/klines", params=dict(symbol=sym, interval="1s", startTime=t_from * 1000, limit=1000), timeout=20)
                j = r.json()
                if isinstance(j, list):
                    return j
                log(f"backfill {sym}: HTTP {r.status_code} {str(j)[:80]} - retry {k+1}")
            except Exception as ex:
                log(f"backfill {sym}: {repr(ex)[:80]} - retry {k+1}")
            time.sleep(2 + 3 * k)
        raise RuntimeError(f"backfill {sym} failed at {t_from}")
    def pull(sym, setter):
        t = a_end; now = int(time.time()); n = 0; t_w = time.time()
        while t < now - 2:
            r = klines(sym, t)
            if not r: break
            for k in r: setter(k)
            t = int(r[-1][0]) // 1000 + 1; n += 1
            if n % 10 == 0: log(f"backfill {sym}: {dt.datetime.fromtimestamp(t, dt.UTC):%H:%M} reached ({n} requests, {time.time()-t_w:.0f}s)")
            if len(r) < 1000: break
        return t
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:
        f_spot = ex.submit(pull, "BTCUSDT", lambda k: tl.set_spot(int(k[0]) // 1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[7]), int(k[8]), float(k[9])))
        f_eth = ex.submit(pull, "ETHUSDT", lambda k: tl.set_eth(int(k[0]) // 1000, float(k[4]), float(k[5]), float(k[9])))
        t = f_spot.result(); f_eth.result()
    tl.finalize_to(t)
    log(f"spot backfilled to {dt.datetime.fromtimestamp(t, dt.UTC)} (perp gap synthetic until live OKX bars arrive)")


# ------------------------------------------------------------------------------------ feeds
def spot_ws(tl, stop):
    while not stop.is_set():
        try:
            ws = websocket.create_connection("wss://data-stream.binance.vision/ws/btcusdt@kline_1s", timeout=30)
            while not stop.is_set():
                d = json.loads(ws.recv())["k"]
                if d["x"]:
                    sec = int(d["t"]) // 1000
                    tl.set_spot(sec, float(d["o"]), float(d["h"]), float(d["l"]), float(d["c"]), float(d["v"]), float(d["q"]), int(d["n"]), float(d["V"]))
                    tl.last_spot_sec = sec
        except Exception as e:
            log("spot ws error", repr(e)[:120]); time.sleep(2)


def eth_ws(tl, stop):
    while not stop.is_set():
        try:
            ws = websocket.create_connection("wss://data-stream.binance.vision/ws/ethusdt@kline_1s", timeout=30)
            while not stop.is_set():
                d = json.loads(ws.recv())["k"]
                if d["x"]:
                    tl.set_eth(int(d["t"]) // 1000, float(d["c"]), float(d["v"]), float(d["V"]))
        except Exception as e:
            log("eth ws error", repr(e)[:120]); time.sleep(2)


def okx_ws(tl, stop):
    """OKX perp trades by REST polling (the OKX websocket is reset from this network): newer-than-last-id pages."""
    CT = 0.01  # BTC per contract
    last_id = None
    while not stop.is_set():
        try:
            if last_id is None:
                r = requests.get("https://www.okx.com/api/v5/market/trades", params=dict(instId="BTC-USDT-SWAP", limit=100), timeout=10).json()
                rows = r.get("data", [])
            else:
                rows = []
                for _ in range(10):
                    r = requests.get("https://www.okx.com/api/v5/market/history-trades", params=dict(instId="BTC-USDT-SWAP", type="1", before=last_id, limit=100), timeout=10).json()
                    d = r.get("data", [])
                    if not d: break
                    rows += d; last_id = max(int(x["tradeId"]) for x in d)
                    if len(d) < 100: break
            if rows:
                last_id = max(int(x["tradeId"]) for x in rows)
                with tl.lock:
                    for tr in sorted(rows, key=lambda x: int(x["tradeId"])):
                        sec = int(tr["ts"]) // 1000; px = float(tr["px"]); q = float(tr["sz"]) * CT; buy = tr["side"] == "buy"
                        b = tl.perp_pending.get(sec)
                        if b is None:
                            b = tl.perp_pending[sec] = dict(p_open=px, p_high=px, p_low=px, p_close=px, p_buy=0.0, p_sell=0.0, p_ntr=0, p_maxq=0.0, pv=0.0, vv=0.0, p_nb=0, p_lbv=0.0, p_lsv=0.0)
                        b["p_high"] = max(b["p_high"], px); b["p_low"] = min(b["p_low"], px); b["p_close"] = px
                        b["p_buy" if buy else "p_sell"] += q; b["p_ntr"] += 1; b["p_maxq"] = max(b["p_maxq"], q); b["pv"] += px * q; b["vv"] += q
                        if buy: b["p_nb"] += 1
                        if q >= 1.0: b["p_lbv" if buy else "p_lsv"] += q
                    tl.last_perp_sec = max(tl.perp_pending) if tl.perp_pending else None
            time.sleep(0.4)
        except Exception as e:
            log("okx poll error", repr(e)[:120]); time.sleep(2)


class PolyFeed:
    """Websocket subscription for one window's two tokens; records UP-equivalent prints and books."""
    def __init__(self, T, token_up, token_down):
        self.T, self.up, self.down = T, token_up, token_down
        self.prints = []           # (epoch_ms, up_equiv_price, token_side('UP'|'DOWN'), size)
        self.best = {"UP": (np.nan, np.nan), "DOWN": (np.nan, np.nan)}
        self.stop = threading.Event(); self.lock = threading.Lock()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while not self.stop.is_set():
            try:
                ws = websocket.create_connection("wss://ws-subscriptions-clob.polymarket.com/ws/market", timeout=30)
                ws.send(json.dumps({"assets_ids": [self.up, self.down], "type": "market"}))
                while not self.stop.is_set():
                    m = ws.recv()
                    try: d = json.loads(m)
                    except Exception: continue
                    for ev in (d if isinstance(d, list) else [d]):
                        et = ev.get("event_type")
                        if et == "last_trade_price":
                            side = "UP" if ev["asset_id"] == self.up else "DOWN"; p = float(ev["price"])
                            with self.lock:
                                self.prints.append((int(ev.get("timestamp", time.time() * 1000)), p if side == "UP" else 1 - p, side, float(ev.get("size", 0))))
                        elif et == "book":
                            side = "UP" if ev["asset_id"] == self.up else "DOWN"
                            bids = [float(x["price"]) for x in ev.get("bids", [])]; asks = [float(x["price"]) for x in ev.get("asks", [])]
                            with self.lock:
                                self.best[side] = (max(bids) if bids else np.nan, min(asks) if asks else np.nan)
                        elif et == "price_change":
                            for pc in ev.get("price_changes", []):
                                side = "UP" if pc["asset_id"] == self.up else "DOWN"
                                bb, ba = pc.get("best_bid"), pc.get("best_ask")
                                if bb is not None and ba is not None:
                                    with self.lock:
                                        self.best[side] = (float(bb), float(ba))
            except Exception as e:
                if not self.stop.is_set():
                    log("poly ws error", repr(e)[:100]); time.sleep(1)

    def last_up(self, before_ms=None):
        with self.lock:
            for ts, p, side, sz in reversed(self.prints):
                if before_ms is None or ts <= before_ms:
                    return p, ts
        return np.nan, None

    def prints_since(self, ms):
        with self.lock:
            return [x for x in self.prints if x[0] > ms]


def get_tokens(T):
    for _ in range(3):
        try:
            e = requests.get(f"https://gamma-api.polymarket.com/events?slug=btc-updown-5m-{T}", timeout=15).json()
            if e:
                m = e[0]["markets"][0]; t = json.loads(m["clobTokenIds"])
                return t[0], t[1], m.get("conditionId")
        except Exception as ex:
            log("gamma error", repr(ex)[:80]); time.sleep(1)
    return None


def get_outcome(T):
    try:
        e = requests.get(f"https://gamma-api.polymarket.com/events?slug=btc-updown-5m-{T}", timeout=15).json()
        op = json.loads(e[0]["markets"][0]["outcomePrices"])
        return 1 if float(op[0]) == 1.0 else (0 if float(op[1]) == 1.0 else None)
    except Exception:
        return None


# ------------------------------------------------------------------------------------ model
class Model:
    def __init__(self, tl):
        self.tl = tl; self.pr = Predictor.load_dir("/home/user/btcmovement2/models/wf5_20260430"); self.view = ViewTL(tl)
        self.regime_cache = {}

    def regime(self, T, d0, e):
        """Regime features for the window starting at T+d0 (shifted frame), elapsed e. Slow part (~seconds)."""
        s = d0 % 300; tls = S.ShiftedTL(self.view, s)
        T0_abs = np.array([(T - self.tl.t0_epoch) + d0 - s], dtype=np.int64)
        R = regime_features(tls, T0_abs, np.array([e], dtype=np.int64))
        self.regime_cache[(T, d0, e)] = R
        return R

    def forecast(self, T, d0, e):
        """Mirror of score_m2.score_points for one point, with cached regime features."""
        t = time.time()
        R = self.regime_cache.pop((T, d0, e), None)
        if R is None: R = self.regime(T, d0, e)
        s = d0 % 300; tls = S.ShiftedTL(self.view, s)
        Tloc = T - self.tl.t0_epoch
        c0 = Tloc; a = ((c0 - s - 2 * DAY) // 3600) * 3600 - s; a = max(a, 0)
        while (tls.t0_epoch + a) % 3600: a += 1
        b = min(tls.n, Tloc - s + 900)
        base = S.Base(tls, a, b); fe = S.FeatureEngine(base, self.pr.season)
        T0_s = np.array([Tloc + d0 - s]); T0 = (T0_s - a).astype(np.int64); ee = np.array([e], dtype=np.int64)
        X, aux = fe.compute(T0, ee)
        for k in R.columns: X[k] = R[k].values
        X["xcur"] = aux["cur"] - aux["p0"]; X["v"] = aux["v"].astype(np.float32)
        out = {"p0": float(np.exp(aux["p0"][0])), "price_now": float(np.exp(aux["cur"][0]))}
        for k in ("seas_now", "reg_ll_beta_1h", "reg_trend_1d", "reg_catchup_1h", "lrv_300", "hod_et"):   # regime flags for prospective validation
            out[k] = float(X[k].values[0])
        drift_px = {}
        from btcmodel.models import add_derived
        from btcmodel.predictor import HEAD_H, DIRECT_FEATS
        from scipy.special import logit, expit
        for head, h in HEAD_H.items():
            H = (300 - e) if h is None else h
            if e + H > 300: continue
            d = X.copy(); d[f"{head}__H"] = np.int16(H)
            d[f"{head}__sint"] = self.pr.season.integrated(T0 + ee, np.array([H]), base.how_hour).astype(np.float32)
            hm = self.pr.models[head]["head"]
            for sh in HEAD_H:
                if f"mu_{sh}_px" in hm.extra: d[f"mu_{sh}_px"] = np.float32(drift_px.get(sh, np.nan))
            d = add_derived(d, head)
            if any(c.startswith("mu_h") for c in hm.extra):
                for c in hm.extra: d[c] = (d[c] / d["naive_scale"]).astype(np.float32)
            hm.prepare(d)
            p_m = float(np.clip(hm.p_up(d)[0], 1e-5, 1 - 1e-5))
            dm = self.pr.models[head]["direct"]; dfe = sorted(set(DIRECT_FEATS + list(getattr(hm, "regime", []))))
            p_d = float(np.clip(dm.predict(d[dfe], num_iteration=dm.best_iteration)[0], 1e-5, 1 - 1e-5))
            drift_px[head] = float(d["mu"].values[0] * d["sig"].values[0])
            out[f"{head}_p"] = float(expit(0.5 * logit(p_m) + 0.5 * logit(p_d)))
        out["compute_s"] = round(time.time() - t, 2)
        return out


# ------------------------------------------------------------------------------------ paper book
class Position:
    def __init__(self, strat, T, side, entry, fill_ms, cfg, p_model, market):
        self.strat, self.T, self.side, self.entry, self.fill_ms, self.cfg = strat, T, side, entry, fill_ms, cfg
        self.p_model, self.market = p_model, market; self.open = True; self.rec = None; self.last_poll = None

    def check(self, feed, now):
        """Exit logic on UP-equivalent prints of our side."""
        cfg = self.cfg
        if feed is not None:
            for ts, pu, tok, sz in feed.prints_since(self.fill_ms):
                ps = pu if self.side == "UP" else 1 - pu
                if cfg.get("tp") is not None and ps >= cfg["tp"]:
                    return self.close(cfg["tp"], "TP", ts, maker=True)
                if cfg.get("sl") is not None and ps <= cfg["sl"]:
                    return self.close(ps - cfg["slip"], "SL", ts)
            if cfg.get("tstop") is not None and now >= self.T + cfg["tstop"]:
                bb = feed.best[self.side][0]
                px = bb if np.isfinite(bb) else (feed.last_up()[0] if self.side == "UP" else 1 - feed.last_up()[0]) - cfg["slip"]
                return self.close(px, "TIME", int(now * 1000))
        if cfg.get("settle") and now >= self.T + 300 + 20 and (self.last_poll is None or now - self.last_poll > 30):   # resolution: 1.00 / 0.00, no fee
            self.last_poll = now; out = get_outcome(self.T)
            if out is not None:
                return self.close(1.0 if (out == 1) == (self.side == "UP") else 0.0, "SETTLE", int((self.T + 300) * 1000), maker=True)
        return None

    def close(self, px, kind, ts, maker=False):
        px = float(np.clip(px, 0.0, 1.0)) if kind == "SETTLE" else float(np.clip(px, 0.001, 0.999)); f = 0.0 if maker else fee(px, self.cfg["fee"])
        self.open = False
        self.rec = dict(strategy=self.strat, T=self.T, side=self.side, p_model=self.p_model, market=self.market, entry=self.entry,
                        fill_time=self.fill_ms / 1000 - self.T, exit=px, exit_kind=kind, exit_time=ts / 1000 - self.T, fee=f, pnl=px - self.entry - f,
                        tp=self.cfg["tp"], sl=self.cfg["sl"], tstop=self.cfg["tstop"])
        return self.rec


class Trader:
    def __init__(self):
        now = int(time.time()); t0 = (now // DAY - 10) * DAY
        self.tl = LiveTimeline(t0); self.stop = threading.Event()
        backfill(self.tl)
        threading.Thread(target=spot_ws, args=(self.tl, self.stop), daemon=True).start()
        threading.Thread(target=okx_ws, args=(self.tl, self.stop), daemon=True).start()
        threading.Thread(target=eth_ws, args=(self.tl, self.stop), daemon=True).start()
        from bp_live import BPLive
        self.bp = BPLive(); self.bp_z = {}
        self.model = Model(self.tl); self.feeds = {}; self.pending = []; self.positions = []; self.done = set()
        self.ftrade = open(os.path.join(PAPER, "paper_trades.jsonl"), "a"); self.ffc = open(os.path.join(PAPER, "forecasts.jsonl"), "a")
        self.forder = open(os.path.join(PAPER, "orders.jsonl"), "a")

    def lat(self, scheduled, feed=None):
        """Feed freshness and timing at a decision: seconds since the last received spot bar / perp trade / Polymarket print, and how late the decision is."""
        now = time.time(); tl = self.tl
        d = dict(decision_delay=round(now - scheduled, 2),
                 spot_lag=round(now - (tl.last_spot_sec + 1), 2) if tl.last_spot_sec else None,
                 perp_lag=round(now - (tl.last_perp_sec + 1), 2) if tl.last_perp_sec else None)
        if feed is not None:
            _, ts = feed.last_up(); d["pm_age"] = round(now - ts / 1000, 2) if ts else None
        if d["spot_lag"] is not None and d["spot_lag"] > 3: log(f"STALE spot feed at decision: last bar {d['spot_lag']:.1f}s old (forward-filled)")
        return d

    def feed_for(self, T):
        if T not in self.feeds:
            tk = get_tokens(T)
            if tk is None: return None
            self.feeds[T] = PolyFeed(T, tk[0], tk[1]); log(f"subscribed window {dt.datetime.fromtimestamp(T, dt.UTC):%H:%M}")
        return self.feeds[T]

    def finalize(self):
        """Seconds are final ~1.5 s after they end (spot kline close); commit the timeline up to now-2."""
        now = time.time(); sec = int(now) - 1
        with self.tl.lock:
            for s_ in [s for s in self.tl.perp_pending if s < sec - 1]:
                bb = self.tl.perp_pending.pop(s_); bb["p_vwap"] = bb["pv"] / bb["vv"] if bb["vv"] > 0 else bb["p_close"]; self.tl.set_perp(s_, bb)
        self.tl.finalize_to(sec)

    def run(self):
        log("paper trader started"); last_T = None; t_start = time.time()
        self.heartbeat = time.time()
        def watchdog():                                    # main loop silent > 180 s: dump all stacks, exit (restart loop)
            import faulthandler
            while True:
                time.sleep(10)
                if time.time() - self.heartbeat > 180:
                    log("WATCHDOG: main loop stalled for 180 s - dumping stacks to stdout.log and restarting")
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True); sys.stderr.flush(); LOG.flush(); os._exit(3)
        threading.Thread(target=watchdog, daemon=True).start()
        while True:
            try:
                now = time.time(); T = int(now) // 300 * 300 + 300          # next nominal open
                self.heartbeat = now
                if now - t_start < 60:
                    self.finalize(); time.sleep(0.2); continue
                self.finalize()
                for Tx in (T, T + 300):
                    if now >= Tx - 150: self.feed_for(Tx)
                # ---- regime pre-computation
                if now >= T - 70 and (T, -45, 0) not in self.model.regime_cache and ("A_reg", T) not in self.done:
                    self.model.regime(T, -45, 0); self.done.add(("A_reg", T))
                if now >= T - 20 and (T, -LAG, LAG) not in self.model.regime_cache and ("B_reg", T) not in self.done:
                    self.model.regime(T, -LAG, LAG); self.done.add(("B_reg", T))
                if "C" in ENABLED and now >= T - 25 and (T, -LAG, LAG + CFG_C["decide"]) not in self.model.regime_cache and ("C_reg", T) not in self.done:
                    self.model.regime(T, -LAG, LAG + CFG_C["decide"]); self.done.add(("C_reg", T))
                if now >= T - 300 + CFG_D["decide"] - 30 and (T - 300, -LAG, LAG + CFG_D["decide"]) not in self.model.regime_cache and ("D_reg", T - 300) not in self.done:
                    self.model.regime(T - 300, -LAG, LAG + CFG_D["decide"]); self.done.add(("D_reg", T - 300))
                # ---- strategy D decision at T+90 of the running window: model >= q for a side that still trades <= maxpx -> bid at the last print, hold to settlement
                Td = T - 300
                if Td + CFG_D["decide"] + 1.6 <= now < Td + CFG_D["decide"] + 10 and ("D", Td) not in self.done and self.tl.known >= (Td + CFG_D["decide"]) - self.tl.t0_epoch:
                    self.done.add(("D", Td)); fc = self.model.forecast(Td, -LAG, LAG + CFG_D["decide"]); p = fc.get("close_p", np.nan)
                    feed = self.feed_for(Td); mkt, _ = feed.last_up() if feed else (np.nan, None)
                    plausible = bool(np.isfinite(fc.get("p0", np.nan)) and np.isfinite(p) and 0.01 < p < 0.99)
                    side = "UP" if p >= 0.5 else "DOWN"; ps_model = p if side == "UP" else 1 - p; ps_mkt = mkt if side == "UP" else 1 - mkt
                    action = "skip" if not plausible else (f"BID {side}" if np.isfinite(mkt) and ps_model >= CFG_D["q"] and ps_mkt <= CFG_D["maxpx"] else "none")
                    self.ffc.write(json.dumps(dict(strategy="D", T=Td, ts=round(time.time(), 2), market=mkt, action=action, **self.lat(Td + CFG_D["decide"] + 1.6, feed), **fc)) + "\n"); self.ffc.flush()
                    log(f"[D] T={dt.datetime.fromtimestamp(Td, dt.UTC):%H:%M} p_close={p:.3f} market={mkt:.3f} ({fc['compute_s']}s)")
                    if plausible and action.startswith("BID"):
                        self.pending.append(dict(strategy="D", T=Td, side=side, bid=ps_mkt, placed_ms=int(time.time() * 1000), until=time.time() + CFG_D["cancel"], p=p, mkt=mkt))
                        self.forder.write(json.dumps(dict(strategy="D", T=Td, side=side, bid=ps_mkt, placed=round(time.time(), 2), placed_delay=round(time.time() - Td, 2), event="placed")) + "\n"); self.forder.flush()
                        log(f"[D] REST BID {side} @{ps_mkt:.3f} for {CFG_D['cancel']}s (model {ps_model:.3f})")
                # ---- strategy C decision at T-5 (bars through T-6 final at ~T-4.5): disagreement with the last print, bid until the open
                if "C" in ENABLED and T + CFG_C["decide"] + 1.4 <= now < T and ("C", T) not in self.done and self.tl.known >= (T + CFG_C["decide"]) - self.tl.t0_epoch:
                    self.done.add(("C", T)); fc = self.model.forecast(T, -LAG, LAG + CFG_C["decide"]); p = fc.get("close_p", np.nan)
                    feed = self.feed_for(T); mkt, _ = feed.last_up() if feed else (np.nan, None)
                    plausible = bool(np.isfinite(fc.get("p0", np.nan)) and np.isfinite(p) and 0.05 < p < 0.95)
                    action = "skip" if not plausible else ("BID " + ("UP" if p > mkt else "DOWN") if np.isfinite(mkt) and abs(p - mkt) >= CFG_C["delta"] else "none")
                    self.ffc.write(json.dumps(dict(strategy="C", T=T, ts=round(time.time(), 2), market=mkt, action=action, **self.lat(T + CFG_C["decide"] + 1.4, feed), **fc)) + "\n"); self.ffc.flush()
                    log(f"[C] T={dt.datetime.fromtimestamp(T, dt.UTC):%H:%M} p_close={p:.3f} market={mkt:.3f} ({fc['compute_s']}s)")
                    if plausible and action.startswith("BID"):
                        side = action[4:]; bid = mkt if side == "UP" else 1 - mkt
                        self.pending.append(dict(strategy="C", T=T, side=side, bid=bid, placed_ms=int(time.time() * 1000), until=T, p=p, mkt=mkt))
                        self.forder.write(json.dumps(dict(strategy="C", T=T, side=side, bid=bid, placed=round(time.time(), 2), placed_delay=round(time.time() - T, 2), event="placed")) + "\n"); self.forder.flush()
                        log(f"[C] REST BID {side} @{bid:.3f} until the open")
                # ---- btc-predict forecast at the minute boundary T-60 (seconds < T-60)
                if now >= T - 58 and ("bp", T) not in self.done and self.tl.known >= (T - 60) - self.tl.t0_epoch:
                    self.done.add(("bp", T))
                    try:
                        lo = (T - 60) - 60 * 2880 - 900; i0 = lo - self.tl.t0_epoch; i1 = (T - 60) - self.tl.t0_epoch
                        v = lambda k: self.tl.view(k, i0, i1)
                        spot = dict(o=v("s_open"), h=v("s_high"), l=v("s_low"), c=v("s_close"), v=v("s_volume"), qv=v("s_qvol"), n=v("s_count"), bv=v("s_tb_base"))
                        um = dict(o=v("p_open"), h=v("p_high"), l=v("p_low"), c=v("p_close"), v=v("p_buy") + v("p_sell"), qv=v("p_qv"), n=v("p_ntr"), bv=v("p_buy"), nb=v("p_nb"), lbv=v("p_lbv"), lsv=v("p_lsv"), mxq=v("p_maxq"))
                        eth = dict(c=v("e_c"), v=v("e_v"), bv=v("e_bv"))
                        tb = time.time(); z, ts_last = self.bp.forecast(spot, um, eth, lo); self.bp_z[T] = z
                        log(f"[bp] T={dt.datetime.fromtimestamp(T, dt.UTC):%H:%M} z_s5={z['s5']:+.3f} z_s1={z['s1']:+.3f} (bar {ts_last:%H:%M}, {time.time()-tb:.1f}s, spot lag {self.lat(T - 58)['spot_lag']}s)")
                    except Exception as e:
                        log("bp error", repr(e)[:200])
                # ---- strategy A decision at T-45 (bars through T-46 final at ~T-44)
                if now >= T - 43.5 and ("A", T) not in self.done and self.tl.known >= (T - 45) - self.tl.t0_epoch:
                    self.done.add(("A", T)); fc = self.model.forecast(T, -45, 0); p = fc.get("close_p", np.nan)
                    feed = self.feed_for(T); mkt, _ = feed.last_up() if feed else (np.nan, None)
                    z5 = self.bp_z.get(T, {}).get("s5", np.nan)
                    agree = bool(np.isfinite(z5) and np.sign(z5) == np.sign(p - 0.5))
                    plausible = bool(np.isfinite(fc.get("p0", np.nan)) and np.isfinite(p) and 0.05 < p < 0.95)
                    action = "skip" if not plausible else ("BUY " + ("UP" if p > 0.5 else "DOWN") if abs(p - 0.5) >= CFG_A["thr"] and feed is not None and agree else "none")
                    self.ffc.write(json.dumps(dict(strategy="A", T=T, ts=round(time.time(), 2), market=mkt, z_s5=z5, agree=agree, action=action, **self.lat(T - 43.5, feed), **fc)) + "\n"); self.ffc.flush()
                    if not plausible:
                        log(f"[A] T={dt.datetime.fromtimestamp(T, dt.UTC):%H:%M} SKIP: implausible forecast (p0={fc.get('p0')}, p={p}) - data gap?"); continue
                    log(f"[A] T={dt.datetime.fromtimestamp(T, dt.UTC):%H:%M} p_close={p:.3f} bp_z5={z5:+.3f} agree={agree} market={mkt:.3f} ({fc['compute_s']}s)")
                    if action.startswith("BUY"):
                        side = action[4:]
                        self.positions.append(Position("A", T, side, CFG_A["entry"], int((T - 45) * 1000), CFG_A, p, mkt)); log(f"[A] BUY {side} @0.50 (assumed pre-open fill)")
                        self.positions.append(Position("A0", T, side, CFG_A0["entry"], int((T - 45) * 1000), CFG_A0, p, mkt))   # shadow with the previous exits
                # ---- strategy B decision at T (bars through T-1 final at ~T+1.5)
                if T - 300 + 1.6 <= now < T - 300 + 10 and ("B", T - 300) not in self.done and self.tl.known >= (T - 300) - self.tl.t0_epoch:   # skip stale windows after a restart
                    Tb = T - 300; self.done.add(("B", Tb))
                    fc = self.model.forecast(Tb, -LAG, LAG); p = fc.get("close_p", np.nan)
                    feed = self.feed_for(Tb); mkt, mts = feed.last_up() if feed else (np.nan, None)
                    plausible = bool(np.isfinite(fc.get("p0", np.nan)) and np.isfinite(p) and 0.05 < p < 0.95)
                    action = "skip" if not plausible else ("BID " + ("UP" if p > mkt else "DOWN") if np.isfinite(mkt) and abs(p - mkt) >= CFG_B["delta"] else "none")
                    mkt_preT = feed.last_up(before_ms=Tb * 1000)[0] if feed else np.nan
                    self.ffc.write(json.dumps(dict(strategy="B", T=Tb, ts=round(time.time(), 2), market=mkt, market_preT=mkt_preT, action=action, **self.lat(Tb + 1.6, feed), **fc)) + "\n"); self.ffc.flush()
                    log(f"[B] T={dt.datetime.fromtimestamp(Tb, dt.UTC):%H:%M} p_close={p:.3f} market={mkt:.3f} p0={fc['p0']:.1f} now={fc['price_now']:.1f} ({fc['compute_s']}s)")
                    if not plausible:
                        log(f"[B] T={dt.datetime.fromtimestamp(Tb, dt.UTC):%H:%M} SKIP: implausible forecast (p0={fc.get('p0')}, p={p}) - data gap?"); continue
                    if action.startswith("BID"):
                        side = action[4:]; bid = mkt if side == "UP" else 1 - mkt
                        self.pending.append(dict(strategy="B", T=Tb, side=side, bid=bid, placed_ms=int(time.time() * 1000), until=time.time() + CFG_B["cancel"], p=p, mkt=mkt))
                        self.forder.write(json.dumps(dict(strategy="B", T=Tb, side=side, bid=bid, placed=round(time.time(), 2), placed_delay=round(time.time() - Tb, 2), event="placed")) + "\n"); self.forder.flush()
                        log(f"[B] REST BID {side} @{bid:.3f} for {CFG_B['cancel']}s")
                # ---- pending maker orders
                for o in list(self.pending):
                    feed = self.feeds.get(o["T"]); filled = None
                    if feed:
                        for ts, pu, tok, sz in feed.prints_since(o["placed_ms"]):
                            ps = pu if o["side"] == "UP" else 1 - pu
                            if ps <= o["bid"] + 1e-9: filled = ts; break
                    st = o.get("strategy", "B")
                    if filled:
                        self.pending.remove(o); self.positions.append(Position(st, o["T"], o["side"], o["bid"], filled, CFGS[st], o["p"], o["mkt"]))
                        if st == "B": self.positions.append(Position("B0", o["T"], o["side"], o["bid"], filled, CFG_B0, o["p"], o["mkt"]))   # shadow with the previous exits
                        self.forder.write(json.dumps(dict(strategy=st, T=o["T"], side=o["side"], bid=o["bid"], event="filled", fill_time=round(filled / 1000 - o["T"], 3))) + "\n"); self.forder.flush()
                        log(f"[{st}] FILLED {o['side']} @{o['bid']:.3f} at T{filled/1000 - o['T']:+.1f}s")
                    elif time.time() > o["until"]:
                        self.pending.remove(o); log(f"[{st}] cancelled unfilled bid {o['side']} @{o['bid']:.3f}")
                        self.forder.write(json.dumps(dict(strategy=st, T=o["T"], side=o["side"], bid=o["bid"], event="cancelled")) + "\n"); self.forder.flush()
                # ---- open positions
                for pos in list(self.positions):
                    feed = self.feeds.get(pos.T)
                    if feed is None and not pos.cfg.get("settle"): continue
                    rec = pos.check(feed, time.time())
                    if rec:
                        self.positions.remove(pos); self.ftrade.write(json.dumps(rec) + "\n"); self.ftrade.flush()
                        log(f"[{rec['strategy']}] EXIT {rec['exit_kind']} {rec['side']} @{rec['exit']:.3f} at T+{rec['exit_time']:.0f}s  pnl/share {rec['pnl']:+.4f}")
                # ---- cleanup old feeds
                for Tx in [t for t in self.feeds if t + 330 < now]:
                    self.feeds[Tx].stop.set(); del self.feeds[Tx]
                time.sleep(0.2)
            except Exception as e:
                log("loop error", traceback.format_exc()[-600:]); time.sleep(1)


def summary():
    f = os.path.join(PAPER, "paper_trades.jsonl")
    if not os.path.exists(f): print("no trades yet"); return
    t = pd.read_json(f, lines=True)
    if len(t) == 0: print("no trades yet"); return
    outs = {}
    for T in t["T"].unique():
        outs[T] = get_outcome(int(T))
    t["resolved_up"] = t["T"].map(outs)
    t["settle_pnl"] = np.where(t.resolved_up.isna(), np.nan, np.where((t.resolved_up == 1) == (t.side == "UP"), 1.0, 0.0) - t.entry)
    pd.set_option("display.width", 220)
    SHADOW = {"A0": "A", "B0": "B"}                      # old exits replayed on the same entries, for comparison only
    live = t[t.strategy.isin(ENABLED)]                   # only the strategies that are switched on
    for s, g in live.groupby("strategy"):
        w_, l_ = g.pnl[g.pnl > 0].mean(), g.pnl[g.pnl <= 0].mean(); be = (-l_ / (w_ - l_)) if np.isfinite(w_) and np.isfinite(l_) and w_ > l_ else np.nan
        print(f"\n== strategy {s}: {len(g)} trades, mean pnl/share {g.pnl.mean():+.4f}, win {np.mean(g.pnl > 0):.2%} (break-even {be:.1%} from avg win {w_:+.3f} / avg loss {l_:+.3f}), total {g.pnl.sum():+.3f}, "
              f"settle-equivalent {g.settle_pnl.mean():+.4f} | exits {g.exit_kind.value_counts().to_dict()}")
    print(f"\n== all live strategies combined: {len(live)} trades, total {live.pnl.sum():+.3f} per share; at $100 per trade: ${(live.pnl / live.entry * 100).sum():+.0f}")
    for sh, base in SHADOW.items():
        g0 = t[t.strategy == sh]
        if not len(g0): continue
        g1 = t[(t.strategy == base) & t["T"].isin(g0["T"])]
        print(f"   shadow {sh} (old exits on the same {len(g0)} entries): {g0.pnl.sum():+.3f} per share, win {np.mean(g0.pnl > 0):.0%}  vs  {base} on those windows: {g1.pnl.sum():+.3f}, win {np.mean(g1.pnl > 0):.0%}")
    print(live.tail(15)[["strategy", "T", "side", "p_model", "market", "entry", "fill_time", "exit_kind", "exit_time", "exit", "pnl", "resolved_up"]].round(3).to_string(index=False))


if __name__ == "__main__":
    if "--summary" in sys.argv: summary()
    else: Trader().run()
