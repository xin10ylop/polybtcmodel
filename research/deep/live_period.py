"""A rule replayed over the live period: signals from the production models on Binance's official data, fills on the real Polymarket tape (bid 0.50 from T-118,
refused when our side was last seen on offer at <= 0.50, filled once somebody sells below 0.50 or $50 has been sold at 0.50, the T-43 check cancels an unfilled
bid, then T-15), results from Polymarket's resolution.  Trading hours (weekends + weekday 16-24 UTC).
  python research/deep/live_period.py START END  ->  prints both rules day by day (current; 1.6x + strong T-45 check)"""
import sys, os, glob, json, pickle, numpy as np, pandas as pd, lightgbm as lgb, requests
from concurrent.futures import ThreadPoolExecutor
DEEP = "/home/user/data/deep/"; MOD = "/home/user/polybtcmodel/models/"; t0 = pd.Timestamp(sys.argv[1]).timestamp(); t1 = pd.Timestamp(sys.argv[2]).timestamp()
def prod(dec):
    F = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DEEP}live_check_feats_d{dec}_*.parquet"))]).drop_duplicates("T", keep="last").set_index("T").sort_index(); meta = json.load(open(f"{MOD}ourcase_BTC_d{dec}_live.json"))
    return pd.Series(lgb.Booster(model_file=f"{MOD}ourcase_BTC_d{dec}_live.txt").predict(F[meta["features"]].values), index=F.index), meta["tier_abs_score"]["top30"]
S120, thr = prod(-120); S45, thr45 = prod(-45)
T = [int(w) for w in S120.index if t0 <= w < t1 and abs(S120[w]) >= thr and (pd.Timestamp(w, unit="s").dayofweek >= 5 or pd.Timestamp(w, unit="s").hour >= 16)]
CACHE = f"{DEEP}strong_lab_tapes.pkl"; TP = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
def fetch(w):
    for k in range(4):
        try:
            m = requests.get("https://gamma-api.polymarket.com/events", params=dict(slug=f"btc-updown-5m-{w}"), timeout=20).json()[0]["markets"][0]; p = json.loads(m.get("outcomePrices") or "[]"); up = (float(p[0]) > 0.5) if p and (float(p[0]) > 0.99 or float(p[0]) < 0.01) else None; pr = []; cur = None
            while True:
                d = requests.get("https://data-api.polymarket.com/v2/trades", params=dict(condition=m["conditionId"], limit=1000, **({"cursor": cur} if cur else {})), timeout=30).json(); pr += [(int(r["timestamp"]) - w, r["side"], int(r["outcome_index"]), float(r["price"]), float(r["size"])) for r in d.get("data") or [] if -200 <= int(r["timestamp"]) - w <= 0]; pg = d.get("pagination") or {}; cur = pg.get("next_cursor")
                if not pg.get("has_more") or not cur: break
            return w, (up, sorted(pr))
        except Exception: pass
    return w, (None, [])
need = [w for w in T if w not in TP or TP[w][0] is None]
if need:
    with ThreadPoolExecutor(6) as ex: TP.update(dict(ex.map(fetch, need)))
    pickle.dump(TP, open(CACHE, "wb"))
def sim(w, strong_check):
    up, pr = TP[w]; s = S120[w]; s45 = S45.get(w, np.nan); sd = 0 if s > 0 else 1
    if up is None or not pr: return None
    ag = (not np.isfinite(s45)) or ((s45 > 0) == (s > 0) and (abs(s45) >= thr45 if strong_check else True)); stop = -15 if ag else -44
    rel = np.array([p[0] for p in pr]); o0 = np.array([p[2] == 0 for p in pr], bool); buy = np.array([p[1] == "BUY" for p in pr], bool); px = np.array([p[3] for p in pr]); usd = px * np.array([p[4] for p in pr]); pu = np.where(o0, px, 1 - px); lu = (o0 & buy) | (~o0 & ~buy)
    po = pu if sd == 0 else 1 - pu; tb = lu if sd == 0 else ~lu; last = np.flatnonzero((rel < -118) & (rel >= -178))
    if len(last) and ((tb[last[-1]] and po[last[-1]] <= 0.50 + 1e-6) or (not tb[last[-1]] and po[last[-1]] < 0.495)): return None
    k = np.flatnonzero(~tb & (rel >= -118) & (rel <= stop) & (po <= 0.50 + 1e-6))
    if not len(k): return None
    c = np.cumsum(usd[k]); okq = (po[k] < 0.495) | (c >= 50.0)
    return (up if sd == 0 else not up) if okq.any() else None
rows = []
for w in T:
    d = pd.Timestamp(w, unit="s").strftime("%a %m-%d"); m = abs(S120[w]) / thr
    a = sim(w, False); b = sim(w, True) if m >= 1.6 else None
    rows.append(dict(day=d, T=w, cur=a, new=b, dbl=m >= 2.2))
X = pd.DataFrame(rows); out = []
for d, g in X.groupby("day", sort=False):
    c = g[g.cur.notna()]; n = g[g.new.notna()]
    out.append({"day (UTC)": d, "CURRENT trades": len(c), "won": int(c.cur.sum()), "win %": round(100 * c.cur.mean(), 1) if len(c) else np.nan, "$10 flat": int(np.where(c.cur.astype(bool), 10, -10).sum()), "$10, strongest x2": int((np.where(c.cur.astype(bool), 10, -10) * np.where(c.dbl, 2, 1)).sum()),
                "1.6x RULE trades": len(n), "won ": int(n.new.sum()), "win % ": round(100 * n.new.mean(), 1) if len(n) else np.nan, "$10": int(np.where(n.new.astype(bool), 10, -10).sum())})
O = pd.DataFrame(out); tot = O.sum(numeric_only=True); tot["day (UTC)"] = "TOTAL"; tot["win %"] = round(100 * tot["won"] / tot["CURRENT trades"], 1); tot["win % "] = round(100 * tot["won "] / tot["1.6x RULE trades"], 1); O = pd.concat([O, pd.DataFrame([tot])], ignore_index=True)
pd.set_option("display.width", 250); print(O.to_string(index=False))
