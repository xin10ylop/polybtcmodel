"""What the live rule would have done over a period with no real trading: signals from the production models on Binance's official data, fills simulated on
the real Polymarket tape (bid 0.50 from T-118, refused when our side was last seen on offer at 0.50 or lower, filled once somebody sells below 0.50 or $50
has been sold at 0.50, cancelled at T-44 if the T-45 model disagrees and at T-15), result from Polymarket's own resolution.  Trading hours only.
  python research/deep/since_stop.py START_UTC END_UTC   e.g. 2026-11-15T04:20 2026-11-17T00:00"""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, requests, glob
from concurrent.futures import ThreadPoolExecutor
t0 = pd.Timestamp(sys.argv[1]).timestamp(); t1 = pd.Timestamp(sys.argv[2]).timestamp(); M = "/home/user/polybtcmodel/models/"; D = "/home/user/data/deep/"
def load(dec):
    F = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{D}live_check_feats_d{dec}_*.parquet"))]).drop_duplicates("T", keep="last").set_index("T"); meta = json.load(open(f"{M}ourcase_BTC_d{dec}_live.json")); b = lgb.Booster(model_file=f"{M}ourcase_BTC_d{dec}_live.txt"); return pd.Series(b.predict(F[meta["features"]].values), index=F.index), meta, F
s120, m120, F = load(-120); s45, _, _ = load(-45); thr = m120["tier_abs_score"]["top30"]; T = F.index[(F.index >= t0) & (F.index < t1)]; ts = pd.to_datetime(T, unit="s"); good = (ts.dayofweek >= 5) | (ts.hour >= 16); sig = [int(w) for w, g in zip(T, good) if g and abs(s120[w]) >= thr]
def fetch(w):
    for k in range(4):
        try:
            m = requests.get("https://gamma-api.polymarket.com/events", params=dict(slug=f"btc-updown-5m-{w}"), timeout=20).json()[0]["markets"][0]; p = json.loads(m.get("outcomePrices") or "[]"); up = (float(p[0]) > 0.5) if p and (float(p[0]) > 0.99 or float(p[0]) < 0.01) else None; pr = []; cur = None
            while True:
                d = requests.get("https://data-api.polymarket.com/v2/trades", params=dict(condition=m["conditionId"], limit=1000, **({"cursor": cur} if cur else {})), timeout=30).json(); pr += [(int(r["timestamp"]) - w, r["side"], int(r["outcome_index"]), float(r["price"]), float(r["size"])) for r in d.get("data") or [] if -200 <= int(r["timestamp"]) - w <= 0]; pg = d.get("pagination") or {}; cur = pg.get("next_cursor")
                if not pg.get("has_more") or not cur: break
            return w, up, sorted(pr)
        except Exception as e: err = e
    return w, None, []
with ThreadPoolExecutor(6) as ex: R = {w: (up, pr) for w, up, pr in ex.map(fetch, sig)}
rows = []
for w in sig:
    up, pr = R[w]; s = s120[w]; sd = 0 if s > 0 else 1
    if up is None: continue
    rel = np.array([p[0] for p in pr]); o0 = np.array([p[2] == 0 for p in pr], bool); buy = np.array([p[1] == "BUY" for p in pr], bool); px = np.array([p[3] for p in pr]); usd = px * np.array([p[4] for p in pr]); pu = np.where(o0, px, 1 - px); lu = (o0 & buy) | (~o0 & ~buy); po = pu if sd == 0 else 1 - pu; tb = lu if sd == 0 else ~lu
    last = np.flatnonzero((rel < -118) & (rel >= -178)); refused = len(last) > 0 and ((tb[last[-1]] and po[last[-1]] <= 0.50 + 1e-6) or (not tb[last[-1]] and po[last[-1]] < 0.495)); agree = (s45.get(w, np.nan) > 0) == (s > 0) if np.isfinite(s45.get(w, np.nan)) else True; stop = -15 if agree else -44
    k = np.flatnonzero(~tb & (rel >= -118) & (rel <= stop) & (po <= 0.50 + 1e-6)) if len(rel) else np.zeros(0, int); filled = False
    if len(k) and not refused: c = np.cumsum(usd[k]); filled = bool((po[k] < 0.495).any() or c[-1] >= 50.0)
    won = up if sd == 0 else not up; rows.append(dict(T=w, t=pd.Timestamp(w, unit="s"), side="UP" if sd == 0 else "DOWN", score=s, strong=abs(s) >= 2.2 * thr, agree=agree, refused=refused, filled=filled, won=won))
X = pd.DataFrame(rows); X["stake"] = np.where(X.strong, 20.0, 10.0); X["pnl"] = np.where(X.filled, np.where(X.won, X.stake, -X.stake), 0.0); f = X[X.filled]; X.to_csv(D + "since_stop.csv", index=False)
print(f"period {sys.argv[1]} .. {sys.argv[2]} UTC, trading hours: {len(X)} signals, all of them right {100*X.won.mean():.1f} %; refused as crossing {int(X.refused.sum())}; would have FILLED {len(f)}: won {int(f.won.sum())}, lost {int((~f.won).sum())} -> {100*f.won.mean():.1f} %; not filled: right {100*X.won[~X.filled].mean():.1f} %")
print(f"   profit at the $10 stake with the strongest doubled: ${f.pnl.sum():+.0f} (doubled trades: {int(f.strong.sum())}, won {int((f.strong & f.won).sum())}); flat $10: ${np.where(f.won, 10, -10).sum():+.0f}")
X["day"] = X.t.dt.strftime("%m-%d"); print(X.groupby("day").apply(lambda g: pd.Series({"signals": len(g), "signals right %": round(100 * g.won.mean(), 1), "fills": int(g.filled.sum()), "fills won": int((g.filled & g.won).sum()), "fill win %": round(100 * g.won[g.filled].mean(), 1) if g.filled.any() else np.nan, "pnl $": g.pnl.sum()})).to_string())
