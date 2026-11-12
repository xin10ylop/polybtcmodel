"""15-minute Up/Down markets (Polymarket) with the live rule: the T-120 score of the window that starts on the quarter hour, top 30 %, post-only bid 0.50 on that
side from T-118, cancel when the T-45 model disagrees (if not filled yet) and at T-15, hold to settlement; trading hours = weekends + weekday 16-24 UTC.
Queue-aware fills on the real 15-minute tape: filled once somebody sells our side below 0.50, or once $Q has been sold at 0.50; the stake is limited to the dollars
sold into 0.50 after that.  Tape on disk: Oct 27 - Nov 7 only (12 days), so the trade count and fill sizes are reliable, the realised profit is not: the
'expected' column uses the 14-month hit rate instead.     ASSET=BTC|ETH python research/deep/pm15_check.py"""
import os, glob, numpy as np, pandas as pd
A = os.environ.get("ASSET", "BTC"); DEEP = "/home/user/data/deep"; mk = pd.read_parquet(glob.glob(f"/home/user/data/pm15/{A.lower()}/markets_*.parquet")[0]); mk = mk[mk.uma_status == "resolved"].set_index("window_start"); up = mk.outcome_up_price > 0.5
s120 = pd.read_parquet(f"{DEEP}/ourcase_BTC_d-120_ph3.parquet").set_index("T")["lgb_reg300"]; s45 = pd.read_parquet(f"{DEEP}/ourcase_{A}_ph5.parquet").set_index("T")["lgb_reg300"]      # direction always from the BTC T-120 model; confirmation from the asset's own T-45 model
thr = s120[(s120.index >= pd.Timestamp("2026-06-01").timestamp()) & (s120.index < pd.Timestamp("2026-10-01").timestamp())].abs().quantile(0.70)
T = np.array([t for t in mk.index if t in s120.index and np.isfinite(s120[t])]); ts = pd.to_datetime(T, unit="s"); good = (ts.dayofweek >= 5) | (ts.hour >= 16); days = len(np.unique(ts.date)); P = {}
for f in sorted(glob.glob(f"/home/user/data/pm15/{A.lower()}/trades/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t["rel"] = t.ts - t.window_start; t = t[(t.rel >= -200) & (t.rel <= 0)].sort_values(["window_start", "ts"], kind="stable")
    o0 = t.outcome_index.values == 0; buy = t.side.values == "BUY"; t["pu"] = np.where(o0, t.price.values, 1 - t.price.values); t["lu"] = (o0 & buy) | (~o0 & ~buy); t["usd"] = t.price.values * t["size"].values
    for w, g in t.groupby("window_start", sort=False): P[int(w)] = (g.rel.values, g.pu.values, g.lu.values, g.usd.values)
def sim(Q, end=-15, hours=True, confirm=True):
    out = []
    for w, gd in zip(T, good):
        s = s120[w]
        if abs(s) < thr or (hours and not gd): continue
        sd = 0 if s > 0 else 1; won = bool(up[w]) if sd == 0 else not bool(up[w]); rel, pu, lu, usd = P.get(int(w), (np.zeros(0), np.zeros(0), np.zeros(0, bool), np.zeros(0))); po = pu if sd == 0 else 1 - pu; tb = lu if sd == 0 else ~lu
        last = np.flatnonzero((rel < -118) & (rel >= -178)); refused = len(last) > 0 and ((tb[last[-1]] and po[last[-1]] <= 0.50 + 1e-6) or (not tb[last[-1]] and po[last[-1]] < 0.495))      # our side was last seen on offer at 0.50 or lower: a post-only bid would be refused
        stop = end
        if confirm and w in s45.index and np.isfinite(s45[w]) and (s45[w] > 0) != (s > 0): stop = min(end, -44)
        k = np.flatnonzero(~tb & (rel >= -118) & (rel <= stop) & (po <= 0.50 + 1e-6)); fill = None; va = 0.0
        if len(k) and not refused:
            c = np.cumsum(usd[k]); thru = np.flatnonzero(po[k] < 0.495); a = thru[0] if len(thru) else None; b = np.flatnonzero(c >= Q); b = b[0] if len(b) else None
            j = min([x for x in (a, b) if x is not None], default=None)
            if j is not None: fill = rel[k][j]; kk = np.flatnonzero(~tb & (rel >= fill) & (rel <= end) & (po <= 0.50 + 1e-6)); va = float(usd[kk].sum() - (min(Q, c[j]) if (a is None or j != a) else 0.0))
        out.append((w, won, refused, fill is not None, max(va, 0.0)))
    return pd.DataFrame(out, columns=["T", "won", "refused", "filled", "va"])
HIT = 0.56 if A == "BTC" else 0.53           # long-history hit rate of filled trades to expect (BTC 5-minute rule: 56.1 %; BTC signal on ETH: 52-55 %)
print(f"{A} 15-minute markets, {days} days, {len(T)} markets with a score ({len(T)/days:.0f}/day); threshold for the top 30 % = {thr:.4f}")
for lbl, kw in (("live rule, trading hours", {}), ("all hours", dict(hours=False)), ("no T-45 confirmation, trading hours", dict(confirm=False)), ("bid kept until the open (T-1), trading hours", dict(end=-1))):
    for Q in (10.0, 25.0):
        d = sim(Q, **kw); f = d[d.filled]; print(f"\n-- {lbl}, queue ahead ${Q:.0f}: signals {len(d)/days:.1f}/day, refused {100*d.refused.mean():.0f} %, filled {len(f)/days:.1f}/day ({100*len(f)/max(len(d),1):.0f} % of signals); filled trades won {100*f.won.mean():.1f} % ({len(f)} trades, too few to mean much); signals right {100*d.won.mean():.1f} %")
        print(f"   dollars sold into us after our fill: median {f.va.median():.0f}, mean {f.va.mean():.0f}, 75th pct {f.va.quantile(.75):.0f}")
        rows = []
        for S in (10, 20, 50, 100, 200):
            cap = np.minimum(S, f.va.values); real = (cap / 0.5 * np.where(f.won, 0.5, -0.5)).sum(); rows.append({"stake $": S, "fills completely %": round(100 * (cap >= S).mean()), "avg $ filled": round(cap.mean(), 1), "dollars filled per day": round(cap.sum() / days), f"expected $/day at {100*HIT:.0f} % wins": round(cap.sum() / days * 2 * (HIT - 0.5), 1), "realised $ in these 12 days (noisy)": round(real)})
        print(pd.DataFrame(rows).to_string(index=False))
