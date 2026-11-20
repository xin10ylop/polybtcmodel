"""Developing the 'strong signals only' rule.  Three parts, with the selection rule fixed BEFORE looking at the confirmation period:
 1) Raw accuracy on 13 months of walk-forward scores (Nov 2025 - Nov 19 2026; the last 12 days from the production models on Binance's official data):
    T-120 score multiple m (|score| >= m x the top-30 % threshold) x what the T-45 model says.  Trading hours.
 2) Polymarket fills, Jun 1 - Nov 19 (policy_lab backtest to Nov 7, then the live tape), for the bid variants
      V0 live rule: bid 0.50 from T-118, cancelled at T-44 if the T-45 model disagrees, else at T-15
      V1 as V0 but cancelled at T-44 unless the T-45 model agrees AND is strong (its own top 30 %)
      V2 bid only from T-43, only when the T-45 model agrees and is strong
      V3 bid only from T-43, when the T-45 model agrees (any strength)
    x m in 1.0 .. 2.2, at $200 (fill limited to the dollars sold into us after a $50 queue) and at $10.
 3) Pre-declared choice: among rules with at least 3 trades a day, the best profit per dollar of drawdown at $200 on Jun 1 - Sep 30; then judged on Oct 1 - Nov 19.
  python research/deep/strong_lab.py"""
import sys, os, glob, json, pickle, numpy as np, pandas as pd, lightgbm as lgb, requests, warnings; warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from policy_lab import run, Tix, date
DEEP = "/home/user/data/deep/"; MOD = "/home/user/polybtcmodel/models/"; MS = [1.0, 1.3, 1.5, 1.7, 2.0, 2.2]; pd.set_option("display.width", 300); rng = np.random.default_rng(0)
def prod(dec):
    F = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DEEP}live_check_feats_d{dec}_*.parquet"))]).drop_duplicates("T", keep="last").set_index("T").sort_index(); meta = json.load(open(f"{MOD}ourcase_BTC_d{dec}_live.json"))
    return pd.Series(lgb.Booster(model_file=f"{MOD}ourcase_BTC_d{dec}_live.txt").predict(F[meta["features"]].values), index=F.index), meta["tier_abs_score"]["top30"], F.x_twap
L120, thr120, xL = prod(-120); L45, thr45, _ = prod(-45); L0 = pd.Timestamp("2026-11-08").timestamp(); L1 = pd.Timestamp("2026-11-20").timestamp()
# ---------------------------------------------------------------- 1) raw accuracy
W = pd.read_parquet(f"{DEEP}ourcase_BTC_d-120_ph3.parquet")[["T", "ym", "x_twap", "lgb_reg300"]].rename(columns={"lgb_reg300": "s"}).merge(pd.read_parquet(f"{DEEP}ourcase_BTC_ph5.parquet")[["T", "lgb_reg300"]].rename(columns={"lgb_reg300": "s45"}), on="T")
W = W[np.isfinite(W.s) & np.isfinite(W.s45)].reset_index(drop=True); mos = sorted(W.ym.unique()); W["t1"] = np.nan; W["t45"] = np.nan
for m in mos[1:]: k = W.ym.values == m; W.loc[k, "t1"] = W.s[W.ym.values < m].abs().quantile(0.70); W.loc[k, "t45"] = W.s45[W.ym.values < m].abs().quantile(0.70)
Lw = pd.DataFrame({"T": L120.index, "s": L120.values, "s45": L45.reindex(L120.index).values, "x_twap": xL.reindex(L120.index).values, "t1": thr120, "t45": thr45}); Lw = Lw[(Lw["T"] >= L0) & (Lw["T"] < L1)]
R = pd.concat([W[["T", "s", "s45", "x_twap", "t1", "t45"]], Lw], ignore_index=True).dropna(); R = R[R.x_twap != 0]; t = pd.to_datetime(R["T"], unit="s"); R = R[((t.dt.dayofweek >= 5) | (t.dt.hour >= 16)).values]; t = pd.to_datetime(R["T"], unit="s")
R["m"] = R.s.abs() / R.t1; R["hit"] = (R.s > 0) == (R.x_twap > 0); R["ag"] = (R.s45 > 0) == (R.s > 0); R["st45"] = R.ag & (R.s45.abs() >= R.t45); sel = (t < "2026-06-01").values; nd_s = t[sel].dt.normalize().nunique(); nd_c = t[~sel].dt.normalize().nunique()
rows = []
for m in MS:
    for nm, c in (("any", np.ones(len(R), bool)), ("T-45 agrees", R.ag.values), ("T-45 agrees strongly", R.st45.values)):
        k = (R.m.values >= m) & c; rows.append({"T-120 >= m x thr": m, "T-45": nm, "Nov25-May26 right %": round(100 * R.hit.values[k & sel].mean(), 1), "signals/day": round((k & sel).sum() / nd_s, 1), "Jun-Nov19 right %": round(100 * R.hit.values[k & ~sel].mean(), 1), "signals/day ": round((k & ~sel).sum() / nd_c, 1)})
print("1) RAW ACCURACY of the signals (walk-forward, trading hours), selection period Nov 2025 - May 2026 | Jun - Nov 19 2026"); print(pd.DataFrame(rows).to_string(index=False))
# ---------------------------------------------------------------- 2) fills: backtest part (policy_lab, Jun 1 - Nov 7)
good = np.asarray((date.dayofweek >= 5) | (pd.to_datetime(Tix, unit="s").hour >= 16)); VAR = {"V0": dict(), "V1": dict(confirm=dict(cancel=True, need_strong=True, q=0.70)), "V2": dict(stages=((-43, -15, 0.50),), confirm=dict(cancel=True, need_strong=True, q=0.70)), "V3": dict(stages=((-43, -15, 0.50),), confirm=dict(cancel=True))}
B = []
for v, kw in VAR.items():
    a = dict(dec=-120, tier=0.70, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=50.0); a.update(kw); r = run(**a); g = good[r["i"]]
    B.append(pd.DataFrame({"var": v, "T": Tix[r["i"][g]].astype(np.int64), "won": r["v"][g] > 0, "e": r["e"][g], "va": r["vol_after"][g], "m": r["rel_score"][g]}))
B = pd.concat(B, ignore_index=True)
# ---------------------------------------------------------------- 2) fills: live part (Nov 8 - Nov 19, real tape, production models)
CACHE = f"{DEEP}strong_lab_tapes.pkl"; sigT = [int(w) for w in L120.index if L0 <= w < L1 and abs(L120[w]) >= thr120 and (pd.Timestamp(w, unit="s").dayofweek >= 5 or pd.Timestamp(w, unit="s").hour >= 16)]
TP = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
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
need = [w for w in sigT if w not in TP]
if need:
    with ThreadPoolExecutor(6) as ex: TP.update(dict(ex.map(fetch, need)))
    pickle.dump(TP, open(CACHE, "wb"))
def sim(w, start, cond):
    up, pr = TP[w]; s = L120[w]; s45 = L45.get(w, np.nan); sd = 0 if s > 0 else 1
    if up is None: return None
    ag = np.isfinite(s45) and (s45 > 0) == (s > 0); st = ag and abs(s45) >= thr45
    if start == -43 and not (st if cond == "strong" else ag): return None
    keep = (st if cond == "strong" else ag); stop = -15 if keep else -44
    rel = np.array([p[0] for p in pr]); o0 = np.array([p[2] == 0 for p in pr], bool); buy = np.array([p[1] == "BUY" for p in pr], bool); px = np.array([p[3] for p in pr]); usd = px * np.array([p[4] for p in pr]); pu = np.where(o0, px, 1 - px); lu = (o0 & buy) | (~o0 & ~buy)
    if not len(rel): return None
    po = pu if sd == 0 else 1 - pu; tb = lu if sd == 0 else ~lu; last = np.flatnonzero((rel < start) & (rel >= start - 60))
    if len(last) and ((tb[last[-1]] and po[last[-1]] <= 0.50 + 1e-6) or (not tb[last[-1]] and po[last[-1]] < 0.495)): return None
    k = np.flatnonzero(~tb & (rel >= start) & (rel <= stop) & (po <= 0.50 + 1e-6))
    if not len(k): return None
    c = np.cumsum(usd[k]); okq = (po[k] < 0.495) | (c >= 50.0)
    if not okq.any(): return None
    j = int(np.argmax(okq)); va = float(max(c[-1] - 50.0, 0.0)) if c[j] >= 50.0 else float(usd[k[j:]].sum()); return dict(T=w, won=up if sd == 0 else not up, e=0.5, va=va, m=abs(s) / thr120)
Lf = []
for v, (start, cond) in {"V0": (-118, "agree"), "V1": (-118, "strong"), "V2": (-43, "strong"), "V3": (-43, "agree")}.items():
    for w in sigT:
        o = sim(w, start, cond)
        if o: o["var"] = v; Lf.append(o)
A = pd.concat([B, pd.DataFrame(Lf)], ignore_index=True); A["day"] = pd.to_datetime(A["T"], unit="s").dt.strftime("%Y-%m-%d"); A["wk"] = pd.to_datetime(A["T"], unit="s").dt.to_period("W"); A["early"] = A["T"] < pd.Timestamp("2026-10-01").timestamp()
D_ALL = pd.date_range("2026-06-01", "2026-11-19").strftime("%Y-%m-%d"); D_E = [d for d in D_ALL if d < "2026-10-01"]; D_L = [d for d in D_ALL if d >= "2026-10-01"]
def dd(x): c = np.cumsum(x); return float((np.maximum.accumulate(np.maximum(c, 0)) - c).max()) if len(c) else 0.0
def stats(g, stake, days):
    if not len(g): return dict(n=0)
    s_ = np.minimum(stake, g.va.values) if stake > 50 else np.full(len(g), float(stake)); p = np.where(g.won.values, 0.5, -0.5) * s_ / g.e.values + s_ / g.e.values * 0.0035; o = np.argsort(g["T"].values); gd = pd.Series(p).groupby(g.day.values).sum().reindex(days, fill_value=0.0)
    wk = pd.Series(g.won.values.astype(float)).groupby(g.wk.values).agg(["size", "mean"]); wp = pd.Series(p).groupby(g.wk.values).sum()
    return dict(n=len(g), per_day=len(g) / len(days), win=100 * g.won.mean(), tot=p.sum(), pday=p.sum() / len(days), dd=dd(p[o]), ratio=p.sum() / max(dd(p[o]), 1.0), worst=gd.min(), lose_days=int((gd < 0).sum()), wk_sd=100 * wk["mean"].std(), lose_wks=int((wp < 0).sum()), wks=len(wp))
rows = []
for v in VAR:
    for m in MS:
        g = A[(A["var"] == v) & (A.m >= m)]; e = stats(g[g.early], 200, D_E); l = stats(g[~g.early], 200, D_L); a = stats(g, 200, D_ALL); s10 = stats(g, 10, D_ALL)
        rows.append({"rule": f"{v} m>={m}", "trades/day": round(a["per_day"], 1), "win %": round(a["win"], 1), "$200: total": round(a["tot"]), "per day": round(a["pday"]), "max DD": round(a["dd"]), "profit/DD": round(a["ratio"], 1), "worst day": round(a["worst"]), "losing wks": f"{a['lose_wks']}/{a['wks']}", "weekly win-% sd": round(a["wk_sd"], 1),
                     "SEL Jun-Sep: /day": round(e["per_day"], 1), "SEL win %": round(e["win"], 1), "SEL profit/DD": round(e["ratio"], 1), "CONF Oct-Nov19: win %": round(l["win"], 1), "CONF total": round(l["tot"]), "CONF profit/DD": round(l["ratio"], 1), "$10 total": round(s10["tot"])})
Rr = pd.DataFrame(rows); print("\n2) POLYMARKET FILLS, Jun 1 - Nov 19, trading hours.  $200 stakes limited to the dollars sold into us; SEL = Jun 1 - Sep 30 (choice), CONF = Oct 1 - Nov 19 (judgement)"); print(Rr.to_string(index=False))
c = Rr[Rr["SEL Jun-Sep: /day"] >= 3].sort_values("SEL profit/DD", ascending=False); print("\n3) PRE-DECLARED CHOICE (best profit/DD on Jun-Sep at $200 among rules with >= 3 trades a day):"); print(c.head(5)[["rule", "SEL Jun-Sep: /day", "SEL win %", "SEL profit/DD", "CONF Oct-Nov19: win %", "CONF total", "CONF profit/DD"]].to_string(index=False))
cur = Rr[Rr.rule == "V0 m>=1.0"].iloc[0]; print(f"   current rule for comparison: SEL profit/DD {cur['SEL profit/DD']}, CONF win {cur['CONF Oct-Nov19: win %']} %, CONF total {cur['CONF total']}, CONF profit/DD {cur['CONF profit/DD']}")
Rr.to_csv(f"{DEEP}strong_lab_results.csv", index=False)
A.to_parquet(f"{DEEP}strong_lab_fills.parquet", index=False)
# ---------------------------------------------------------------- 4) the main candidates side by side, by month, at $200 and at $10
print("\n4) MAIN CANDIDATES, by month (profit $) and risk, at $200 and at $10 (at $10 every order fills completely)")
A["mon"] = pd.to_datetime(A["T"], unit="s").dt.strftime("%b"); rows = []
for lbl, v, m in (("current (V0, m>=1.0)", "V0", 1.0), ("1.7x (V0, m>=1.7)", "V0", 1.7), ("1.7x + strong T-45 (V1, m>=1.7)", "V1", 1.7), ("1.5x + strong T-45 (V1, m>=1.5)", "V1", 1.5), ("current + strong T-45 (V1, m>=1.0)", "V1", 1.0)):
    g = A[(A["var"] == v) & (A.m >= m)]
    for stake in (200, 10):
        s_ = np.minimum(stake, g.va.values) if stake > 50 else np.full(len(g), float(stake)); p = np.where(g.won.values, 0.5, -0.5) * s_ / g.e.values + s_ / g.e.values * 0.0035; st = stats(g, stake, D_ALL)
        rows.append({"rule": lbl, "stake": f"${stake}", "trades/day": round(st["per_day"], 1), "win %": round(st["win"], 1), "total": round(st["tot"]), **{mm: round(p[g.mon.values == mm].sum()) for mm in ("Jun", "Jul", "Aug", "Sep", "Oct", "Nov")}, "max DD": round(st["dd"]), "profit/DD": round(st["ratio"], 1), "worst day": round(st["worst"]), "losing days": st["lose_days"], "losing weeks": f"{st['lose_wks']}/{st['wks']}"})
print(pd.DataFrame(rows).to_string(index=False))
