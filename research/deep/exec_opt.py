"""Execution of the live rule A (BTC): which way of entering captures most of the 'every signal filled at 0.50' result.
One pre-specified family: resting bid b from T-43 to tc (maker, no fee); optionally take the offer when the bid would cross at placement;
optionally, if still unfilled at tc, take the offer when it is <= cap (taker: offer + fee 0.07 p (1-p)).  Exit unchanged (sell 0.85 resting, else settle).
Chosen on the tune period (Aug 20 - Sep 30), reported on the test period (Oct 1 - Nov 7) and on Jun 1 - Aug 19.
  python research/deep/exec_opt.py"""
import sys, os, glob, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, DEEP
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; ASSET = "BTC"; T_IN = -43; CACHE = f"{DEEP}/exec_prints_{ASSET}.pkl"
W = load(ASSET, ["m2_-45_close_p"]); S = sims_cached(ASSET, 0.85).reindex(W.index); mon = W.month.values; n = len(W)
sc = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_ph5.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
sel = np.zeros(n, bool)
for mo in MONTHS:
    te = (mon == mo) & np.isfinite(sc); sel[te] = np.abs(sc[te]) >= np.quantile(np.abs(sc[before(mo) & np.isfinite(sc)]), 0.85)
up = sc > 0; pnl = np.where(up, S.pnl_up.values, S.pnl_dn.values); m = sel & np.isfinite(pnl)
D = pd.DataFrame({"T": W.index.values[m], "date": pd.to_datetime(W.date.values[m]), "s": np.where(up[m], 0, 1), "pnl": pnl[m]}).set_index("T")
if os.path.exists(CACHE): P = pickle.load(open(CACHE, "rb"))
else:
    P = {}
    for f in sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet")):
        t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t = t[t.window_start.isin(D.index)]
        if not len(t): continue
        t["rel"] = t.ts - t.window_start; t = t[(t.rel >= -110) & (t.rel < 12)].sort_values(["window_start", "ts"], kind="stable")
        for w, g in t.groupby("window_start", sort=False):
            s = D.at[w, "s"]; ours = g.outcome_index.values == s; buy = g.side.values == "BUY"; ps = np.where(ours, g.price.values, 1 - g.price.values)
            lift = (ours & buy) | (~ours & ~buy)          # someone paid our side's offer (price ps)
            P[w] = (g.rel.values.astype(np.float32), ps.astype(np.float32), lift, (g.price.values * g["size"].values).astype(np.float32))
    pickle.dump(P, open(CACHE, "wb"))
def offer(rel, ps, lift, t, back=60, fwd=8):       # the offer of our side around time t: last price a market buyer paid just before, else the first just after
    a = lift & (rel < t) & (rel >= t - back)
    if a.any(): return ps[a][-1]
    b = lift & (rel >= t) & (rel < t + fwd)
    return ps[b][0] if b.any() else np.nan
def run(b, tc, cross, cap):
    """-> per signal: entry price (nan = no trade), fee, kind (0 none, 1 maker, 2 taker at placement, 3 taker late), $ capacity"""
    out = np.full((len(D), 4), np.nan); out[:, 2] = 0
    for i, w in enumerate(D.index):
        if w not in P: continue
        rel, ps, lift, usd = P[w]; a0 = offer(rel, ps, lift, T_IN)
        if np.isfinite(a0) and a0 <= b + 1e-9:                         # our bid would cross: refused by post-only, or taken with the fee
            if cross:
                q = lift & (rel >= T_IN - 2) & (rel < T_IN + 6) & (ps <= b + 1e-9); out[i] = (a0, 0.07 * a0 * (1 - a0), 2, usd[q].sum() if q.any() else 5.0)
            continue
        hit = ~lift & (ps <= b + 1e-9) & (rel >= T_IN) & (rel < tc)       # someone sold into a resting bid of our side at <= b
        if hit.any(): out[i] = (b, 0.0, 1, usd[hit].sum()); continue
        if cap is not None:
            a1 = offer(rel, ps, lift, tc, back=20, fwd=8)
            if np.isfinite(a1) and a1 <= cap + 1e-9:
                q = lift & (rel >= tc) & (rel < tc + 10) & (ps <= cap + 1e-9); out[i] = (a1, 0.07 * a1 * (1 - a1), 3, usd[q].sum() if q.any() else 5.0)
    return out
PER = {"tune Aug20-Sep30": (D.date >= "2026-08-20") & (D.date <= "2026-09-30"), "TEST Oct1-Nov7": (D.date >= "2026-10-01"), "Jun1-Aug19": (D.date < "2026-08-20"), "ALL": np.ones(len(D), bool)}
ND = {k: D.date[v.values if hasattr(v, "values") else v].nunique() for k, v in PER.items()}
FAM = [("A  bid 0.50 to T-30, no fee ever (LIVE NOW)", 0.50, -30, False, None), ("B  bid 0.50 to T-30 + take when it crosses", 0.50, -30, True, None),
       ("C  bid 0.50 to T-15 + cross", 0.50, -15, True, None), ("D  bid 0.50 to T-5 + cross", 0.50, -5, True, None),
       ("E  bid 0.51 to T-30 + cross", 0.51, -30, True, None), ("F  bid 0.52 to T-30 + cross", 0.52, -30, True, None), ("G  bid 0.51 to T-15 + cross", 0.51, -15, True, None)]
for capv in (0.51, 0.52, 0.53, 0.55): FAM.append((f"H  bid 0.50 to T-30 + cross, then take if offer <= {capv:.2f}", 0.50, -30, True, capv))
for capv in (0.51, 0.52, 0.53, 0.55): FAM.append((f"I  bid 0.50 to T-15 + cross, then take if offer <= {capv:.2f}", 0.50, -15, True, capv))
for capv in (0.51, 0.52, 0.53, 0.55, 0.99): FAM.append((f"J  market buy at T-43 if offer <= {capv:.2f}", 0.0, T_IN, True, capv))
rng = np.random.default_rng(7); res = {}
def stats(o, mask, nd, stake=5.0):
    e = o[:, 0]; f = np.isfinite(e) & mask; v = (D.pnl.values + 0.50 - e - o[:, 1]); sh = stake / np.where(f, e, 1)
    capd = np.minimum(100.0, o[:, 3]); usd5 = np.where(f, v * sh, 0.0); usd100 = np.where(f, v * capd / np.where(f, e, 1), 0.0)
    d = pd.Series(usd5[mask]).groupby(D.date.values[mask]).sum(); bs = [d.values[rng.integers(0, len(d), len(d))].sum() / nd for _ in range(1000)]
    return dict(n=f.sum() / nd, win=(v[f] > 0).mean() if f.any() else np.nan, c=v[f].mean() * 100 if f.any() else np.nan, d5=usd5.sum() / nd, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), d100=usd100.sum() / nd,
                mk=(o[f, 2] == 1).mean() if f.any() else np.nan)
ideal = {k: D.pnl.values[np.asarray(v)].sum() * 10 / ND[k] for k, v in PER.items()}
print(f"BTC live rule: {len(D)} signals. If every signal filled at 0.50 ($5 stake): " + ", ".join(f"{k} ${v:.1f}/day" for k, v in ideal.items()))
for nm, b, tc, cross, capv in FAM:
    o = run(b, tc, cross, capv) if b > 0 else None
    if b == 0:      # pure market buy at T-43
        o = np.full((len(D), 4), np.nan); o[:, 2] = 0
        for i, w in enumerate(D.index):
            if w not in P: continue
            rel, ps, lift, usd = P[w]; a0 = offer(rel, ps, lift, T_IN)
            if np.isfinite(a0) and a0 <= capv + 1e-9:
                q = lift & (rel >= T_IN - 2) & (rel < T_IN + 8) & (ps <= capv + 1e-9); o[i] = (a0, 0.07 * a0 * (1 - a0), 2, usd[q].sum() if q.any() else 5.0)
    res[nm] = {k: stats(o, np.asarray(v), ND[k]) for k, v in PER.items()}
for k in PER:
    print(f"\n=== {k} ({ND[k]} days; every signal filled at 0.50 would give ${ideal[k]:.1f}/day at $5) ===")
    print(f"{'entry method':62s} trades/day  win    c/share  $/day@$5 [95% CI]     % of ideal  $/day@$100(vol-capped)  maker share")
    for nm in res:
        r = res[nm][k]; print(f"{nm:62s} {r['n']:>6.1f}   {r['win']:.1%}  {r['c']:+.2f}   {r['d5']:>6.2f} [{r['lo']:>5.2f},{r['hi']:>6.2f}]    {r['d5']/ideal[k]:>5.0%}       {r['d100']:>7.1f}            {r['mk']:.0%}")
best = max(res, key=lambda q: res[q]["tune Aug20-Sep30"]["d5"]); print(f"\nBest on the tune period: {best}\n   on the TEST period: {res[best]['TEST Oct1-Nov7']}\n   on Jun1-Aug19:      {res[best]['Jun1-Aug19']}")
