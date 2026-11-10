"""Would an earlier decision fill better?  Walk-forward models at T-120 / T-90 / T-60 / T-45 (top 15 % of |score|), 0.50 bid placed 2 s after the
decision, taken when it would cross, kept until `cancel`; sell 0.85 after the fill, else settlement.   python research/deep/early_decision.py"""
import sys, os, glob, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, DEEP
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; ASSET = "BTC"; TCS = [-75, -60, -45, -30, -15, 0]; CACHE = f"{DEEP}/early_decision_{ASSET}.pkl"
FILES = {-120: "ourcase_BTC_d-120_ph3book", -90: "ourcase_BTC_d-90_ph3book", -60: "ourcase_BTC_d-60_ph3book", -45: "ourcase_BTC_ph5"}
W = load(ASSET, ["m2_-45_close_p"]); mon = W.month.values; n = len(W); SN = sims_cached(ASSET, None).reindex(W.index); upwin = SN.pnl_up.values > 0; ok = np.isfinite(SN.pnl_up.values)
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
SIG = {}
for dec, fn in FILES.items():
    sc = pd.read_parquet(f"{DEEP}/{fn}.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values; sel = np.zeros(n, bool)
    for mo in MONTHS:
        te = (mon == mo) & np.isfinite(sc); sel[te] = np.abs(sc[te]) >= np.quantile(np.abs(sc[before(mo) & np.isfinite(sc)]), 0.85)
    m = sel & ok; SIG[dec] = pd.DataFrame({"s": np.where(sc[m] > 0, 0, 1), "win": np.where(sc[m] > 0, upwin[m], ~upwin[m]), "date": pd.to_datetime(W.date.values[m])}, index=W.index.values[m])
if os.path.exists(CACHE): R = pickle.load(open(CACHE, "rb"))
else:
    R = {d: {} for d in FILES}; allw = set().union(*[set(v.index) for v in SIG.values()])
    for f in sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet")):
        t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "window_start"]); t = t[t.window_start.isin(allw)]
        if not len(t): continue
        t["rel"] = t.ts - t.window_start; t = t[(t.rel >= -200) & (t.rel <= 300)].sort_values(["window_start", "ts"], kind="stable")
        for w, g in t.groupby("window_start", sort=False):
            rel = g.rel.values; pr = g.price.values; o0 = g.outcome_index.values == 0; buy = g.side.values == "BUY"
            for dec in FILES:
                if w not in SIG[dec].index: continue
                s = SIG[dec].at[w, "s"]; ours = o0 if s == 0 else ~o0; ps = np.where(ours, pr, 1 - pr); lift = (ours & buy) | (~ours & ~buy); t_in = dec + 2
                a = lift & (rel < t_in) & (rel >= t_in - 60); b = lift & (rel >= t_in) & (rel < t_in + 8); a0 = ps[a][-1] if a.any() else (ps[b][0] if b.any() else np.nan)
                hit = ~lift & (ps <= 0.50 + 1e-9) & (rel >= t_in); ft = rel[hit].min() if hit.any() else np.nan
                q1 = lift & (rel > (ft if np.isfinite(ft) else 1e9)); q2 = lift & (rel > t_in)
                R[dec][w] = (a0, ft, ps[q1].max() if q1.any() else 0.0, ps[q2].max() if q2.any() else 0.0)
    pickle.dump(R, open(CACHE, "wb"))
nd = 160; rng = np.random.default_rng(5)
print("decision   signals/day  win if all filled   offer at the decision (mean)  crosses/day |  cancel at  trades/day  win    c/share [95% CI]      $/day@$5   c/share: Jun1-Aug19 | tune | TEST")
for dec in FILES:
    D = SIG[dec]; arr = np.array([R[dec].get(w, (np.nan, np.nan, 0.0, 0.0)) for w in D.index], dtype=float); a0, ft, mxf, mxi = arr.T; win = D.win.values
    cross = np.isfinite(a0) & (a0 <= 0.50 + 1e-9); PERS = {"a": D.date < "2026-08-20", "b": (D.date >= "2026-08-20") & (D.date <= "2026-09-30"), "c": D.date >= "2026-10-01"}
    head = f"T{dec:+4d}       {len(D)/nd:>5.1f}        {win.mean():.1%}                 {np.nanmean(a0):.3f}                    {cross.sum()/nd:>4.1f}"
    for tc in [x for x in TCS if x > dec + 10]:
        mk = ~cross & (ft < tc); f = cross | mk; e = np.where(cross, a0, 0.50); fee = np.where(cross, 0.07 * a0 * (1 - a0), 0.0); mx = np.where(cross, mxi, mxf)
        v = np.where(mx >= 0.85 - 1e-9, 0.85 - e, np.where(win, 1 - e, -e)) - fee; v = np.where(f, v, 0.0)
        d = pd.Series(v).groupby(D.date.values).sum(); c = pd.Series(f.astype(float)).groupby(D.date.values).sum(); bs = [d.values[k].sum() / max(c.values[k].sum(), 1) * 100 for k in (rng.integers(0, len(d), len(d)) for _ in range(1000))]
        per = " | ".join(f"{v[f & np.asarray(PERS[k])].mean()*100:+.2f}" for k in PERS)
        print(f"{head if tc == [x for x in TCS if x > dec + 10][0] else ' ' * len(head)} |   T{tc:+4d}     {f.sum()/nd:>5.1f}   {(v[f] > 0).mean():.1%}   {v[f].mean()*100:+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]    {(v / np.where(f, e, 1) * 5).sum()/nd:>6.2f}     {per}")
