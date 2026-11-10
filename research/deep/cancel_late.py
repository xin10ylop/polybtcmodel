"""Live rule A (BTC): the 0.50 bid placed at T-43 and kept until T-30 ... T+120 (the open is T+0), taking the offer when the bid would cross at
placement.  The sell at 0.85 / 0.90 counts only AFTER the fill; else settlement.   python research/deep/cancel_late.py"""
import sys, os, glob, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, DEEP
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; ASSET = "BTC"; T_IN = -43; TCS = [-30, -15, -5, 0, 5, 10, 20, 30, 60, 120]; CACHE = f"{DEEP}/cancel_late_{ASSET}.pkl"
W = load(ASSET, ["m2_-45_close_p"]); mon = W.month.values; n = len(W)
sc = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_ph5.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
sel = np.zeros(n, bool)
for mo in MONTHS:
    te = (mon == mo) & np.isfinite(sc); sel[te] = np.abs(sc[te]) >= np.quantile(np.abs(sc[before(mo) & np.isfinite(sc)]), 0.85)
up = sc > 0; S85 = sims_cached(ASSET, 0.85).reindex(W.index); SN = sims_cached(ASSET, None).reindex(W.index)
p85 = np.where(up, S85.pnl_up.values, S85.pnl_dn.values); pset = np.where(up, SN.pnl_up.values, SN.pnl_dn.values); h85 = np.where(up, S85.tp_up.values, S85.tp_dn.values); m = sel & np.isfinite(p85)
D = pd.DataFrame({"T": W.index.values[m], "date": pd.to_datetime(W.date.values[m]), "s": np.where(up[m], 0, 1), "p85": p85[m], "win": pset[m] > 0, "h85": h85[m]}).set_index("T")
if os.path.exists(CACHE): R = pickle.load(open(CACHE, "rb"))
else:
    R = {}
    for f in sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet")):
        t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t = t[t.window_start.isin(D.index)]
        if not len(t): continue
        t["rel"] = t.ts - t.window_start; t = t[(t.rel >= -110) & (t.rel <= 300)].sort_values(["window_start", "ts"], kind="stable")
        for w, g in t.groupby("window_start", sort=False):
            s = D.at[w, "s"]; rel = g.rel.values; ours = g.outcome_index.values == s; buy = g.side.values == "BUY"; ps = np.where(ours, g.price.values, 1 - g.price.values)
            lift = (ours & buy) | (~ours & ~buy); usd = g.price.values * g["size"].values
            a = lift & (rel < T_IN) & (rel >= T_IN - 60); b = lift & (rel >= T_IN) & (rel < T_IN + 8)
            a0 = ps[a][-1] if a.any() else (ps[b][0] if b.any() else np.nan)
            hit = ~lift & (ps <= 0.50 + 1e-9) & (rel >= T_IN)
            def tp_after(t0): 
                q = lift & (rel > t0); mx = ps[q].max() if q.any() else 0.0; return mx
            ft = rel[hit].min() if hit.any() else np.nan
            R[w] = dict(a0=a0, ft=ft, mx_fill=tp_after(ft) if np.isfinite(ft) else 0.0, mx_in=tp_after(T_IN), vol=[float(usd[hit & (rel < tc)].sum()) for tc in TCS])
    pickle.dump(R, open(CACHE, "wb"))
g = lambda k: np.array([R.get(w, {}).get(k, np.nan) for w in D.index], dtype=float)
a0, ft, mxf, mxi = g("a0"), g("ft"), g("mx_fill"), g("mx_in"); vol = np.array([R.get(w, {}).get("vol", [0.0] * len(TCS)) for w in D.index]); win = D.win.values; nd = 160
agree = (((mxi >= 0.85) == D.h85.values).mean(), ((mxi > 0.85 + 1e-9) == D.h85.values).mean()); strict = agree[1] > agree[0]
print(f"check of my sell-fill rule against the backtest's (fill at T-45): touch {agree[0]:.1%}, through {agree[1]:.1%} -> using {'through' if strict else 'touch'}")
def hitp(mx, tp): return (mx > tp + 1e-9) if strict else (mx >= tp - 1e-9)
cross = np.isfinite(a0) & (a0 <= 0.50 + 1e-9); rng = np.random.default_rng(11); PERS = {"Jun1-Aug19": D.date < "2026-08-20", "tune": (D.date >= "2026-08-20") & (D.date <= "2026-09-30"), "TEST": D.date >= "2026-10-01"}
print(f"{len(D)/nd:.1f} signals/day; bid crosses at placement (taken with the fee): {cross.sum()/nd:.1f}/day")
print("\nfills by arrival time (resting bid only):   share of signals   win at settlement   reached 0.85 after the fill   c/share (sell 0.85)")
edges = [T_IN] + TCS
for lo, hi in zip(edges[:-1], edges[1:]):
    f = ~cross & (ft >= lo) & (ft < hi)
    if f.sum() < 20: continue
    v = np.where(hitp(mxf, 0.85), 0.35, np.where(win, 0.5, -0.5)); print(f"   T{lo:+4d} .. T{hi:+4d}                              {f.mean():>5.1%}             {win[f].mean():.1%}               {hitp(mxf, 0.85)[f].mean():.1%}                    {v[f].mean()*100:+.2f}")
for tp in (0.85, 0.90, None):
    print(f"\n=== sell at {tp if tp else 'settlement only'} ===\ncancel at   trades/day  win    c/share [95% CI]        $/day@$5   $/day@$100 (volume-capped)   c/share: Jun1-Aug19 | tune | TEST")
    for j, tc in enumerate(TCS):
        mk = ~cross & (ft < tc); f = cross | mk; e = np.where(cross, a0, 0.50); fee = np.where(cross, 0.07 * a0 * (1 - a0), 0.0)
        mx = np.where(cross, mxi, mxf); hp = hitp(mx, tp) if tp else np.zeros(len(D), bool)
        v = np.where(hp, (tp or 0) - e, np.where(win, 1 - e, -e)) - fee; v = np.where(f, v, 0.0)
        d = pd.Series(v).groupby(D.date.values).sum(); c = pd.Series(f.astype(float)).groupby(D.date.values).sum(); bs = [d.values[k].sum() / max(c.values[k].sum(), 1) * 100 for k in (rng.integers(0, len(d), len(d)) for _ in range(1000))]
        capd = np.where(cross, 100.0, np.minimum(100.0, vol[:, j])); u100 = (v * capd / np.where(f, e, 1)).sum() / nd
        per = " | ".join(f"{v[f & np.asarray(PERS[k])].mean()*100:+.2f}" for k in PERS)
        print(f"  T{tc:+4d}       {f.sum()/nd:>5.1f}    {(v[f] > 0).mean():.1%}   {v[f].mean()*100:+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]      {(v / np.where(f, e, 1) * 5).sum()/nd:>6.2f}      {u100:>7.1f}                   {per}")
