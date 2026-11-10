"""Monthly table at $100 stakes, BTC, same format as the backtest table of new A (every signal bought at 0.50, sell 0.85 else settle), next to
the same rule with fills taken from the real trade tape: the setup that was live on Nov 10 and the T-120 setup.  python research/deep/monthly_real.py"""
import sys, os, glob, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, DEEP
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; ASSET = "BTC"; STAKE = 100.0; CACHE = f"{DEEP}/monthly_real_{ASSET}.pkl"
SET = {"live": dict(file="ourcase_BTC_ph5", dec=-45, tc=-30, cross=False), "best45": dict(file="ourcase_BTC_ph5", dec=-45, tc=-15, cross=True), "t120": dict(file="ourcase_BTC_d-120_ph3", dec=-120, tc=-15, cross=True)}
W = load(ASSET, ["m2_-45_close_p"]); mon = W.month.values; n = len(W); S85 = sims_cached(ASSET, 0.85).reindex(W.index); SN = sims_cached(ASSET, None).reindex(W.index); upwin = SN.pnl_up.values > 0; ok = np.isfinite(SN.pnl_up.values)
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
SIG = {}
for k, c in SET.items():
    sc = pd.read_parquet(f"{DEEP}/{c['file']}.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values; sel = np.zeros(n, bool)
    for mo in MONTHS:
        te = (mon == mo) & np.isfinite(sc); sel[te] = np.abs(sc[te]) >= np.quantile(np.abs(sc[before(mo) & np.isfinite(sc)]), 0.85)
    m = sel & ok; SIG[k] = pd.DataFrame({"s": np.where(sc[m] > 0, 0, 1), "win": np.where(sc[m] > 0, upwin[m], ~upwin[m]), "mon": mon[m], "date": pd.to_datetime(W.date.values[m]),
                                         "p85": np.where(sc[m] > 0, S85.pnl_up.values[m], S85.pnl_dn.values[m])}, index=W.index.values[m])
if os.path.exists(CACHE): R = pickle.load(open(CACHE, "rb"))
else:
    R = {k: {} for k in SET}; allw = set().union(*[set(v.index) for v in SIG.values()])
    for f in sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet")):
        t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t = t[t.window_start.isin(allw)]
        if not len(t): continue
        t["rel"] = t.ts - t.window_start; t = t[(t.rel >= -200) & (t.rel <= 300)].sort_values(["window_start", "ts"], kind="stable")
        for w, g in t.groupby("window_start", sort=False):
            rel = g.rel.values; pr = g.price.values; o0 = g.outcome_index.values == 0; buy = g.side.values == "BUY"; usd = pr * g["size"].values
            for k, c in SET.items():
                if w not in SIG[k].index: continue
                s = SIG[k].at[w, "s"]; ours = o0 if s == 0 else ~o0; ps = np.where(ours, pr, 1 - pr); lift = (ours & buy) | (~ours & ~buy); t_in = c["dec"] + 2; tc = c["tc"]
                a = lift & (rel < t_in) & (rel >= t_in - 60); b = lift & (rel >= t_in) & (rel < t_in + 8); a0 = ps[a][-1] if a.any() else (ps[b][0] if b.any() else np.nan)
                hit = ~lift & (ps <= 0.50 + 1e-9) & (rel >= t_in) & (rel < tc); ft = rel[hit].min() if hit.any() else np.nan
                q1 = lift & (rel > (ft if np.isfinite(ft) else 1e9)); q2 = lift & (rel > t_in); qc = lift & (ps <= 0.50 + 1e-9) & (rel >= t_in - 2) & (rel < t_in + 6)
                R[k][w] = (a0, ft, ps[q1].max() if q1.any() else 0.0, ps[q2].max() if q2.any() else 0.0, usd[hit].sum(), usd[qc].sum())
    pickle.dump(R, open(CACHE, "wb"))
def table(k):
    c = SET[k]; D = SIG[k]; arr = np.array([R[k].get(w, (np.nan, np.nan, 0.0, 0.0, 0.0, 0.0)) for w in D.index], dtype=float); a0, ft, mxf, mxi, vmk, vcr = arr.T; win = D.win.values
    crossing = np.isfinite(a0) & (a0 <= 0.50 + 1e-9); cross = crossing & c["cross"]; mk = ~crossing & np.isfinite(ft); f = cross | mk
    e = np.where(cross, a0, 0.50); fee = np.where(cross, 0.07 * a0 * (1 - a0), 0.0); mx = np.where(cross, mxi, mxf)
    v = np.where(mx >= 0.85 - 1e-9, 0.85 - e, np.where(win, 1 - e, -e)) - fee; full = np.where(f, v * STAKE / e, 0.0); capd = np.minimum(STAKE, np.where(cross, np.maximum(vcr, 5.0), vmk)); cap = np.where(f, v * capd / e, 0.0)
    return D, f, v, full, cap, capd
def eq_dd(x): e = np.concatenate([[0], np.cumsum(x)]); return (np.maximum.accumulate(e) - e).max()
D0 = SIG["live"]; base = D0.p85.values * 200
print("BTC, $100 a trade, sell 0.85 else settle.  Jun 1 - Nov 7 2026\n")
print(f"{'month':9s} | {'BACKTEST TABLE: every signal':^30s} | {'LIVE Nov 10: T-45, bid 0.50':^44s} | {'T-45, bid to T-15, take':^44s} | {'T-120, bid to T-15, take':^44s}")
print(f"{'':9s} | {'bought at 0.50, full $100':^30s} | {'to T-30, never a fee':^44s} | {'when it crosses':^44s} | {'when it crosses':^44s}")
print(f"{'':9s} | {'trades / win / P&L':^30s} | " + " | ".join(f"{'trades / win / P&L full $100 / P&L real size':^44s}" for _ in range(3)))
T = {k: table(k) for k in SET}; tot = {}
for mo in MONTHS + ["total"]:
    mm0 = np.ones(len(D0), bool) if mo == "total" else (D0.mon.values == mo); line = f"{('Nov 1-7' if mo == 'Nov' else mo):9s} | {mm0.sum():>6,d} / {(D0.p85.values[mm0] > 0).mean():.1%} / ${base[mm0].sum():>8,.0f}   "
    for k in SET:
        D, f, v, full, cap, capd = T[k]; mm = np.ones(len(D), bool) if mo == "total" else (D.mon.values == mo); g = f & mm
        line += f" | {g.sum():>6,d} / {(v[g] > 0).mean():.1%} / ${full[mm].sum():>8,.0f} / ${cap[mm].sum():>8,.0f}       "
    print(line)
print()
for k, nm in (("live", "LIVE Nov 10"), ("best45", "T-45 improved"), ("t120", "T-120")):
    D, f, v, full, cap, capd = T[k]; print(f"{nm:14s}: share of signals that trade {f.mean():.0%}; average size really filled ${capd[f].mean():.0f} of $100 (median ${np.median(capd[f]):.0f}); P&L per day ${full.sum()/160:,.0f} at full size, ${cap.sum()/160:,.0f} at the size that traded; "
                                                  f"max drawdown ${eq_dd(full):,.0f} / ${eq_dd(cap):,.0f}; share of the backtest table: {full.sum()/base.sum():.0%} / {cap.sum()/base.sum():.0%}")
print(f"backtest table: ${base.sum()/160:,.0f} per day, max drawdown ${eq_dd(base):,.0f}")
