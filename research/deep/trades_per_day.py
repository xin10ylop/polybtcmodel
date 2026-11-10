"""Live rule A on BTC: what happens to the 43 signals a day - refused by post-only (our side already offered at <= 0.50 when the bid goes in),
resting unfilled, or filled - for each cancel time.  python research/deep/trades_per_day.py"""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, DEEP
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; ASSET = "BTC"; T_IN = -43
W = load(ASSET, ["m2_-45_close_p"]); S = sims_cached(ASSET, 0.85).reindex(W.index); mon = W.month.values; n = len(W)
sc = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_ph5.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
sel = np.zeros(n, bool)
for mo in MONTHS:
    te = (mon == mo) & np.isfinite(sc); sel[te] = np.abs(sc[te]) >= np.quantile(np.abs(sc[before(mo) & np.isfinite(sc)]), 0.85)
up = sc > 0; pnl = np.where(up, S.pnl_up.values, S.pnl_dn.values); m = sel & np.isfinite(pnl)
D = pd.DataFrame({"T": W.index.values[m], "date": W.date.values[m], "s": np.where(up[m], 0, 1), "pnl": pnl[m]}).set_index("T"); rec = {}
for f in sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t = t[t.window_start.isin(D.index)]
    if not len(t): continue
    t["rel"] = t.ts - t.window_start; t = t[(t.rel >= -120) & (t.rel < 0)].sort_values(["window_start", "ts"], kind="stable")
    for w, g in t.groupby("window_start", sort=False):
        s = D.at[w, "s"]; rel = g.rel.values; ours = g.outcome_index.values == s; buy = g.side.values == "BUY"; ps = np.where(ours, g.price.values, 1 - g.price.values)
        tb = ours & buy                                              # market buys of our side: they pay the offer
        pre = tb & (rel < T_IN) & (rel >= T_IN - 60); post = tb & (rel >= T_IN) & (rel < T_IN + 5)
        ask = g.price.values[pre][-1] if pre.any() else (g.price.values[post][0] if post.any() else np.nan)
        hit = ((ours & ~buy) | (~ours & buy)) & (ps <= 0.50 + 1e-9) & (rel >= T_IN)     # prints that fill a resting 0.50 bid of our side
        rec[w] = (ask, rel[hit].min() if hit.any() else np.nan)
D["ask"] = [rec.get(w, (np.nan, np.nan))[0] for w in D.index]; D["fill_t"] = [rec.get(w, (np.nan, np.nan))[1] for w in D.index]; nd = 160
cross = D.ask <= 0.50 + 1e-9
print(f"BTC live rule: {len(D)/nd:.1f} signals a day (Jun 1 - Nov 7)")
print(f"  refused by post-only (our side already offered at 0.50 or less when the bid goes in): {cross.sum()/nd:.1f} a day ({cross.mean():.0%}); these would win {(D.pnl[cross] > 0).mean():.1%}, {D.pnl[cross].mean()*100:+.1f} c/share at 0.50")
print(f"  bid accepted on the book: {(~cross).sum()/nd:.1f} a day")
print("\ncancel at   filled trades/day   share of signals   win rate   c/share   $/day at $5   unfilled/day")
for c in (-30, -15, -5, 0):
    f = (~cross) & (D.fill_t < c); print(f"  T{c:+d}         {f.sum()/nd:>5.1f}            {f.mean():>5.0%}          {(D.pnl[f] > 0).mean():.1%}     {D.pnl[f].mean()*100:+.2f}      {D.pnl[f].sum()*10/nd:+.2f}        {((~cross) & ~f).sum()/nd:.1f}")
print("\nby month, filled trades per day with cancel at T-30:", {q: round(float((((~cross) & (D.fill_t < -30)) & (pd.to_datetime(D.date).dt.strftime('%b') == q).values).sum() / max(len(set(D.date[(pd.to_datetime(D.date).dt.strftime('%b') == q).values])), 1)), 1) for q in MONTHS})
