import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, DEEP
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; ASSET = "BTC"
W = load(ASSET, ["m2_-45_close_p"]); S = sims_cached(ASSET, 0.85).reindex(W.index); mon = W.month.values; n = len(W)
sc = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_ph5.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
sel = np.zeros(n, bool)
for mo in MONTHS:
    te = (mon == mo) & np.isfinite(sc); sel[te] = np.abs(sc[te]) >= np.quantile(np.abs(sc[before(mo) & np.isfinite(sc)]), 0.85)
up = sc > 0; pnl = np.where(up, S.pnl_up.values, S.pnl_dn.values); m = sel & np.isfinite(pnl)
D = pd.DataFrame({"T": W.index.values[m], "date": W.date.values[m], "s": np.where(up[m], 0, 1), "pnl": pnl[m]}).set_index("T"); px = {}
for f in sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "window_start"]); t = t[t.window_start.isin(D.index)]
    if not len(t): continue
    t["rel"] = t.ts - t.window_start; t = t[(t.rel >= -45) & (t.rel < -30) & (t.side == "BUY")]; t = t[t.outcome_index.values == D.s.reindex(t.window_start).values]
    for w, g in t.groupby("window_start"): px[w] = float(g.sort_values("ts").price.iloc[0])      # first market buy of our side after the decision = the offer price then
D["ask"] = pd.Series(px).reindex(D.index); E = D[D.ask.notna()].copy(); E["fee"] = 0.07 * E.ask * (1 - E.ask); E["tk"] = E.pnl + 0.50 - E.ask - E.fee; nd = 160; rng = np.random.default_rng(2)
def ci(g):
    d = g.groupby("date").tk.sum(); c = g.groupby("date").size(); b = [d.values[i].sum() / c.values[i].sum() * 100 for i in (rng.integers(0, len(d), len(d)) for _ in range(2000))]; return np.percentile(b, [2.5, 97.5])
print(f"BTC live rule, {len(D)} signals; offer price known (a market buy of our side printed T-45..T-30) for {len(E)/len(D):.0%}")
print(f"cost of a market buy vs a fill at 0.50: price {100*(E.ask.mean()-0.5):.2f} c + fee {100*E.fee.mean():.2f} c = {100*(E.ask.mean()-0.5+E.fee.mean()):.2f} c/share; edge if filled at 0.50 on these signals {100*E.pnl.mean():+.2f} c")
print("\nmarket buy only when the offer is at most ...   signals/day   win rate   c/share after fee [95% CI]   $/day at $5   $/day at $100")
for cap in (0.50, 0.51, 0.52, 0.53, 0.55, 0.60, 1.0):
    g = E[E.ask <= cap + 1e-9]
    if len(g) < 50: continue
    c = ci(g); sh5 = (5 / g.ask); print(f"   {('any price' if cap == 1.0 else f'{cap:.2f}'):>10}                               {len(g)/nd:>5.1f}       {(g.pnl > 0).mean():.1%}     {g.tk.mean()*100:+.2f} [{c[0]:+.2f}, {c[1]:+.2f}]          {(g.tk * sh5).sum()/nd:+.2f}        {(g.tk * sh5).sum()/nd*20:+.1f}")
print("\nby offer price band:   signals   win rate   c/share if filled at 0.50   c/share market buy after fee")
for lo, hi in ((0, 0.505), (0.505, 0.515), (0.515, 0.525), (0.525, 0.545), (0.545, 0.60), (0.60, 1.0)):
    g = E[(E.ask > lo) & (E.ask <= hi)]
    if len(g): print(f"   {lo:.3f}-{hi:.3f}        {len(g):>5}    {(g.pnl > 0).mean():.1%}        {g.pnl.mean()*100:+.2f}                    {g.tk.mean()*100:+.2f}")
