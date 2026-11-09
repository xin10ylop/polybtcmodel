"""Deep dive 3b: the entry, optimised under REAL maker fills (research/deep/maker_sim.py).
Signal = the purpose-built model's score; windows are ranked by |score| against the months before (walk-forward quantiles) into tiers.
For every tier x bid price x cancel time: fill rate, edge per filled share, $ per day (stake capped by what actually printed against the bid).
Selection on TUNE only (bid and cancel per tier maximising $/day under the conservative 'through' fill); every period reported.
  ASSET=BTC NEW=ph5book:lgb_reg300 python research/deep/entry_opt.py"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 280)
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, PER, DEEP  # noqa: E402
ASSET = os.environ.get("ASSET", "BTC"); NEW = os.environ.get("NEW", "ph5book:lgb_reg300"); STAKE = float(os.environ.get("STAKE", "100" if ASSET == "BTC" else "25"))
Z = np.load(f"{DEEP}/maker_{ASSET}.npz"); BIDS, CANCELS, TP = Z["bids"], Z["cancels"], float(Z["tp"])
W = load(ASSET, ["m2_-45_close_p", "bp_-60_z_s5"]); W = W[W.index.isin(Z["W"])]; pos = pd.Series(np.arange(len(Z["W"])), index=Z["W"]).reindex(W.index).values
fill, tph, usd, up = Z["fill"][pos], Z["tph"][pos], Z["usd"][pos], Z["up"][pos]
tag, col = NEW.split(":"); score = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_{tag}.parquet").set_index("T")[col].reindex(W.index).values
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; mon = W.month.values; per = W.period.values; date = W.date.values
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
EDG = [0.0, 0.05, 0.15, 0.30, 0.50, 0.75, 1.0]; TIERS = ["top 5%", "5-15%", "15-30%", "30-50%", "50-75%", "75-100%"]; tier = np.full(len(W), -1)
for mo in MONTHS:
    te = (mon == mo) & np.isfinite(score); a = np.abs(score[before(mo) & np.isfinite(score)]); q = [np.quantile(a, 1 - e) if e < 1 else -1.0 for e in EDG[1:]]
    r = np.abs(score[te]); tier[te] = np.select([r >= q[0], r >= q[1], r >= q[2], r >= q[3], r >= q[4]], [0, 1, 2, 3, 4], 5)
side = np.where(score > 0, 0, 1); idx = np.arange(len(W)); win = np.where(side == 0, up, ~up)
def trade(b, c, k):
    """pnl per share (NaN if unfilled) and $ P&L at STAKE for bid index b, cancel index c, fill definition k (0 exact, 1 through)."""
    fr = fill[idx, side, b, k]; f = np.isfinite(fr) & (fr <= CANCELS[c]); pnl = np.where(f, np.where(tph[idx, side, b, k], TP - BIDS[b], win.astype(float) - BIDS[b]), np.nan)
    return pnl, np.where(f, pnl * np.minimum(STAKE, usd[idx, side, b, c]) / BIDS[b], 0.0)
nd = {p_: len(pd.date_range(*PER[p_])) for p_ in PER}; rows = []
for ti, tn in enumerate(TIERS):
    for b in range(len(BIDS)):
        for c in range(len(CANCELS)):
            for k, kn in enumerate(("exact", "through")):
                pnl, d = trade(b, c, k)
                for p_ in PER:
                    m = (tier == ti) & (per == p_); f = m & np.isfinite(pnl)
                    if m.sum() == 0: continue
                    rows.append(dict(tier=tn, bid=BIDS[b], cancel=CANCELS[c], fills=kn, period=p_, signals=int(m.sum()), fill_rate=f.sum() / m.sum(), win=(pnl[f] > 0).mean() if f.sum() else np.nan,
                                     edge_c=pnl[f].mean() * 100 if f.sum() else np.nan, usd_day=d[m].sum() / nd[p_]))
E = pd.DataFrame(rows); E.to_csv(f"/home/user/polybtcmodel/reports/deep_entry_grid_{ASSET}.csv", index=False)
print(f"{ASSET}: {len(W)} windows, stake ${STAKE:.0f}, sell {TP} else settle\n\n== edge per filled share (cents), 'through' fills, cancel at T-30, pooled over all periods: tier x bid ==")
pool = E[(E.fills == "through") & (E.cancel == -30)].assign(w=lambda d: d.signals * d.fill_rate).groupby(["tier", "bid"]).apply(lambda d: pd.Series({"edge": np.nansum(d.edge_c * d.w) / d.w.sum(), "fill": d.w.sum() / d.signals.sum(), "usd_day": d.usd_day.mean()}))
for mname in ("edge", "fill", "usd_day"): print(f"-- {mname} --"); print(pool[mname].unstack("bid").reindex(TIERS).round(2 if mname == "fill" else 1).to_string())
# selection on tune: per tier the (bid, cancel) with the best $/day under 'through' fills; trade the tier only if that is > 0
t = E[(E.period == "tune") & (E.fills == "through")]; CH = {}
for tn in TIERS:
    g = t[t.tier == tn].sort_values("usd_day", ascending=False).iloc[0]
    if g.usd_day > 0: CH[tn] = (int(np.argmin(np.abs(BIDS - g.bid))), int(np.argmin(np.abs(CANCELS - g.cancel))))
print("\nchosen on tune (tier -> bid, cancel):", {k: (float(BIDS[b]), int(CANCELS[c])) for k, (b, c) in CH.items()})
def portfolio(choice, k, label):
    tot = np.zeros(len(W)); pn = np.full(len(W), np.nan)
    for tn, (b, c) in choice.items():
        pnl, d = trade(b, c, k); m = tier == TIERS.index(tn); tot[m] = d[m]; pn[m] = pnl[m]
    out = []
    for p_ in PER:
        m = (per == p_); f = m & np.isfinite(pn); dd = pd.Series(tot[m]).groupby(date[m]).sum().reindex(pd.date_range(*PER[p_]), fill_value=0.0); eq = dd.cumsum()
        out.append(dict(rule=label, period=p_, fills_per_day=f.sum() / nd[p_], win=(pn[f] > 0).mean(), edge_c=pn[f].mean() * 100, usd_day=dd.mean(), maxdd=(eq.cummax() - eq).max(), worst_day=dd.min(), neg_days=int((dd < 0).sum())))
    return out
pm = W["m2_-45_close_p"].values; z5 = W["bp_-60_z_s5"].values; selA = np.isfinite(pm) & (np.abs(pm - 0.5) >= 0.05) & np.isfinite(z5) & (np.sign(z5) == np.sign(pm - 0.5))
def frozenA(k, label, b=3, c=0):
    s = np.where(pm > 0.5, 0, 1); fr = fill[idx, s, b, k]; f = selA & np.isfinite(fr) & (fr <= CANCELS[c]); w_ = np.where(s == 0, up, ~up)
    pn = np.where(f, np.where(tph[idx, s, b, k], TP - BIDS[b], w_.astype(float) - BIDS[b]), np.nan); tot = np.where(f, pn * np.minimum(STAKE, usd[idx, s, b, c]) / BIDS[b], 0.0); out = []
    for p_ in PER:
        m = per == p_; ff = m & np.isfinite(pn); dd = pd.Series(tot[m]).groupby(date[m]).sum().reindex(pd.date_range(*PER[p_]), fill_value=0.0); eq = dd.cumsum()
        out.append(dict(rule=label, period=p_, fills_per_day=ff.sum() / nd[p_], win=(pn[ff] > 0).mean() if ff.sum() else np.nan, edge_c=pn[ff].mean() * 100 if ff.sum() else np.nan, usd_day=dd.mean(), maxdd=(eq.cummax() - eq).max(), worst_day=dd.min(), neg_days=int((dd < 0).sum())))
    return out
R = frozenA(0, "FROZEN A, bid 0.50 cancel T-40 [exact]") + frozenA(1, "FROZEN A, bid 0.50 cancel T-40 [through]") + frozenA(1, "FROZEN A, bid 0.50 cancel T-30 [through]", c=1)
R += portfolio(CH, 1, "NEW tiers, tune-chosen bids [through]") + portfolio(CH, 0, "NEW tiers, tune-chosen bids [exact]")
top = {k: v for k, v in CH.items() if k in ("top 5%", "5-15%")}; R += portfolio(top, 1, "NEW top 15% only [through]")
top3 = {k: v for k, v in CH.items() if k in ("top 5%", "5-15%", "15-30%")}; R += portfolio(top3, 1, "NEW top 30% only [through]")
flat = {tn: (3, 1) for tn in TIERS[:4]}; R += portfolio(flat, 1, "NEW top 50%, flat bid 0.50 cancel T-30 [through]")
P = pd.DataFrame(R); P.to_csv(f"/home/user/polybtcmodel/reports/deep_entry_portfolio_{ASSET}.csv", index=False)
for mname in ("fills_per_day", "win", "edge_c", "usd_day", "maxdd", "worst_day"):
    print(f"\n-- {mname} --"); print(P.pivot(index="rule", columns="period", values=mname).reindex(P.rule.unique())[list(PER)].round(3 if mname == "win" else 1).to_string())
