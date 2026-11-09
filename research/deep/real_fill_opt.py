"""Deep dive 3c: the pre-open rule optimised under REAL maker fills, including WHEN to bid.
Signal S_d = purpose-built model at decision time d (T-120 / -90 / -60 / -45).  A BUY of the signal's side rests at `bid` from T+d until
`cancel` (a later bucket edge, at most the open); it fills at the first print that trades against it (maker_sim2: 'through' = strictly
better than our bid, 'exact' = at our bid or better); exit: sell limit, else settle.  Tiers = walk-forward quantiles of |S_d|.
$ at stake S: min(S, $ of qualifying taker flow while the bid rested).  Option FLIP: an unfilled bid is cancelled at T-45 when the
T-45 model disagrees with it.  Selection on TUNE only, per (d, tier): bid and cancel maximising $/day under 'through' fills.
  ASSET=BTC python research/deep/real_fill_opt.py"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 280)
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, PER, DEEP  # noqa: E402
ASSET = os.environ.get("ASSET", "BTC"); STAKES = (100, 300, 1000) if ASSET == "BTC" else (25, 50, 100)
Z = np.load(f"{DEEP}/maker2_{ASSET}.npz"); BIDS, EDGES, TP = Z["bids"], Z["edges"], float(Z["tp"]); NB = len(EDGES) - 1
W = load(ASSET, ["m2_-45_close_p", "bp_-60_z_s5"]); W = W[W.index.isin(Z["W"])]; pos = pd.Series(np.arange(len(Z["W"])), index=Z["W"]).reindex(W.index).values
fill, tph, usd, up = Z["fill"][pos], Z["tph"][pos], Z["usd"][pos], Z["up"][pos]; per = W.period.values; date = W.date.values; n = len(W); idx = np.arange(n)
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; mon = W.month.values; nd = {p_: len(pd.date_range(*PER[p_])) for p_ in PER}
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
SIG = {}
for d, f in ((-45, f"{DEEP}/ourcase_{ASSET}_ph5book.parquet"), (-60, f"{DEEP}/ourcase_{ASSET}_d-60_ph3book.parquet"), (-90, f"{DEEP}/ourcase_{ASSET}_d-90_ph3book.parquet"), (-120, f"{DEEP}/ourcase_{ASSET}_d-120_ph3book.parquet")):
    if os.path.exists(f): SIG[d] = pd.read_parquet(f).set_index("T")["lgb_reg300"].reindex(W.index).values
EDG = [0.05, 0.15, 0.30, 0.50, 1.0]; TIERS = ["top 5%", "5-15%", "15-30%", "30-50%", "rest"]
def tiers(score):
    t = np.full(n, -1)
    for mo in MONTHS:
        te = (mon == mo) & np.isfinite(score); a = np.abs(score[before(mo) & np.isfinite(score)]); q = [np.quantile(a, 1 - e) for e in EDG[:-1]]; r = np.abs(score[te])
        t[te] = np.select([r >= q[0], r >= q[1], r >= q[2], r >= q[3]], [0, 1, 2, 3], 4)
    return t
def run(d, b, kend, k, flip=False):
    """pnl per share (NaN unfilled) and capacity $ for the signal at d, bid index b, resting through bucket kend, fill definition k."""
    sc = SIG[d]; s = np.where(sc > 0, 0, 1); k0 = int(np.flatnonzero(EDGES == d)[0]); win = np.where(s == 0, up, ~up); pn = np.full(n, np.nan); cap = np.zeros(n)
    k45 = int(np.flatnonzero(EDGES == -45)[0]); dis = (np.sign(SIG[-45]) != np.sign(sc)) if flip and d < -45 else np.zeros(n, bool)
    for kk in range(k0, kend + 1):
        live = ~(dis & (kk >= k45)); f = np.isfinite(fill[idx, s, b, kk, k]) & live; m = np.isnan(pn) & f
        pn[m] = np.where(tph[idx, s, b, kk, k], TP - BIDS[b], win.astype(float) - BIDS[b])[m]; cap += np.where(live, usd[idx, s, b, kk], 0.0)
    return pn, cap
rows = []; TIER = {d: tiers(SIG[d]) for d in SIG}
for d in SIG:
    k0 = int(np.flatnonzero(EDGES == d)[0])
    for b in range(len(BIDS)):
        for kend in range(k0, NB):
            for k, kn in enumerate(("exact", "through")):
                for flip in ((False, True) if d < -45 else (False,)):
                    pn, cap = run(d, b, kend, k, flip)
                    for ti, tn in enumerate(TIERS):
                        for p_ in PER:
                            m = (TIER[d] == ti) & (per == p_); f = m & np.isfinite(pn)
                            if m.sum() == 0: continue
                            rows.append(dict(d=d, tier=tn, bid=BIDS[b], cancel=int(EDGES[kend + 1]) if kend < NB - 1 else 0, kend=kend, fills=kn, flip=flip, period=p_, signals=int(m.sum()), fill_rate=f.sum() / m.sum(),
                                             win=(pn[f] > 0).mean() if f.sum() else np.nan, edge_c=pn[f].mean() * 100 if f.sum() else np.nan, usd_day=(pn[f] * np.minimum(STAKES[0], cap[f]) / BIDS[b]).sum() / nd[p_]))
E = pd.DataFrame(rows); E.to_csv(f"/home/user/polybtcmodel/reports/deep_realfill_grid_{ASSET}.csv", index=False)
print(f"{ASSET}: {n} windows; signals at {sorted(SIG)}; stake ${STAKES[0]} for the grid; sell {TP} else settle")
pool = E[(E.fills == "through") & (~E.flip) & (E.bid == 0.50) & (E.cancel == 0)].assign(w=lambda x: x.signals * x.fill_rate)
g = pool.groupby(["d", "tier"]).apply(lambda x: pd.Series({"fill": x.w.sum() / x.signals.sum(), "edge_c": np.nansum(x.edge_c * x.w) / x.w.sum(), "usd_day": np.average(x.usd_day, weights=[nd[p_] for p_ in x.period])}))
print("\n== bid 0.50 resting from the decision until the open, 'through' fills, pooled Jun-Nov: by decision time and tier =="); print(g.unstack("d").reindex(TIERS).round(2).to_string())
t = E[(E.period == "tune") & (E.fills == "through")]; CH = {}
for d in SIG:
    for flip in ((False, True) if d < -45 else (False,)):
        ch = {}
        for tn in TIERS:
            gg = t[(t.d == d) & (t.tier == tn) & (t.flip == flip)].sort_values("usd_day", ascending=False).iloc[0]
            if gg.usd_day > 0: ch[tn] = (int(np.argmin(np.abs(BIDS - gg.bid))), int(gg.kend))
        CH[(d, flip)] = ch
        print(f"chosen on tune, decision T{d}{' +flip-cancel' if flip else ''}:", {k_: (float(BIDS[b]), int(EDGES[ke + 1]) if ke < NB - 1 else 0) for k_, (b, ke) in ch.items()})
def portfolio(d, flip, choice, k, label, stake):
    tot = np.zeros(n); pn_all = np.full(n, np.nan)
    for tn, (b, ke) in choice.items():
        pn, cap = run(d, b, ke, k, flip); m = (TIER[d] == TIERS.index(tn)) & np.isfinite(pn); pn_all[m] = pn[m]; tot[m] = pn[m] * np.minimum(stake, cap[m]) / BIDS[b]
    out = []
    for p_ in PER:
        m = per == p_; f = m & np.isfinite(pn_all); dd = pd.Series(tot[m]).groupby(date[m]).sum().reindex(pd.date_range(*PER[p_]), fill_value=0.0); eq = dd.cumsum()
        out.append(dict(rule=label, stake=stake, period=p_, fills_day=f.sum() / nd[p_], win=(pn_all[f] > 0).mean() if f.sum() else np.nan, edge_c=pn_all[f].mean() * 100 if f.sum() else np.nan, usd_day=dd.mean(), maxdd=(eq.cummax() - eq).max(), worst_day=dd.min()))
    return out
R = []
for (d, flip), ch in CH.items():
    if not ch: continue
    for st in STAKES:
        R += portfolio(d, flip, ch, 1, f"NEW @T{d}{' flip' if flip else ''}: tune-chosen tiers [through]", st)
        top = {k_: v for k_, v in ch.items() if k_ in ("top 5%", "5-15%")}
        if top: R += portfolio(d, flip, top, 1, f"NEW @T{d}{' flip' if flip else ''}: top 15% only [through]", st)
    R += portfolio(d, flip, ch, 0, f"NEW @T{d}{' flip' if flip else ''}: tune-chosen tiers [exact]", STAKES[0])
P = pd.DataFrame(R); P.to_csv(f"/home/user/polybtcmodel/reports/deep_realfill_portfolio_{ASSET}.csv", index=False)
for st in STAKES:
    print(f"\n==== stake ${st} ====")
    for mname in ("fills_day", "win", "edge_c", "usd_day", "maxdd"):
        pp = P[P.stake == st]; print(f"-- {mname} --"); print(pp.pivot(index="rule", columns="period", values=mname).reindex(pp.rule.unique())[list(PER)].round(3 if mname == "win" else 1).to_string())
        if st != STAKES[0] and mname == "fills_day": continue
