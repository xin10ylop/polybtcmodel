"""When does the pre-open flow become informed?  From maker_sim2: a resting BUY at 0.50 / 0.51 filled in each bucket of the timeline.
 1. no signal: the maker's P&L per filled share (sell limit else settle) for BOTH sides of every window = minus the takers' edge.
 2. with the signal known at the start of the bucket (frozen close head at T-120 / -90 / -60 / -45, purpose-built model at T-45 and,
    when trained, at T-120 / -90 / -60): fills only on the signal's side; fill rate, edge, $ per day at the stake.
  ASSET=BTC python research/deep/flow_timing.py"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, PER, DEEP  # noqa: E402
ASSET = os.environ.get("ASSET", "BTC"); STAKE = 100 if ASSET == "BTC" else 25
Z = np.load(f"{DEEP}/maker2_{ASSET}.npz"); BIDS, EDGES, TP = Z["bids"], Z["edges"], float(Z["tp"]); NB = len(EDGES) - 1; BL = [f"[{EDGES[k]},{EDGES[k + 1] if k < NB - 1 else 0}{')' if k < NB - 1 else ']'}" for k in range(NB)]
W = load(ASSET, [f"m2_{o}_close_p" for o in (-120, -90, -60, -45)]); W = W[W.index.isin(Z["W"])]; pos = pd.Series(np.arange(len(Z["W"])), index=Z["W"]).reindex(W.index).values
fill, tph, usd, up = Z["fill"][pos], Z["tph"][pos], Z["usd"][pos], Z["up"][pos]; per = W.period.values; n = len(W); idx = np.arange(n)
def pnl_of(s, b, k, d):     # side array s (0 up / 1 down), bid index, bucket, definition -> pnl per share (NaN unfilled), $ capacity
    f = np.isfinite(fill[idx, s, b, k, d]); win = np.where(s == 0, up, ~up); return np.where(f, np.where(tph[idx, s, b, k, d], TP - BIDS[b], win.astype(float) - BIDS[b]), np.nan), usd[idx, s, b, k]
print(f"{ASSET}: {n} windows.  1. NO SIGNAL: maker P&L per filled share (cents), both sides pooled; fill = share of window-sides filled in the bucket")
rows = []
for b in (2, 3):
    for d, dn in enumerate(("exact", "through")):
        r = {"bid": BIDS[b], "fills": dn}; r2 = {"bid": BIDS[b], "fills": dn + " fill rate"}
        for k in range(NB):
            pp = np.concatenate([pnl_of(np.zeros(n, int), b, k, d)[0], pnl_of(np.ones(n, int), b, k, d)[0]]); r[BL[k]] = np.nanmean(pp) * 100; r2[BL[k]] = np.isfinite(pp).mean()
        rows += [r, r2]
print(pd.DataFrame(rows).round(2).to_string(index=False))
SIG = {}
lz = lambda q: q - 0.5
for o in (-120, -90, -60, -45): SIG[f"frozen head @T{o}"] = (o, lz(W[f"m2_{o}_close_p"].values))
for o, f in ((-45, f"{DEEP}/ourcase_{ASSET}_ph5book.parquet"), (-60, f"{DEEP}/ourcase_{ASSET}_d-60_ph3book.parquet"), (-90, f"{DEEP}/ourcase_{ASSET}_d-90_ph3book.parquet"), (-120, f"{DEEP}/ourcase_{ASSET}_d-120_ph3book.parquet")):
    if os.path.exists(f): SIG[f"NEW model @T{o}"] = (o, pd.read_parquet(f).set_index("T")["lgb_reg300"].reindex(W.index).values)
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; mon = W.month.values
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
def top(score, frac):
    sel = np.zeros(n, bool)
    for mo in MONTHS:
        te = (mon == mo) & np.isfinite(score); a = np.abs(score[before(mo) & np.isfinite(score)]); sel[te] = np.abs(score[te]) >= (np.quantile(a, 1 - frac) if frac < 1 else 0)
    return sel
print("\n2. WITH A SIGNAL: bid on the signal's side from the signal time until the open; fills in the first bucket that fills ('through' fills)")
rows = []
for nm, (o, sc) in SIG.items():
    k0 = int(np.flatnonzero(EDGES == o)[0]); s = np.where(sc > 0, 0, 1)
    for frac in (0.05, 0.15, 0.30, 1.0):
        sel = top(sc, frac)
        for b in (2, 3, 4):
            for kend, lab in ((k0, "first bucket only"), (NB - 1, "until the open")):
                pn = np.full(n, np.nan); cp = np.zeros(n)
                for k in range(k0, kend + 1):
                    p_, c_ = pnl_of(s, b, k, 1); m = np.isnan(pn) & np.isfinite(p_); pn[m] = p_[m]; cp[m] = c_[m]
                f = sel & np.isfinite(pn); r = dict(signal=nm, top=f"{int(frac * 100)}%", bid=BIDS[b], rest=lab, signals_day=sel.sum() / 160, fill=f.sum() / max(sel.sum(), 1), win=(pn[f] > 0).mean(), edge_c=pn[f].mean() * 100,
                                                    usd_day=(pn[f] * np.minimum(STAKE, cp[f]) / BIDS[b]).sum() / 160)
                for p_ in PER: mm = f & (per == p_); r[p_] = pn[mm].mean() * 100 if mm.sum() >= 20 else np.nan
                rows.append(r)
E = pd.DataFrame(rows); E.to_csv(f"/home/user/polybtcmodel/reports/deep_flow_timing_{ASSET}.csv", index=False)
print(E[E.rest == "until the open"].drop(columns="rest").round(2).to_string(index=False))
