"""Overfitting guard for the real-fill rules.  A small PRE-SPECIFIED family, nothing chosen per tier or per period:
   signal = purpose-built model at T-120 / -90 / -60 / -45;  trade set = top 15 % (or top 5 %) of |score| vs the months before;
   bid 0.50 or 0.51 on the signal's side from the decision until the open;  'through' fills;  sell limit else settle.
Reported: every period, and pooled over the 160 days with a day-block bootstrap (2,000 resamples) of $/day and of the edge per share.
Next to it: the frozen rule A under the same real fills, and both under the old 'always filled at 0.50' assumption.
  ASSET=BTC python research/deep/robust_eval.py"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 280)
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, PER, DEEP  # noqa: E402
ASSET = os.environ.get("ASSET", "BTC"); STAKE = float(os.environ.get("STAKE", "100" if ASSET == "BTC" else "25"))
Z = np.load(f"{DEEP}/maker2_{ASSET}.npz"); BIDS, EDGES, TP = Z["bids"], Z["edges"], float(Z["tp"]); NB = len(EDGES) - 1
W = load(ASSET, ["m2_-45_close_p", "bp_-60_z_s5"]); W = W[W.index.isin(Z["W"])]; pos = pd.Series(np.arange(len(Z["W"])), index=Z["W"]).reindex(W.index).values
fill, tph, usd, up = Z["fill"][pos], Z["tph"][pos], Z["usd"][pos], Z["up"][pos]; per = W.period.values; date = W.date.values; n = len(W); idx = np.arange(n)
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; mon = W.month.values; days = np.unique(date); dix = np.searchsorted(days, date); rng = np.random.default_rng(7)
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
def top(score, frac):
    sel = np.zeros(n, bool)
    for mo in MONTHS:
        te = (mon == mo) & np.isfinite(score); sel[te] = np.abs(score[te]) >= np.quantile(np.abs(score[before(mo) & np.isfinite(score)]), 1 - frac)
    return sel
def real(side, sel, d, b, k=1):
    k0 = int(np.flatnonzero(EDGES == d)[0]); win = np.where(side == 0, up, ~up); pn = np.full(n, np.nan); cap = np.zeros(n)
    for kk in range(k0, NB):
        f = np.isfinite(fill[idx, side, b, kk, k]); m = np.isnan(pn) & f; pn[m] = np.where(tph[idx, side, b, kk, k], TP - BIDS[b], win.astype(float) - BIDS[b])[m]; cap += usd[idx, side, b, kk]
    pn = np.where(sel, pn, np.nan); return pn, np.where(np.isfinite(pn), pn * np.minimum(STAKE, cap) / BIDS[b], 0.0)
S0 = sims_cached(ASSET, TP).reindex(W.index)
def assumed(side, sel):
    pn = np.where(sel, np.where(side == 0, S0.pnl_up.values, S0.pnl_dn.values), np.nan); return pn, np.where(np.isfinite(pn), pn * STAKE / 0.5, 0.0)
def summarise(label, sel, pn, usd_):
    f = np.isfinite(pn); r = dict(rule=label, signals_day=sel.sum() / len(days), fills_day=f.sum() / len(days), win=(pn[f] > 0).mean(), edge_c=pn[f].mean() * 100)
    dsum = np.bincount(dix, weights=usd_, minlength=len(days)); dpn = np.bincount(dix, weights=np.nan_to_num(pn), minlength=len(days)); dcnt = np.bincount(dix, weights=f.astype(float), minlength=len(days))
    bs = rng.integers(0, len(days), (2000, len(days))); r["usd_day"] = dsum.mean(); lo, hi = np.percentile(dsum[bs].mean(1), [2.5, 97.5]); r["usd_day_95ci"] = f"[{lo:.0f}, {hi:.0f}]"
    e = dpn[bs].sum(1) / np.maximum(dcnt[bs].sum(1), 1) * 100; r["edge_95ci"] = f"[{np.percentile(e, 2.5):.1f}, {np.percentile(e, 97.5):.1f}]"; r["P(edge<=0)"] = (e <= 0).mean()
    eq = np.cumsum(dsum); r["maxdd"] = (np.maximum.accumulate(eq) - eq).max(); r["worst_day"] = dsum.min(); r["neg_days"] = f"{int((dsum < 0).sum())}/{len(days)}"
    for p_ in PER:
        m = f & (per == p_); r[p_] = f"{pn[m].mean() * 100:+.1f}c ${usd_[per == p_].sum() / len(pd.date_range(*PER[p_])):.0f}" if m.sum() >= 15 else "-"
    return r
pm = W["m2_-45_close_p"].values; z5 = W["bp_-60_z_s5"].values; selA = np.isfinite(pm) & (np.abs(pm - 0.5) >= 0.05) & np.isfinite(z5) & (np.sign(z5) == np.sign(pm - 0.5)); sA = np.where(pm > 0.5, 0, 1)
ROWS = [summarise("FROZEN A | assumed fill at 0.50", selA, *assumed(sA, selA)), summarise("FROZEN A | real fills, bid 0.50 from T-45 to the open", selA, *real(sA, selA, -45, 2)),
        summarise("FROZEN A | real fills, bid 0.51 from T-45 to the open", selA, *real(sA, selA, -45, 3))]
SIG = {}
for d, f_ in ((-45, "ph5book"), (-60, "d-60_ph3book"), (-90, "d-90_ph3book"), (-120, "d-120_ph3book")):
    p_ = f"{DEEP}/ourcase_{ASSET}_{f_}.parquet"
    if os.path.exists(p_): SIG[d] = pd.read_parquet(p_).set_index("T")["lgb_reg300"].reindex(W.index).values
for d, sc in SIG.items():
    side = np.where(sc > 0, 0, 1)
    for frac in (0.15, 0.05):
        sel = top(sc, frac)
        if d == -45: ROWS.append(summarise(f"NEW @T{d} top {int(frac * 100)}% | assumed fill at 0.50", sel, *assumed(side, sel)))
        for b in (2, 3): ROWS.append(summarise(f"NEW @T{d} top {int(frac * 100)}% | real fills, bid {BIDS[b]:.2f} until the open", sel, *real(side, sel, d, b)))
if len(SIG) > 1:      # equal-weight consensus of the decision times available by T-90: sides must agree, rank by the mean of standardised scores
    ds = [d for d in SIG if d <= -90]
    if len(ds) == 2:
        z = np.mean([SIG[d] / np.nanstd(SIG[d]) for d in ds], axis=0); agree = np.sign(SIG[ds[0]]) == np.sign(SIG[ds[1]]); sel = top(z, 0.15) & agree; side = np.where(z > 0, 0, 1)
        for b in (2, 3): ROWS.append(summarise(f"NEW consensus T-120 & T-90, top 15% | real fills, bid {BIDS[b]:.2f} from T-90", sel, *real(side, sel, -90, b)))
R = pd.DataFrame(ROWS); R.to_csv(f"/home/user/polybtcmodel/reports/deep_robust_{ASSET}.csv", index=False)
print(f"{ASSET}: {len(days)} days, stake ${STAKE:.0f}, sell {TP} else settle; real fills = 'through', $ capped by the taker flow that traded against the bid\n")
print(R[["rule", "signals_day", "fills_day", "win", "edge_c", "edge_95ci", "P(edge<=0)", "usd_day", "usd_day_95ci", "maxdd", "worst_day", "neg_days"]].round(3).to_string(index=False))
print("\nby period (edge per filled share, $ per day):"); print(R[["rule"] + list(PER)].to_string(index=False))
