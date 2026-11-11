"""ALLOW_TAKER audit, second pass.  The book state when the bid goes in (T-118) is read from the nearest print: the last one before, else the
first one within 30 s after (a market buy of our side at p -> offer p; a market sell at p -> bid p, offer p + 0.01).  Signals whose side is
offered at <= 0.50 are REFUSED live (post-only); with ALLOW_TAKER=1 they are bought at that offer + fee.  The others rest (queue-aware fills).
  python research/deep/policy_taker_audit2.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
import policy_lab as L
from policy_lab import STAKE, date, Tix, SC, THR, ok, MONTHS, P, side_view, upwin
rng = np.random.default_rng(5); alld = pd.DatetimeIndex(sorted(set(date))); ND = len(alld); T_IN = -118
sc = SC[-120]; a45 = SC[-45]; sel = np.flatnonzero(ok & np.isfinite(sc) & (np.abs(sc) >= THR[-120, 0.70])); rows = []
for i in sel:
    w = Tix[i]
    if w not in P: continue
    s = 0 if sc[i] > 0 else 1; rel, ps, lift, usd, mxp = side_view(w, s); win = bool(upwin[i] if s == 0 else not upwin[i])
    b = (rel < T_IN); f = (rel >= T_IN) & (rel < T_IN + 30)
    if b.any(): j = np.flatnonzero(b)[-1]; ask = float(ps[j]) if lift[j] else float(ps[j]) + 0.01; src = 0; age = T_IN - rel[j]
    elif f.any(): j = np.flatnonzero(f)[0]; ask = float(ps[j]) if lift[j] else float(ps[j]) + 0.01; src = 1; age = rel[j] - T_IN
    else: ask = np.nan; src = 2; age = np.nan
    agree = bool(np.isfinite(a45[i]) and (a45[i] > 0) == (sc[i] > 0))
    r = L.simulate(w, s, [(T_IN, -15, 0.50)], None, dict(cancel=True, agree=agree), cross="rest", queue=50.0)      # what a resting bid would do
    rows.append(dict(i=i, win=win, ask=ask, src=src, age=age, rest_fill=r is not None, rest_vol=(L.VA[0] if r is not None else 0.0)))
D = pd.DataFrame(rows); t = pd.to_datetime(Tix[D.i.values], unit="s"); D["good"] = (t.dayofweek.values >= 5) | (t.hour.values >= 16); D["date"] = date[D.i.values]; D["mon"] = pd.to_datetime(D.date).dt.strftime("%b")
D["cross"] = D.ask <= 0.50 + 1e-6; G = D[D.good].copy()
print(f"trading hours: {len(G)/ND:.1f} signals/day. Book state known for {(G.src < 2).mean():.0%} (from a print before placement {(G.src == 0).mean():.0%}, median age {G.age[G.src == 0].median():.0f} s; from the first print after {(G.src == 1).mean():.0%})")
K = G[G.src < 2]; print(f"  our side offered at <= 0.50 (refused live): {K.cross.mean():.0%} of the signals with a known book   [live on the evening of Nov 11: 48 %; night of Nov 11, all hours: 34 %]")
print(f"  offer distribution on our side: " + ", ".join(f"{p:.2f}: {(K.ask.round(2) == p).mean():.0%}" for p in (0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54)) + f", <0.47: {(K.ask < 0.465).mean():.0%}, >0.54: {(K.ask > 0.545).mean():.0%}")
print(f"  model right at settlement: refused signals {K.win[K.cross].mean():.1%} (n {K.cross.sum()}), others {K.win[~K.cross].mean():.1%}; unknown book {G.win[G.src == 2].mean():.1%} (n {(G.src == 2).sum()})")
def pnl_take(g, worst):      # buy the offer (<= 0.50) + fee, hold to settlement, $100 stake
    e = np.where(worst, 0.50, np.minimum(g.ask.values, 0.50)); e = np.maximum(e, 0.02); fee = 0.07 * e * (1 - e); v = np.where(g.win.values, 1 - e, -e) - fee; return v * STAKE / e, v
def boot(pn, dts):
    d = pd.Series(pn).groupby(dts).sum().reindex(alld, fill_value=0.0); bs = [d.values[rng.integers(0, ND, ND)].sum() for _ in range(2000)]; return np.percentile(bs, [2.5, 97.5]), d
X = G[G.cross]; R = G[~G.cross & G.rest_fill]; pr = np.where(R.win.values, 100.0, -100.0)
PERS = {"Jun-Aug19": lambda d: d < pd.Timestamp("2026-08-20"), "Aug20-Sep30": lambda d: (d >= pd.Timestamp("2026-08-20")) & (d <= pd.Timestamp("2026-09-30")), "Oct-Nov7": lambda d: d >= pd.Timestamp("2026-10-01")}
print(f"\n$100 a trade, filled in full, hold to settlement")
for nm, pn, dts, wn in (("resting fills (what ALLOW_TAKER=0 trades)", pr, R.date.values, R.win.values), ("refused bids taken at the offer + fee", pnl_take(X, False)[0], X.date.values, X.win.values), ("refused bids taken, worst case 0.50 + fee", pnl_take(X, True)[0], X.date.values, X.win.values)):
    ci, d = boot(pn, dts); dd_ = pd.to_datetime(dts)
    print(f"  {nm:44s} {len(pn)/ND:>5.1f}/day  win {wn.mean():.1%}  {pn.mean():+.2f} $/trade  total ${pn.sum():>7,.0f} [{ci[0]:>7,.0f},{ci[1]:>7,.0f}] | by period " + " / ".join(f"{pn[fn(dd_)].sum():>7,.0f}" for fn in PERS.values()))
print(f"  share of signals that trade: ALLOW_TAKER=0 {len(R)/len(G):.0%} (live Nov 11 evening 29 %, night 31 %);  ALLOW_TAKER=1 {(len(R)+len(X))/len(G):.0%}")
print("\nmonth     | resting fills: trades / win / P&L | refused bids taken: trades / win / P&L (offer + fee) | same, worst case 0.50 | TOTAL with ALLOW_TAKER=1 (worst case)")
tw, _ = pnl_take(X, True); tb, _ = pnl_take(X, False)
for mo in MONTHS + ["total"]:
    mr = np.ones(len(R), bool) if mo == "total" else (R.mon.values == mo); mx = np.ones(len(X), bool) if mo == "total" else (X.mon.values == mo)
    print(f"{('Nov 1-7' if mo=='Nov' else mo):9s} | {mr.sum():>5d} / {R.win.values[mr].mean():.1%} / ${pr[mr].sum():>7,.0f}       | {mx.sum():>5d} / {X.win.values[mx].mean():.1%} / ${tb[mx].sum():>7,.0f}                     | ${tw[mx].sum():>7,.0f}             | ${pr[mr].sum() + tw[mx].sum():>7,.0f}")
print("\nrefused signals by how cheap our side is offered: " + " | ".join(f"{lo:.2f}-{hi:.2f}: n {((X.ask > lo - .005) & (X.ask <= hi + .005)).sum()}, win {X.win[(X.ask > lo - .005) & (X.ask <= hi + .005)].mean():.1%}" for lo, hi in ((0.50, 0.50), (0.49, 0.49), (0.47, 0.48), (0.40, 0.46), (0.0, 0.39))))
wk = pd.to_datetime(X.date.values); ci, dx = boot(tw, X.date.values); print(f"taker trades alone (worst-case price): losing days {(dx < 0).sum()}, winning {(dx > 0).sum()}; months positive {sum(tw[X.mon.values == m].sum() > 0 for m in MONTHS)}/6")
# ---- alternatives for the refused signals that stay fee-free
print("\nALTERNATIVES for the refused signals (no fee): P&L of those signals only, $100 a trade filled in full")
alt = {"A. skip them (ALLOW_TAKER=0, now)": [], "B. take the offer + fee (ALLOW_TAKER=1), worst-case price": [], "C. rest one tick lower at 0.49 until T-15": [], "D. keep re-sending the 0.50 bid; it rests once our side is offered above 0.50": []}
for _, x in X.iterrows():
    i = int(x.i); w = Tix[i]; s = 0 if sc[i] > 0 else 1; rel, ps, lift, usd, mxp = side_view(w, s); agree = bool(np.isfinite(a45[i]) and (a45[i] > 0) == (sc[i] > 0)); s45 = dict(cancel=True, agree=agree); dte = x.date; win = x.win
    e = 0.50; alt["B. take the offer + fee (ALLOW_TAKER=1), worst-case price"].append((dte, ((1 - e if win else -e) - 0.07 * e * (1 - e)) * STAKE / e, win))
    r = L.simulate(w, s, [(T_IN, -15, 0.49)], None, s45, cross="rest", queue=50.0)
    if r is not None: alt["C. rest one tick lower at 0.49 until T-15"].append((dte, (1 - 0.49 if win else -0.49) * STAKE / 0.49, win))
    # D: first moment after placement when the nearest print shows our side offered above 0.50
    t2 = np.nan
    for j in np.flatnonzero((rel >= T_IN) & (rel < -20)):
        a = float(ps[j]) if lift[j] else float(ps[j]) + 0.01
        if a > 0.50 + 1e-6: t2 = float(rel[j]) + 1.0; break
    if np.isfinite(t2):
        r = L.simulate(w, s, [(t2, -15, 0.50)], None, s45, cross="rest", queue=50.0)
        if r is not None: alt["D. keep re-sending the 0.50 bid; it rests once our side is offered above 0.50"].append((dte, (0.5 if win else -0.5) * STAKE / 0.5, win))
for nm, lst in alt.items():
    if not lst: print(f"  {nm:80s} 0 trades, $0"); continue
    dts = np.array([q[0] for q in lst]); pn = np.array([q[1] for q in lst]); wn = np.array([q[2] for q in lst]); ci, d = boot(pn, dts); dd_ = pd.to_datetime(dts); mo_ = dd_.strftime("%b")
    print(f"  {nm:80s} {len(pn)/ND:>4.1f}/day  win {wn.mean():.1%}  total ${pn.sum():>7,.0f} [{ci[0]:>7,.0f},{ci[1]:>7,.0f}] | periods " + " / ".join(f"{pn[fn(dd_)].sum():>6,.0f}" for fn in PERS.values()) + " | months " + " ".join(f"{pn[mo_ == m].sum():>6,.0f}" for m in MONTHS))
