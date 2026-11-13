"""Adapting when the bid does not fill: about two thirds of the live rule's signals never fill, and they win at least as often as the filled ones.
For every top-30 % signal in trading hours whose 0.50 bid was NOT filled (and that the T-45 model did not contradict), buy our side at the market at time t
if it is on offer at or below a cap.  Entry = what the next real buyer of our side paid, first print stamped 5-20 s after t (a print's stamp is a few seconds
after the trade), plus the taker fee.  Hold to settlement.  $100 a trade; 'real size' limits the stake to the dollars that buyers took at or below the cap then.
Tune period Aug 20 - Sep 30; judged on Oct 1 - Nov 7 and Jun 1 - Aug 19.     python research/deep/late_take.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from policy_lab import simulate, side_view, SC, THR, ok, P, Tix, date, upwin
sc = SC[-120]; good = np.asarray((date.dayofweek >= 5) | (pd.to_datetime(Tix, unit="s").hour >= 16)); idx = [i for i in np.flatnonzero(ok & np.isfinite(sc) & (np.abs(sc) >= THR[-120, 0.70]) & good) if Tix[i] in P]
day = date.values.astype("datetime64[D]"); per = np.where(date < "2026-08-20", "Jun1-Aug19", np.where(date < "2026-10-01", "Aug20-Sep30", "Oct1-Nov7")); rng = np.random.default_rng(11); ND = len(np.unique(day)); S = []
for i in idx:
    s = 0 if sc[i] > 0 else 1; a = SC[-45][i]; agree = bool(np.isfinite(a) and (a > 0) == (sc[i] > 0)); s45 = dict(cancel=True, agree=agree)
    filled = simulate(Tix[i], s, [(-118, -15, 0.50)], None, s45, "skip", 5, 50.0) is not None; refused = (not filled) and simulate(Tix[i], s, [(-118, -15, 0.50)], None, s45, "take", 5, 50.0) is not None and simulate(Tix[i], s, [(-118, -15, 0.50)], None, s45, "rest", 5, 50.0) is None
    S.append((i, s, agree, filled, refused, bool(upwin[i]) if s == 0 else not bool(upwin[i])))
S = pd.DataFrame(S, columns=["i", "s", "agree", "filled", "refused", "win"]); print(f"{len(S)} signals in trading hours ({len(S)/ND:.1f}/day): filled {S.filled.sum()} (win {100*S.win[S.filled].mean():.1f} %), not filled {(~S.filled).sum()} (win {100*S.win[~S.filled].mean():.1f} %); not filled and the T-45 model agrees: {(~S.filled & S.agree).sum()} (win {100*S.win[~S.filled & S.agree].mean():.1f} %); not filled, T-45 disagrees: {(~S.filled & ~S.agree).sum()} (win {100*S.win[~S.filled & ~S.agree].mean():.1f} %)")
U = S[~S.filled & S.agree].reset_index(drop=True)
def take(t, cap, sub=U):
    o = []
    for i, s, win in zip(sub.i.values, sub.s.values, sub.win.values):
        rel, ps, lift, usd, _ = side_view(Tix[i], s); k = np.flatnonzero(lift & (rel >= t + 5) & (rel < t + 20))
        if not len(k) or ps[k[0]] > cap + 1e-6: continue
        p = float(ps[k[0]]); kk = k[ps[k] <= cap + 1e-6]; o.append((i, win, p + 0.07 * p * (1 - p), float(usd[kk].sum())))
    return pd.DataFrame(o, columns=["i", "win", "cost", "usd"])
def row(lbl, o):
    if len(o) < 40: return {"rule": lbl, "trades": len(o)}
    e = (o.win - o.cost).values; full = 100 / o.cost.values * e; real = np.minimum(100, o.usd.values) / o.cost.values * e; g = pd.Series(real).groupby(day[o.i.values]).sum().values; b = g[rng.integers(0, len(g), (2000, len(g)))].sum(1); pp = per[o.i.values]
    return {"rule": lbl, "trades": len(o), "per day": round(len(o) / ND, 1), "win %": round(100 * o.win.mean(), 1), "paid ¢ (fee in)": round(100 * o.cost.mean(), 1), "¢/share": round(100 * e.mean(), 2), "full $": round(full.sum()), "real $": round(real.sum()), "95% lo": round(np.quantile(b, .025)), "95% hi": round(np.quantile(b, .975)),
            "Jun1-Aug19 ¢": round(100 * e[pp == "Jun1-Aug19"].mean(), 2), "Aug20-Sep30 ¢": round(100 * e[pp == "Aug20-Sep30"].mean(), 2), "Oct1-Nov7 ¢": round(100 * e[pp == "Oct1-Nov7"].mean(), 2), "median $ on offer": round(o.usd.median())}
pd.set_option("display.width", 260); print("\nBuy the unfilled signals at the market (T-45 model agrees).  Rows: when, and the highest price accepted")
print(pd.DataFrame([row(f"at T{t:+d}s, up to {cap:.2f}", take(t, cap)) for t in (-40, -30, -15, -5, 5, 15, 30) for cap in (0.50, 0.52, 0.55, 0.60)]).to_string(index=False))
print("\nfor comparison, the same market buy on signals the T-45 model CONTRADICTS (should be worse):"); print(pd.DataFrame([row(f"at T{t:+d}s, up to {cap:.2f}", take(t, cap, S[~S.filled & ~S.agree].reset_index(drop=True))) for t in (-15, 5) for cap in (0.52, 0.55)]).to_string(index=False))
