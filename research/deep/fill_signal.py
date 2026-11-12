"""Is the amount sold into our bid a warning sign, and does selling before the open pay?  Live setup (T-120, top 30 %, T-45 confirmation, bid 0.50 until T-15,
queue-aware fills, trading hours), Jun 1 - Nov 7.  For every simulated fill: the dollars takers sold into 0.50 on our side AFTER our fill, until T-15.
  A) win rate by that amount                      B) rule 'once X dollars have been sold into us, sell before the open' (taker, fee included, real next sale price)
  C) the opposite rule 'if less than X was sold into us by T-15, sell before the open'.     Per share and at $100 a trade (200 shares).
  python research/deep/fill_signal.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from policy_lab import run, side_view, Tix, SC, date, PER
r = run(dec=-120, tier=0.70, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=50.0); I = r["i"]; hrs = np.asarray((date.dayofweek >= 5) | (pd.to_datetime(Tix, unit="s").hour >= 16))[I]
I = I[hrs]; win = (r["v"] > 0)[hrs]; ft = r["ft"][hrs]; day = date.values.astype("datetime64[D]")[I]; n = len(I); rng = np.random.default_rng(0)
XS = [50, 100, 200, 300, 500, 1000]; Va = np.zeros(n); tau = np.full((n, len(XS)), np.nan); ex = np.full((n, len(XS)), np.nan); exlate = np.full(n, np.nan)
def sale(rel, ps, lift, t):          # what the next seller of our side got after time t, before the open (fee taken off); the last pre-open price minus a cent if nobody sold
    k = np.flatnonzero(~lift & (rel > t) & (rel <= -1)); p = float(ps[k[0]]) if len(k) else (float(ps[rel <= -1][-1]) - 0.01 if (rel <= -1).any() else 0.49); return p - 0.07 * p * (1 - p)
for j, i in enumerate(I):
    s = 0 if SC[-120][i] > 0 else 1; rel, ps, lift, usd, _ = side_view(Tix[i], s); k = ~lift & (rel >= ft[j]) & (rel <= -15) & (ps <= 0.50 + 1e-6); c = np.cumsum(usd[k]); Va[j] = c[-1] if len(c) else 0.0
    for q, x in enumerate(XS):
        h = np.flatnonzero(c >= x)
        if len(h): tau[j, q] = rel[k][h[0]]; ex[j, q] = sale(rel, ps, lift, tau[j, q] + 1)
    exlate[j] = sale(rel, ps, lift, -14)
hold = np.where(win, 0.5, -0.5)
def ci(d):
    g = pd.Series(d).groupby(day).sum().values; b = g[rng.integers(0, len(g), (2000, len(g)))].sum(1); return round(np.quantile(b, 0.025)), round(np.quantile(b, 0.975))
print(f"{n} fills in trading hours, win rate {100*win.mean():.1f} %, hold to settlement = {100*hold.mean():+.2f}¢ a share, ${200*hold.sum():,.0f} at $100 a trade (every fill complete)\n")
print("A) by the dollars sold into our bid after our fill (until T-15)"); B = [0, 25, 50, 100, 200, 500, 1000, 1e9]; rows = []
for a, b in zip(B[:-1], B[1:]):
    m = (Va >= a) & (Va < b); rows.append({"sold into us": f"${a:.0f} - {b:.0f}" if b < 1e9 else f"over ${a:.0f}", "trades": int(m.sum()), "win %": round(100 * win[m].mean(), 1), "¢/share if held": round(100 * hold[m].mean(), 2), "Jun1-Aug19 win %": round(100 * win[m & PER["Jun1-Aug19"][I]].mean(), 1), "Aug20-Sep30": round(100 * win[m & PER["tune"][I]].mean(), 1), "Oct1-Nov7": round(100 * win[m & PER["TEST"][I]].mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))
def rule(lbl, m, exitpx):
    d = np.where(m, exitpx - 0.50, hold) - hold; lo, hi = ci(200 * d); t = PER["TEST"][I]
    return {"rule": lbl, "trades sold": int(m.sum()), "their win % if held": round(100 * win[m].mean(), 1), "sale price ¢ (after fee)": round(100 * np.nanmean(exitpx[m]), 1), "change ¢/share on those": round(100 * d[m].mean(), 2), "change in total $": round(200 * d.sum()), "95% lo": lo, "95% hi": hi, "Jun1-Sep30 $": round(200 * d[~t].sum()), "Oct1-Nov7 $": round(200 * d[t].sum())}
print("\nB) once X dollars have been sold into our bid, sell our shares before the open (change versus simply holding)")
print(pd.DataFrame([rule(f"X = ${x}", np.isfinite(tau[:, q]), ex[:, q]) for q, x in enumerate(XS)]).to_string(index=False))
print("\nC) the opposite: if LESS than X was sold into us by T-15, sell before the open")
print(pd.DataFrame([rule(f"less than ${x}", Va < x, exlate) for x in (25, 50, 100, 200)]).to_string(index=False))
