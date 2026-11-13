"""Two ways of adapting, measured against the live rule on the same windows (trading hours, queue-aware fills, Jun 1 - Nov 7):
 S) stake by signal strength at order time;   R) after T-45, when the T-45 model agrees, raise the unfilled bid from 0.50 to 0.51 (maker, no fee).
Per-window profit of each policy, difference to the live rule with a day-block range, by period, and risk (daily Sharpe, max drawdown, worst day).
  python research/deep/adaptive_policy.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from policy_lab import run, SC, THR, Tix, date
N = len(Tix); good = np.asarray((date.dayofweek >= 5) | (pd.to_datetime(Tix, unit="s").hour >= 16)); day = date.values.astype("datetime64[D]"); per = np.where(date < "2026-08-20", 0, np.where(date < "2026-10-01", 1, 2)); rng = np.random.default_rng(21); days = np.unique(day); ND = len(days)
def table(st):
    r = run(dec=-120, tier=0.70, stages=st, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=50.0); m = good[r["i"]]; v = np.zeros(N); e = np.full(N, np.nan); va = np.zeros(N); q = np.zeros(N); i = r["i"][m]; v[i] = r["v"][m]; e[i] = r["e"][m]; va[i] = r["vol_after"][m]; q[i] = r["rel_score"][m]; return v, e, va, q
A = table(((-118, -15, 0.50),)); B = table(((-118, -43, 0.50), (-43, -15, 0.51))); s45 = SC[-45]; s120 = SC[-120]; strong = np.isfinite(s45) & ((s45 > 0) == (s120 > 0)) & (np.abs(s45) >= THR[-45, 0.70])
def pnl(tab, base, cap, mult=None, liq=True):
    v, e, va, q = tab; k = np.isfinite(e); st = np.zeros(N); f = np.ones(N) if mult is None else mult(q); st[k] = np.minimum(f[k] * base, cap); st[k] = np.minimum(st[k], va[k]) if liq else st[k]; p = np.zeros(N); p[k] = v[k] * st[k] / e[k] + st[k] / e[k] * 0.07 * e[k] * (1 - e[k]) * 0.2; return p
def mix(pa, pb, cond): return np.where(cond, pb, pa)
def stats(lbl, p, ref):
    d = pd.Series(p).groupby(day).sum().reindex(days, fill_value=0.0).values; dr = pd.Series(ref).groupby(day).sum().reindex(days, fill_value=0.0).values; o = np.argsort(Tix); c = np.cumsum(p[o]); dd = float((np.maximum.accumulate(np.maximum(c, 0)) - c).max()); x = d - dr; b = x[rng.integers(0, ND, (4000, ND))].sum(1)
    return {"policy": lbl, "trades": int((p != 0).sum()), "total $": round(p.sum()), "vs live $": round(x.sum()), "95% lo": round(np.quantile(b, .025)), "95% hi": round(np.quantile(b, .975)), "P(better)": round((b > 0).mean(), 3), "Jun1-Aug19": round((p - ref)[per == 0].sum()), "Aug20-Sep30": round((p - ref)[per == 1].sum()), "Oct1-Nov7": round((p - ref)[per == 2].sum()),
            "daily Sharpe": round(d.mean() / d.std() * np.sqrt(365), 2), "max DD $": round(dd), "worst day $": round(d.min()), "losing days": int((d < 0).sum())}
prop = lambda q: np.clip(q, 1.0, 3.0); two = lambda q: np.where(q >= 1.7, 2.0, 1.0); tri = lambda q: np.where(q >= 2.2, 3.0, np.where(q >= 1.7, 2.0, 1.0)); hl = lambda q: np.where(q >= 1.7, 2.0, np.where(q >= 1.35, 1.0, 0.5)); top = lambda q: np.where(q >= 2.2, 2.0, 1.0)
pd.set_option("display.width", 280)
for base, cap, liq, ttl in ((10, 1e9, False, "BASE STAKE $10 (orders fill completely at this size)"), (100, 200, True, "BASE STAKE $100, never more than $200 a trade, limited to the dollars sold into us")):
    L = pnl(A, base, cap, None, liq); rows = [stats("live rule (flat stake, 0.50)", L, L)]
    rows.append(stats("R: raise to 0.51 after T-45 agreement", pnl(B, base, cap, None, liq), L)); rows.append(stats("R': raise to 0.51 only when T-45 agrees strongly", mix(L, pnl(B, base, cap, None, liq), strong), L))
    for nm, f in (("S: 2x on the strongest (2.2x threshold +)", top), ("S: 2x from 1.7x threshold", two), ("S: 2x from 1.7x, 3x from 2.2x", tri), ("S: half below 1.35x, 2x from 1.7x", hl), ("S: proportional to strength, 1x-3x", prop)): rows.append(stats(nm, pnl(A, base, cap, f, liq), L))
    rows.append(stats("S + R: proportional stake and raise to 0.51", pnl(B, base, cap, prop, liq), L)); rows.append(stats("S + R: half/1x/2x stake and raise to 0.51", pnl(B, base, cap, hl, liq), L))
    if liq: F2 = pnl(A, 200, 200, None, True); rows.append(stats("for reference: flat $200", F2, L))
    print("\n" + ttl); print(pd.DataFrame(rows).to_string(index=False))
