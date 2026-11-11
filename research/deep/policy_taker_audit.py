"""Audit of ALLOW_TAKER=1 for the live BTC setup (T-120, top 30 %, T-45 confirmation, weekends + weekday 16-24 UTC, settle, queue-aware fills).
Crossing at placement is judged from the last print before the bid goes in (book_ask), which reproduces the refusal rate measured live
(34 % and 48 % of signals on Nov 11).  Compared: refused bids skipped (ALLOW_TAKER=0) vs taken with the fee (ALLOW_TAKER=1), month by month.
  python research/deep/policy_taker_audit.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from policy_lab import run, STAKE, date, Tix, SC, THR, ok, MONTHS, P, side_view, book_ask
rng = np.random.default_rng(21); alld = pd.DatetimeIndex(sorted(set(date))); ND = len(alld)
def good(i): t = pd.to_datetime(Tix[i], unit="s"); return (t.dayofweek.values >= 5) | (t.hour.values >= 16)
PER = {"Jun-Aug19": date < "2026-08-20", "Aug20-Sep30": (date >= "2026-08-20") & (date <= "2026-09-30"), "Oct-Nov7": date >= "2026-10-01"}
sc = SC[-120]; sel = np.flatnonzero(ok & np.isfinite(sc) & (np.abs(sc) >= THR[-120, 0.70])); g = good(sel); cr = []
for i in sel:
    w = Tix[i]
    if w not in P: cr.append(np.nan); continue
    rel, ps, lift, usd, _ = side_view(w, 0 if sc[i] > 0 else 1); cr.append(book_ask(rel, ps, lift, -118))
cr = np.array(cr); crossing = cr <= 0.50 + 1e-6
print(f"signals: {len(sel)/ND:.1f}/day ({g.sum()/ND:.1f} in the trading hours). Our side already offered at <= 0.50 when the bid goes in: {np.nanmean(crossing[g]):.0%} in the trading hours, {np.nanmean(crossing[~g]):.0%} on weekday 00-16 UTC   [live on Nov 11: 48 % and 34 %]")
print(f"   offer when it crosses: mean {np.nanmean(cr[crossing & g]):.3f}; share at 0.50 exactly {np.mean(np.round(cr[crossing & g], 2) == 0.50):.0%}, at 0.49 {np.mean(np.round(cr[crossing & g], 2) == 0.49):.0%}, lower {np.mean(cr[crossing & g] < 0.485):.0%}")
B = dict(dec=-120, tier=0.70, confirm=dict(cancel=True), tp=None, queue=50.0, ask_rule="last")
def stats(nm, r, m=None, full_stake=True):
    k = good(r["i"]) if m is None else m; v = r["v"][k]; e = r["e"][k]; i = r["i"][k]; pnl = v * STAKE / e
    d = pd.Series(pnl).groupby(date[i]).sum().reindex(alld, fill_value=0.0); bs = [d.values[rng.integers(0, ND, ND)].sum() for _ in range(2000)]
    c = np.concatenate([[0], np.cumsum(pnl[np.argsort(Tix[i])])]); dd = (np.maximum.accumulate(c) - c).max()
    print(f"{nm:58s} {len(v)/ND:>5.1f}/d  win {(v>0).mean():.1%}  {v.mean()*100:+.2f}c  ${pnl.sum():>7,.0f} [{np.percentile(bs,2.5):>7,.0f},{np.percentile(bs,97.5):>7,.0f}]  maxDD ${dd:>5,.0f} | " + " / ".join(f"{pnl[PER[q][i]].sum():>7,.0f}" for q in PER), flush=True)
    return pnl, i, v
print(f"\n{'$100 a trade if filled in full':58s} trades  win    c/share  total [95% range]            drawdown | by period: Jun-Aug19 / Aug20-Sep30 / Oct-Nov7")
rs = run(cross="skip", **B); a = stats("ALLOW_TAKER=0: refused bids skipped (running now)", rs)
rt = run(cross="take", **B); b = stats("ALLOW_TAKER=1: refused bids taken at the offer + fee", rt)
rw = run(cross="take", worst_px=True, **B); c = stats("ALLOW_TAKER=1, worst case: every taken share costs 0.50", rw)
k = good(rt["i"]); tk = rt["tk"]; print("\nthe taken trades on their own (ALLOW_TAKER=1):")
stats("   taker fills only", rt, k & tk); stats("   taker fills only, worst case price 0.50", rw, good(rw["i"]) & rw["tk"]); stats("   resting fills in the same run", rt, k & ~tk)
print("\nmonth     | ALLOW_TAKER=0: trades / win / P&L | ALLOW_TAKER=1: trades / win / P&L | taker fills alone: trades / win / P&L (worst-case price) | losing days 0 -> 1 | worst day 0 -> 1")
for mo in MONTHS + ["total"]:
    out = f"{('Nov 1-7' if mo=='Nov' else mo):9s} |"; days = {}
    for key, r, extra in (("s", rs, None), ("t", rt, None), ("w", rw, "tk")):
        kk = good(r["i"]) & (r["tk"] if extra else True); mm = kk & (np.ones(len(kk), bool) if mo == "total" else (pd.to_datetime(date[r["i"]]).strftime("%b") == mo)); pnl = r["v"][mm] * STAKE / r["e"][mm]
        out += f" {mm.sum():>5d} / {(r['v'][mm]>0).mean():.1%} / ${pnl.sum():>7,.0f}        |"
        md = np.ones(ND, bool) if mo == "total" else (alld.strftime("%b") == mo); days[key] = pd.Series(pnl).groupby(date[r["i"]][mm]).sum().reindex(alld[md], fill_value=0.0)
    print(out + f"            {(days['s']<0).sum():>2d} -> {(days['t']<0).sum():>2d}       | ${days['s'].min():>6,.0f} -> ${days['t'].min():>6,.0f}")
