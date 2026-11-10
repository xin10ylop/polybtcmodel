"""One more search for a higher win rate on the audited setup (T-120, top 30 %, bid 0.50 to T-15, T-45 confirmation, hold to settlement,
conservative fills: through 0.50 or $50 traded, post-only refusals skipped).  Every variant is reported on the three periods; a lever counts
only if it helps in all three.   python research/deep/policy_winrate.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
import policy_lab as L
from policy_lab import run, STAKE, date, Tix, SC, THR, mon, MONTHS
n = len(Tix); th = np.full(n, np.inf)
for mo in MONTHS:
    te = mon == mo; th[te] = np.quantile(np.abs(SC[-120][L.before(mo) & np.isfinite(SC[-120])]), 0.95)
THR[-120, 0.95] = th
BASE = dict(dec=-120, tier=0.70, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=50.0)
PER = {"Jun-Aug19": date < "2026-08-20", "tune": (date >= "2026-08-20") & (date <= "2026-09-30"), "TEST": date >= "2026-10-01"}
def row(nm, r, mask=None):
    m = np.ones(len(r["v"]), bool) if mask is None else mask; v = r["v"][m]; e = r["e"][m]; i = r["i"][m]
    if len(v) < 30: print(f"{nm:66s} too few trades ({len(v)})"); return
    w = " ".join(f"{(v[PER[k][i]] > 0).mean():.1%}" for k in PER); c = " ".join(f"{v[PER[k][i]].mean()*100:+.2f}" for k in PER)
    print(f"{nm:66s} {len(v)/160:>5.1f}/day  win {(v > 0).mean():.1%}  {v.mean()*100:+.2f} c  ${(v*STAKE/e).sum():>7,.0f} | win by period {w} | c/sh {c}", flush=True)
print(f"{'variant':66s} trades     win     c/share  total@$100 | Jun-Aug19 / tune / TEST")
b = run(**BASE); row("BASELINE (the audited setup)", b)
print("-- stronger signals only")
row("top 15 % of signals", run(**{**BASE, "tier": 0.85})); row("top 5 % of signals", run(**{**BASE, "tier": 0.95}))
for lo, hi in ((1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 99)): row(f"   baseline trades with |score| {lo:.1f}-{hi:.1f} x the threshold", b, (b["rel_score"] >= lo) & (b["rel_score"] < hi))
print("-- stricter confirmation at T-43 (keep the bid only if the T-45 model agrees AND is in its own top tier)")
for q, nm in ((0.50, "top 50 %"), (0.70, "top 30 %"), (0.85, "top 15 %")): row(f"T-45 model must agree within its {nm}", run(**{**BASE, "confirm": dict(cancel=True, need_strong=True, q=q)}))
print("-- how long the bid rests")
for tc in (-90, -60, -45, -30): row(f"cancel at T{tc}", run(**{**BASE, "stages": ((-118, tc, 0.50),)}))
for lo, hi in ((-118, -100), (-100, -75), (-75, -45), (-45, -15)): row(f"   baseline trades filled between T{lo} and T{hi}", b, (b["ft"] >= lo) & (b["ft"] < hi))
print("-- bid price")
for bid in (0.48, 0.49, 0.51): row(f"bid {bid:.2f} instead of 0.50", run(**{**BASE, "stages": ((-118, -15, bid),)}))
print("-- what the market shows when the bid goes in (last price a buyer of our side paid)")
a0 = b["a0"]
for lo, hi, nm in ((0.0, 0.505, "our side offered at 0.50 or less"), (0.505, 0.515, "0.51"), (0.515, 0.525, "0.52"), (0.525, 0.545, "0.53-0.54"), (0.545, 1.0, "0.55 or more")): row(f"   baseline trades, offer {nm}", b, (a0 > lo) & (a0 <= hi))
row("   baseline trades, no recent buyer of our side", b, ~np.isfinite(a0))
print("-- time")
hr = (Tix[b["i"]] % 86400) // 3600; dow = pd.to_datetime(Tix[b["i"]], unit="s").dayofweek.values
for lo, hi in ((0, 6), (6, 12), (12, 16), (16, 20), (20, 24)): row(f"   baseline trades, {lo:02d}-{hi:02d} UTC", b, (hr >= lo) & (hr < hi))
row("   baseline trades, weekdays", b, dow < 5); row("   baseline trades, weekend", b, dow >= 5)
print("-- side")
side_up = SC[-120][b["i"]] > 0; row("   baseline trades, bought UP", b, side_up); row("   baseline trades, bought DOWN", b, ~side_up)
