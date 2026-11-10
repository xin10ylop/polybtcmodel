"""Grid of execution policies for rule A (BTC) on the simulator of policy_lab.py.  python research/deep/policy_grid.py"""
import sys, pickle, numpy as np; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from policy_lab import run, summ, TABLE, DEEP
B = ((-118, -15, 0.50),); POL = {}
POL["B0  T-120, bid 0.50 to T-15 (baseline)"] = dict()
for t in (-15, -43):
    for cap in (0.51, 0.52, 0.53, 0.55, 0.60, 0.99): POL[f"M   B0 + market buy of what is left at T{t} if offer <= {cap:.2f}"] = dict(stages=((-118, t, 0.50),), take=(t, cap))
for t1 in (-90, -60, -45):
    for b1 in (0.51, 0.52): POL[f"R   bid 0.50 to T{t1}, then {b1:.2f} to T-15"] = dict(stages=((-118, t1, 0.50), (t1, -15, b1)))
POL["R   bid 0.50 -> 0.51 at T-80 -> 0.52 at T-45, to T-15"] = dict(stages=((-118, -80, 0.50), (-80, -45, 0.51), (-45, -15, 0.52)))
POL["R   bid 0.50 -> 0.51 at T-80 -> 0.52 at T-45 -> 0.53 at T-25, to T-15"] = dict(stages=((-118, -80, 0.50), (-80, -45, 0.51), (-45, -25, 0.52), (-25, -15, 0.53)))
for b in (0.51, 0.52): POL[f"H   bid {b:.2f} from the start to T-15"] = dict(stages=((-118, -15, b),))
POL["C   B0, cancel at T-43 when the T-45 model disagrees"] = dict(confirm=dict(cancel=True))
for q in (0.85, 0.70):
    for cap in (0.53, 0.55, 0.60, 0.99): POL[f"C   B0, cancel on disagreement; T-45 model agrees in its top {int(round((1-q)*100))}%: market buy <= {cap:.2f}"] = dict(confirm=dict(cancel=True, take_cap=cap, q=q))
for cap in (0.51, 0.52, 0.55, 0.99): POL[f"T   market buy at the T-120 decision if offer <= {cap:.2f}"] = dict(stages=(), take=(-118, cap))
for tier, nm in ((0.70, "top 30 %"), (0.50, "top 50 %")):
    POL[f"S   B0 on the {nm} of signals"] = dict(tier=tier)
    POL[f"S   B0 on the {nm}, cancel on T-45 disagreement"] = dict(tier=tier, confirm=dict(cancel=True))
for tp, nm in ((0.90, "0.90"), (None, "settlement")): POL[f"X   B0, sell at {nm}"] = dict(tp=tp)
RES = {k: run(**v) for k, v in POL.items()}; pickle.dump({k: (POL[k], {p: summ(RES[k], p) for p in ("ALL", "Jun1-Aug19", "tune", "TEST")}) for k in POL}, open(f"{DEEP}/policy_grid_res.pkl", "wb"))
print(f"backtest table: ${TABLE:,.0f}  |  $100 a trade; full = whole stake filled, real = limited to the $ that traded at our price\n")
print(f"{'policy':88s} trades  win    c/sh  taker |  ALL full   real  | % of table |  full: Jun-Aug19   tune    TEST |  real: Jun-Aug19   tune    TEST")
for k in POL:
    a = summ(RES[k]); p = {q: summ(RES[k], q) for q in ("Jun1-Aug19", "tune", "TEST")}
    print(f"{k:88s} {a['n']:>5d}  {a['win']:.1%}  {a['c']:+.2f}  {a['tk']:>4.0%} | {a['full']:>8,.0f} {a['real']:>7,.0f} |  {a['full']/TABLE:>3.0%} / {a['real']/TABLE:>3.0%} |      {p['Jun1-Aug19']['full']:>8,.0f} {p['tune']['full']:>7,.0f} {p['TEST']['full']:>7,.0f} |      {p['Jun1-Aug19']['real']:>8,.0f} {p['tune']['real']:>7,.0f} {p['TEST']['real']:>7,.0f}")
