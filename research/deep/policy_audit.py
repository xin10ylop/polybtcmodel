"""Audit of the policy simulator (2026-11-11): the same setups under stricter fill rules - a resting bid fills only once a print trades through
0.50 or $50 / $100 has traded at 0.50 (queue position), crossing taken / refused, take-profit on touch or through, hold to settlement.
  python research/deep/policy_audit.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from policy_lab import run, summ, TABLE, date, STAKE
C = dict(cancel=True); days = pd.DatetimeIndex(sorted(set(date))); rng = np.random.default_rng(1)
def line(nm, **kw):
    r = run(**kw); full = r["v"] * STAKE / r["e"]; s = pd.Series(full).groupby(pd.to_datetime(date[r["i"]]).values).sum().reindex(days, fill_value=0.0).values
    bs = [s[rng.integers(0, len(s), len(s))].sum() for _ in range(1000)]; p = {q: summ(r, q) for q in ("Jun1-Aug19", "tune", "TEST")}
    print(f"{nm:74s} {len(full):>6d} {(r['v']>0).mean():.1%} {r['v'].mean()*100:+.2f} | ${full.sum():>7,.0f} {full.sum()/TABLE:>4.0%} [{np.percentile(bs,2.5):>7,.0f},{np.percentile(bs,97.5):>7,.0f}] | {p['Jun1-Aug19']['full']:>7,.0f} {p['tune']['full']:>7,.0f} {p['TEST']['full']:>7,.0f} | c/sh {p['Jun1-Aug19']['c']:+.2f} {p['tune']['c']:+.2f} {p['TEST']['c']:+.2f}", flush=True)
print(f"{'variant (T-120, bid 0.50 to T-15)':74s} trades  win   c/sh |   full  %tab  [95% range]       |  Jun-Aug19   tune    TEST")
B = dict(tier=0.70, confirm=C)
print("-- as simulated (first touch fills)")
line("K2  top30, confirm, sell 0.90, cross=take (the claim)", tp=0.90, **B)
line("    top30, confirm, settle, cross=take", tp=None, **B)
line("    top30, confirm, settle, cross=skip (post-only refused, no order)", tp=None, cross="skip", **B)
line("    top30, confirm, settle, cross=take only with fresh evidence (+-5 s)", tp=None, fresh=5, **B)
line("    top30, confirm, settle, cross=skip, fresh +-5 s", tp=None, cross="skip", fresh=5, **B)
print("-- conservative fills: through 0.50 or $50 traded at 0.50 before we count a fill")
Q = dict(queue=50.0, fresh=5)
line("    top30, confirm, sell 0.90 (touch), cross=take", tp=0.90, **B, **Q)
line("    top30, confirm, sell 0.90 (through), cross=take", tp=0.90, tp_through=True, **B, **Q)
line("    top30, confirm, settle, cross=take", tp=None, **B, **Q)
line("    top30, confirm, settle, cross=skip", tp=None, cross="skip", **B, **Q)
line("    top15, confirm, settle, cross=take", tp=None, tier=0.85, confirm=C, **Q)
line("    top15, confirm, settle, cross=skip", tp=None, tier=0.85, confirm=C, cross="skip", **Q)
line("    top50, confirm, settle, cross=take", tp=None, tier=0.50, confirm=C, **Q)
line("    top30, NO confirm, settle, cross=take", tp=None, tier=0.70, **Q)
line("    top30, confirm, settle, cross=take, cancel at T-30", tp=None, stages=((-118, -30, 0.50),), **B, **Q)
print("-- stricter queue: through or $100")
line("    top30, confirm, settle, cross=take", tp=None, queue=100.0, fresh=5, **B)
line("    top15, confirm, settle, cross=take", tp=None, tier=0.85, confirm=C, queue=100.0, fresh=5)
print("-- for reference: T-45 decision, bid to T-30, conservative fills ($50), settle / 0.85")
line("    T-45 top15, bid to T-30, sell 0.85, cross=skip (what ran live on Nov 10)", dec=-45, stages=((-43, -30, 0.50),), tp=0.85, cross="skip", **Q)
line("    T-45 top15, bid to T-15, settle, cross=take", dec=-45, stages=((-43, -15, 0.50),), tp=None, **Q)
