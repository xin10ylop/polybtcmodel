"""ETH markets traded on the BTC T-120 signal (ETH has no T-120 model of its own), queue-aware fills.  ASSET=ETH XSCORE=BTC python research/deep/policy_eth_x.py"""
import os, sys, numpy as np, pandas as pd; os.environ.setdefault("ASSET", "ETH"); os.environ.setdefault("XSCORE", "BTC"); sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
import policy_lab as L
from policy_lab import run, STAKE, date, Tix, SC, THR, ok, upwin
rng = np.random.default_rng(6); PER = {"a": date < "2026-08-20", "b": (date >= "2026-08-20") & (date <= "2026-09-30"), "c": date >= "2026-10-01"}; ND = pd.Series(date).nunique()
def good(i): t = pd.to_datetime(Tix[i], unit="s"); return (t.dayofweek.values >= 5) | (t.hour.values >= 16)
def row(nm, r, m=None):
    m = np.ones(len(r["v"]), bool) if m is None else m; v = r["v"][m]; e = r["e"][m]; i = r["i"][m]
    if len(v) < 40: print(f"{nm:66s} too few ({len(v)})"); return
    d = pd.Series(v * STAKE / e).groupby(date[i]).sum(); bs = [d.values[q].sum() for q in (rng.integers(0, len(d), len(d)) for _ in range(1000))]
    print(f"{nm:66s} {len(v)/ND:>5.1f}/d  {(v>0).mean():.1%}  {v.mean()*100:+.2f}c  ${(v*STAKE/e).sum():>7,.0f} [{np.percentile(bs,2.5):>7,.0f},{np.percentile(bs,97.5):>7,.0f}] | " + " ".join(f"{(v[PER[k][i]]>0).mean():.1%}" for k in PER) + f" | med $ {np.median(r['vol'][m]):.0f} / after queue {np.median(r['vol_after'][m]):.0f}", flush=True)
sc = SC[-120]
for tier, tn in ((0.85, "top15"), (0.70, "top30")):
    sel = ok & np.isfinite(sc) & (np.abs(sc) >= THR[-120, tier]); w = np.where(sc[sel] > 0, upwin[sel], ~upwin[sel]); g = good(np.flatnonzero(sel))
    print(f"BTC T-120 signal on ETH outcomes, {tn}: {sel.sum()/ND:.1f} signals/day, right {w.mean():.1%} on all signals (good hours {w[g].mean():.1%}, weekday 00-16 UTC {w[~g].mean():.1%})")
print(f"\n{'variant (BTC T-120 signal, ETH market, bid 0.50, post-only, settle)':66s} trades  win   c/sh   total@$100 full [95%]            | win by period | $ at our price")
for Q in (0.0, 10.0, 25.0):
    print(f"-- queue ahead of us: ${Q:.0f}")
    for tier, tn in ((0.85, "top15"), (0.70, "top30")):
        for cf, cn in ((None, "no confirmation"), (dict(cancel=True), "ETH T-45 confirmation")):
            r = run(dec=-120, tier=tier, confirm=cf, tp=None, cross="skip", fresh=5, queue=Q); row(f"   {tn}, {cn}, all hours", r); row(f"   {tn}, {cn}, weekends + wd 16-24", r, good(r["i"]))
