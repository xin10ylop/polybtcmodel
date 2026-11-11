"""ETH: the same audited treatment as BTC (queue-aware fills on the real trade tape), with the T-45 model (no T-120 history for ETH), and the
stake the market can really fill.   ASSET=ETH python research/deep/policy_eth.py"""
import os, sys, numpy as np, pandas as pd; os.environ.setdefault("ASSET", "ETH"); sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
import policy_lab as L
from policy_lab import run, STAKE, date, Tix, SC, THR, ok, S85, mon, MONTHS, TABLE, ASSET, P
rng = np.random.default_rng(4); PER = {"Jun-Aug19": date < "2026-08-20", "tune": (date >= "2026-08-20") & (date <= "2026-09-30"), "TEST": date >= "2026-10-01"}; ND = pd.Series(date).nunique()
def good(i): t = pd.to_datetime(Tix[i], unit="s"); return (t.dayofweek.values >= 5) | (t.hour.values >= 16)
def row(nm, r, m=None):
    m = np.ones(len(r["v"]), bool) if m is None else m; v = r["v"][m]; e = r["e"][m]; i = r["i"][m]
    if len(v) < 40: print(f"{nm:62s} too few ({len(v)})"); return
    d = pd.Series(v * STAKE / e).groupby(date[i]).sum(); bs = [d.values[q].sum() for q in (rng.integers(0, len(d), len(d)) for _ in range(1000))]
    print(f"{nm:62s} {len(v)/ND:>5.1f}/d  {(v>0).mean():.1%}  {v.mean()*100:+.2f}c  ${(v*STAKE/e).sum():>7,.0f} [{np.percentile(bs,2.5):>7,.0f},{np.percentile(bs,97.5):>7,.0f}] | " + " ".join(f"{(v[PER[k][i]]>0).mean():.1%}" for k in PER) + " | med $ at our price " + f"{np.median(r['vol'][m]):.0f}", flush=True)
sel = ok & np.isfinite(SC[-45]) & (np.abs(SC[-45]) >= THR[-45, 0.85]); bp = np.where(SC[-45] > 0, S85.pnl_up.values, S85.pnl_dn.values)
print(f"{ASSET}: {ND} days, old table (T-45, top 15 %, every signal at 0.50, sell {L.TP0}): {sel.sum()} trades, win {(bp[sel]>0).mean():.1%}, ${TABLE:,.0f} at $100; windows cached {len(P)}")
print(f"\n{'variant (decision T-45, bid 0.50, post-only)':62s} trades  win   c/sh   total@$100 if filled in full [95%]   | win by period | volume")
for Q in (0.0, 10.0, 25.0, 50.0):
    print(f"-- queue ahead of us: ${Q:.0f}" + (" (first sale at 0.50 fills us)" if Q == 0 else " (filled only once a print trades through 0.50 or this much has traded at 0.50)"))
    for tier, tn in ((0.85, "top15"), (0.70, "top30"), (0.50, "top50")):
        for tc in (-30, -15):
            for tp, tpn in ((L.TP0, f"sell {L.TP0}"), (None, "settle")):
                if Q in (0.0, 50.0) and not (tier == 0.70 and tc == -15): continue
                r = run(dec=-45, tier=tier, stages=((-43, tc, 0.50),), tp=tp, cross="skip", fresh=5, queue=Q); row(f"   {tn}, cancel T{tc}, {tpn}, all hours", r); row(f"   {tn}, cancel T{tc}, {tpn}, weekends + wd 16-24", r, good(r["i"]))
