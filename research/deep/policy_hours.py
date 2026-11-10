"""Where the edge sits in the week (audited setup), and whether the same pattern is in the model's raw accuracy on an independent earlier
sample (Oct 2025 - May 2026, Binance TWAP label, no Polymarket fills involved).   python research/deep/policy_hours.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from policy_lab import run, STAKE, date, Tix, DEEP
BASE = dict(dec=-120, tier=0.70, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=50.0)
PER = {"Jun-Aug19": date < "2026-08-20", "tune": (date >= "2026-08-20") & (date <= "2026-09-30"), "TEST": date >= "2026-10-01"}
b = run(**BASE); t = pd.to_datetime(Tix[b["i"]], unit="s"); hr = t.hour.values; we = t.dayofweek.values >= 5; late = hr >= 16; rng = np.random.default_rng(2)
def row(nm, m):
    v = b["v"][m]; e = b["e"][m]; i = b["i"][m]; d = pd.Series(v).groupby(date[i]).sum(); c = pd.Series(np.ones(len(v))).groupby(date[i]).sum()
    bs = [d.values[k].sum() / c.values[k].sum() * 100 for k in (rng.integers(0, len(d), len(d)) for _ in range(2000))]
    print(f"{nm:44s} {len(v)/160:>5.1f}/day  win {(v>0).mean():.1%}  {v.mean()*100:+.2f} c [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]  ${(v*STAKE/e).sum():>7,.0f} | win " + " ".join(f"{(v[PER[k][i]]>0).mean():.1%}" for k in PER) + " | c/sh " + " ".join(f"{v[PER[k][i]].mean()*100:+.2f}" for k in PER))
print("AUDITED SETUP WITH FILLS (Jun 1 - Nov 7 2026)   trades   win    c/share [95%]        total@$100 | by period Jun-Aug19 / tune / TEST")
row("all", np.ones(len(hr), bool)); row("weekend, any hour", we); row("weekday 16-24 UTC", ~we & late); row("weekday 00-16 UTC", ~we & ~late)
row("weekend OR 16-24 UTC  (candidate filter)", we | late); row("weekend 16-24 UTC", we & late); row("weekend 00-16 UTC", we & ~late)
for h in range(0, 24, 2): row(f"   {h:02d}-{h+2:02d} UTC (all days)", (hr >= h) & (hr < h + 2))
# independent check: raw model accuracy on the Binance label, earlier months, no fills
D = pd.read_parquet(f"{DEEP}/ourcase_BTC_d-120_ph3.parquet"); D = D[np.isfinite(D.lgb_reg300)]; D["a"] = D.lgb_reg300.abs(); D["thr"] = D.groupby("ym").a.transform(lambda x: x.quantile(0.70)); D = D[D.a >= D.thr]
y = D.y_twap.values; D["hit"] = ((D.lgb_reg300.values > 0) == (y > 0.5 if set(np.unique(y[np.isfinite(y)])) <= {0, 1} else y > 0)); tt = pd.to_datetime(D["T"], unit="s"); D["we"] = tt.dt.dayofweek.values >= 5; D["late"] = tt.dt.hour.values >= 16
print("\nMODEL ONLY: share of top-30 % signals pointing the right way (every signal, no fills)")
for nm, m in (("Oct 2025 - May 2026 (earlier, independent)", D.ym <= "2026-05"), ("Jun - Nov 2026", D.ym >= "2026-06")):
    g = D[m]; print(f"  {nm:42s} all {g.hit.mean():.1%} (n {len(g)}) | weekend {g.hit[g.we].mean():.1%} | weekday 16-24 UTC {g.hit[~g.we & g.late].mean():.1%} | weekday 00-16 UTC {g.hit[~g.we & ~g.late].mean():.1%} | weekend-or-late {g.hit[g.we | g.late].mean():.1%}")
    print("     by 4 h block (all days):", " ".join(f"{h:02d}-{h+4:02d}: {g.hit[(pd.to_datetime(g['T'], unit='s').dt.hour.values // 4) == h // 4].mean():.1%}" for h in range(0, 24, 4)))
