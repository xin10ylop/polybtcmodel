"""Deeper look at the trading-hours effect.  A-E use the models' raw accuracy (every top-30 % signal, Binance TWAP label, no Polymarket fills)
over all walk-forward months since Oct 2025; F-G use the audited setup with fills (Jun 1 - Nov 7 2026).
  python research/deep/policy_hours2.py"""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from zoneinfo import ZoneInfo
DEEP = "/home/user/data/deep"; rng = np.random.default_rng(3)
def prep(fn, col="lgb_reg300"):
    D = pd.read_parquet(f"{DEEP}/{fn}.parquet"); D = D[np.isfinite(D[col]) & np.isfinite(D.x_twap)].copy(); D["a"] = D[col].abs(); D = D[D.a >= D.groupby("ym").a.transform(lambda x: x.quantile(0.70))]
    D["hit"] = (D[col] > 0) == (D.x_twap > 0); D = D[D.x_twap != 0]; t = pd.to_datetime(D["T"], unit="s", utc=True); D["day"] = t.dt.date; D["hr"] = t.dt.hour; D["we"] = t.dt.dayofweek >= 5
    D["et"] = t.dt.tz_convert(ZoneInfo("America/New_York")).dt.hour; D["seg"] = np.where(D.we, "weekend", np.where(D.hr >= 16, "wd 16-24", "wd 00-16")); return D
def ci(g):
    d = g.groupby("day").hit.agg(["sum", "count"]); b = [d["sum"].values[k].sum() / d["count"].values[k].sum() for k in (rng.integers(0, len(d), len(d)) for _ in range(1000))]; return np.percentile(b, [2.5, 97.5])
B120 = prep("ourcase_BTC_d-120_ph3"); B45 = prep("ourcase_BTC_ph5"); E45 = prep("ourcase_ETH_ph5")
print("A. BTC T-120 model, share of top-30 % signals pointing the right way, month by month")
pv = B120.pivot_table(index="ym", columns="seg", values="hit", aggfunc="mean")[["weekend", "wd 16-24", "wd 00-16"]]; n = B120.pivot_table(index="ym", columns="seg", values="hit", aggfunc="count")
print((pv * 100).round(1).assign(n_weekend=n["weekend"], n_wd_late=n["wd 16-24"], n_wd_day=n["wd 00-16"]).to_string())
print(f"   months where weekend beats weekday 00-16: {(pv['weekend'] > pv['wd 00-16']).sum()}/{len(pv)}; where weekday 16-24 beats weekday 00-16: {(pv['wd 16-24'] > pv['wd 00-16']).sum()}/{len(pv)}")
print("\nB. Pooled, with day-bootstrap 95 % ranges (three models; ETH is an independent market)")
for nm, D in (("BTC T-120", B120), ("BTC T-45", B45), ("ETH T-45", E45)):
    for per, m in (("Oct25-May26", D.ym <= "2026-05"), ("Jun-Nov26", D.ym >= "2026-06"), ("all", D.ym > "")):
        g = D[m]
        if len(g) < 500: continue
        out = f"   {nm:9s} {per:12s}"
        for s in ("weekend", "wd 16-24", "wd 00-16"):
            q = g[g.seg == s]; c = ci(q); out += f" | {s} {q.hit.mean():.1%} [{c[0]:.1%},{c[1]:.1%}] n={len(q)}"
        print(out)
print("\nC. BTC T-120, all months pooled, by 2-hour block (UTC): weekday | weekend")
for h in range(0, 24, 2):
    a = B120[(B120.hr // 2 == h // 2) & ~B120.we]; b = B120[(B120.hr // 2 == h // 2) & B120.we]; print(f"   {h:02d}-{h+2:02d} UTC   weekday {a.hit.mean():.1%} (n {len(a):>4d})   weekend {b.hit.mean():.1%} (n {len(b):>4d})")
print("\nD. BTC T-120, weekdays only, by hour in New York time (is it tied to the US session?)")
wd = B120[~B120.we]; print("   " + "  ".join(f"{h:02d}h {wd.hit[wd.et == h].mean():.0%}" for h in range(24)))
print("\nE. Is it the clock or the market state?  BTC T-120, all months: accuracy by volatility and by volume at the decision (terciles within month)")
F = pd.concat([pd.read_parquet(f, columns=["T", "lrv_3600", "lvol_3600", "lrv_300"]) for f in glob.glob(f"{DEEP}/feats_BTC_d-120_ph0_*.parquet")]).drop_duplicates("T").set_index("T"); G = B120.join(F, on="T")
for col, nm in (("lrv_3600", "1-hour volatility"), ("lvol_3600", "1-hour volume")):
    G["q"] = G.groupby("ym")[col].transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=["low", "mid", "high"]))
    print(f"   {nm}: " + " | ".join(f"{q}: all {G.hit[G.q == q].mean():.1%}, weekend {G.hit[(G.q == q) & (G.seg == 'weekend')].mean():.1%}, wd 16-24 {G.hit[(G.q == q) & (G.seg == 'wd 16-24')].mean():.1%}, wd 00-16 {G.hit[(G.q == q) & (G.seg == 'wd 00-16')].mean():.1%}" for q in ("low", "mid", "high")))
    print(f"      share of each segment's signals in the low / mid / high tercile: " + " | ".join(f"{s}: " + "/".join(f"{(G.q[G.seg == s] == q).mean():.0%}" for q in ("low", "mid", "high")) for s in ("weekend", "wd 16-24", "wd 00-16")))
# ---- with fills
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); import policy_lab as L
from policy_lab import run, STAKE, Tix, SC, THR, ok, upwin, P, date
b = run(dec=-120, tier=0.70, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=50.0); sc = SC[-120]; sel = ok & np.isfinite(sc) & (np.abs(sc) >= THR[-120, 0.70]); idx = np.flatnonzero(sel)
t = pd.to_datetime(Tix[idx], unit="s"); seg = np.where(t.dayofweek.values >= 5, "weekend", np.where(t.hour.values >= 16, "wd 16-24", "wd 00-16")); win = np.where(sc[idx] > 0, upwin[idx], ~upwin[idx]); filled = np.isin(idx, b["i"])
vol = np.array([float(P[w][3][P[w][0] < 0].sum()) if w in P else np.nan for w in Tix[idx]]); a0 = pd.Series(b["a0"], index=b["i"]).reindex(idx).values
ft = pd.Series(b["ft"], index=b["i"]).reindex(idx).values
print("\nF. With fills (audited setup, Jun 1 - Nov 7): what happens to the signals in each part of the week")
print("   segment     signals/day  model right (all signals)  share filled  win if filled  win if NOT filled  median $ traded before the open  median fill time")
for s in ("weekend", "wd 16-24", "wd 00-16"):
    m = seg == s; print(f"   {s:10s}    {m.sum()/160:>5.1f}          {win[m].mean():.1%}                {filled[m].mean():.0%}          {win[m & filled].mean():.1%}          {win[m & ~filled].mean():.1%}               ${np.nanmedian(vol[m]):>7,.0f}                      T{np.nanmedian(ft[m & filled]):+.0f}s")
print("\nG. Other ways to draw the line (with fills): trades/day, win, c/share, total at $100, win by period Jun-Aug19 / tune / TEST")
tb = pd.to_datetime(Tix[b["i"]], unit="s", utc=True); hr = tb.hour.values; we = tb.dayofweek.values >= 5; et = tb.tz_convert(ZoneInfo("America/New_York")).hour.values; dow = tb.dayofweek.values
PER = {"a": date < "2026-08-20", "b": (date >= "2026-08-20") & (date <= "2026-09-30"), "c": date >= "2026-10-01"}
def row(nm, m):
    v = b["v"][m]; i = b["i"][m]; print(f"   {nm:52s} {len(v)/160:>5.1f}  {(v>0).mean():.1%}  {v.mean()*100:+.2f}  ${(v*STAKE/b['e'][m]).sum():>7,.0f} | " + " ".join(f"{(v[PER[k][i]]>0).mean():.1%}" for k in PER))
row("everything", np.ones(len(hr), bool)); row("weekends + weekdays 16-24 UTC (adopted)", we | (hr >= 16)); row("weekends only", we); row("weekends + weekdays 14-24 UTC", we | (hr >= 14)); row("weekends + weekdays 18-24 UTC", we | (hr >= 18))
row("weekends + weekdays 16-22 UTC", we | ((hr >= 16) & (hr < 22))); row("weekends + weekdays from 12:00 New York time", we | (et >= 12)); row("weekends + weekdays from 11:00 New York time", we | (et >= 11)); row("weekdays 16-24 UTC only", ~we & (hr >= 16))
for d, nm in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")): row(f"   {nm}, 00-16 UTC", (dow == d) & (hr < 16))
for d, nm in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")): row(f"   {nm}, 16-24 UTC", (dow == d) & (hr >= 16))
# ---- H. inside the good hours: does a wider or narrower signal tier pay? (with fills)
print("\nH. Inside weekends + weekday 16-24 UTC, by signal tier (with fills): trades/day, win, c/share, total at $100, win by period")
for tier, nm in ((0.85, "top 15 %"), (0.70, "top 30 % (adopted)"), (0.50, "top 50 %")):
    r = run(dec=-120, tier=tier, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=50.0); tt = pd.to_datetime(Tix[r["i"]], unit="s"); k = (tt.dayofweek.values >= 5) | (tt.hour.values >= 16)
    for lab, m in (("good hours", k), ("weekday 00-16", ~k)):
        v = r["v"][m]; i = r["i"][m]; d = pd.Series(v * STAKE / r["e"][m]).groupby(date[i]).sum(); bs = [d.values[q].sum() for q in (rng.integers(0, len(d), len(d)) for _ in range(1000))]
        print(f"   {nm:20s} {lab:14s} {len(v)/160:>5.1f}  {(v>0).mean():.1%}  {v.mean()*100:+.2f}  ${(v*STAKE/r['e'][m]).sum():>7,.0f} [{np.percentile(bs,2.5):>7,.0f},{np.percentile(bs,97.5):>7,.0f}] | " + " ".join(f"{(v[PER[q][i]]>0).mean():.1%}" for q in PER))
print("\nI. Raw model accuracy by tier and part of the week, BTC T-120, all 14 months (no fills)")
A = pd.read_parquet(f"{DEEP}/ourcase_BTC_d-120_ph3.parquet"); A = A[np.isfinite(A.lgb_reg300) & (A.x_twap != 0)].copy(); A["a"] = A.lgb_reg300.abs(); A["rk"] = A.groupby("ym").a.rank(pct=True); A["hit"] = (A.lgb_reg300 > 0) == (A.x_twap > 0)
ta = pd.to_datetime(A["T"], unit="s"); A["seg"] = np.where(ta.dt.dayofweek >= 5, "weekend", np.where(ta.dt.hour >= 16, "wd 16-24", "wd 00-16"))
for lo, hi, nm in ((0.95, 1.01, "top 5 %"), (0.85, 0.95, "5-15 %"), (0.70, 0.85, "15-30 %"), (0.50, 0.70, "30-50 %"), (0.0, 0.50, "bottom half")):
    g = A[(A.rk >= lo) & (A.rk < hi)]; print(f"   {nm:12s} " + " | ".join(f"{s} {g.hit[g.seg == s].mean():.1%} (n {int((g.seg == s).sum())})" for s in ("weekend", "wd 16-24", "wd 00-16")))
