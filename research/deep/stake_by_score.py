"""Adapting the stake to the strength of the signal.  1) Is the gradient real?  Raw accuracy of the T-120 signal by strength on the long history
(walk-forward scores, trading hours), Oct 2025 - May 2026 and Jun - Nov 2026 separately.  2) What a stake schedule does to the simulated fills of the live setup
(queue-aware, trading hours, stake limited to the dollars sold into us): profit, max drawdown, worst day, profit per dollar of drawdown.
  python research/deep/stake_by_score.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
S = pd.read_parquet("/home/user/data/deep/t120_lab_base.parquet")[["T", "ym", "x_twap", "base_r300"]].rename(columns={"base_r300": "s"}); S = S[(S.x_twap != 0) & np.isfinite(S.s)].reset_index(drop=True); months = sorted(S.ym.unique())
thr = np.full(len(S), np.nan)
for m in months[1:]: thr[S.ym.values == m] = S.s[S.ym.values < m].abs().quantile(0.70)          # threshold from the months before, as the bot does it
t = pd.to_datetime(S["T"], unit="s"); good = ((t.dt.dayofweek >= 5) | (t.dt.hour >= 16)).values; rs = np.abs(S.s.values) / thr; hit = (S.s.values > 0) == (S.x_twap.values > 0); sel = S.ym.values <= "2026-05"
BINS = [(1.0, 1.15), (1.15, 1.35), (1.35, 1.7), (1.7, 2.2), (2.2, 99)]; print("1) RAW ACCURACY BY SIGNAL STRENGTH (score / top-30 % threshold), trading hours, every signal whether filled or not")
for lo, hi in BINS:
    k = (rs >= lo) & (rs < hi) & good; print(f"   {lo:.2f}-{str(hi) if hi < 99 else '+':4s}x: Oct25-May26 {100*hit[k & sel].mean():.1f} % ({(k & sel).sum()})   Jun-Nov26 {100*hit[k & ~sel].mean():.1f} % ({(k & ~sel).sum()})   by quarter: " + " ".join(f"{q}:{100*hit[k & np.isin(S.ym.values, ms)].mean():.0f}" for q, ms in (("Nov-Jan", ["2025-11", "2025-12", "2026-01"]), ("Feb-Apr", ["2026-02", "2026-03", "2026-04"]), ("May-Jul", ["2026-05", "2026-06", "2026-07"]), ("Aug-Nov", ["2026-08", "2026-09", "2026-10", "2026-11"]))))
from policy_lab import run, Tix, date
r = run(dec=-120, tier=0.70, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=50.0); g = np.asarray((date.dayofweek >= 5) | (pd.to_datetime(Tix, unit="s").hour >= 16))[r["i"]]; o = np.argsort(Tix[r["i"][g]])
I = r["i"][g][o]; v = r["v"][g][o]; e = r["e"][g][o]; va = r["vol_after"][g][o]; q = r["rel_score"][g][o]; day = date.values.astype("datetime64[D]")[I]; per = np.where(date[I] < "2026-08-20", 0, np.where(date[I] < "2026-10-01", 1, 2)); ND = 160
def dd(x): c = np.cumsum(x); return float((np.maximum.accumulate(np.maximum(c, 0)) - c).max())
SCH = {"flat (now)": lambda q: np.ones(len(q)), "2x from 1.7x threshold": lambda q: np.where(q >= 1.7, 2.0, 1.0), "2x from 1.7x, 3x from 2.2x": lambda q: np.where(q >= 2.2, 3.0, np.where(q >= 1.7, 2.0, 1.0)),
       "half below 1.35x, 2x from 1.7x, 3x from 2.2x": lambda q: np.where(q >= 2.2, 3.0, np.where(q >= 1.7, 2.0, np.where(q >= 1.35, 1.0, 0.5))), "proportional to strength (1x .. 3x)": lambda q: np.clip(q, 1.0, 3.0), "only signals from 1.35x": lambda q: np.where(q >= 1.35, 1.0, 0.0), "only signals from 1.7x": lambda q: np.where(q >= 1.7, 1.0, 0.0)}
for base, cap, ttl in ((10, 1e9, "2) AT A $10 BASE STAKE (your account: every order fills completely at this size)"), (100, 200, "3) AT A $100 BASE STAKE, never more than $200 in one trade, stake limited to the dollars sold into us")):
    rows = []
    for nm, f in SCH.items():
        st = np.minimum(np.minimum(f(q) * base, cap), np.where(base <= 10, 1e9, va)); p = v * st / e + st / e * 0.0035; gday = pd.Series(p).groupby(day).sum(); k = st > 0
        rows.append({"schedule": nm, "trades": int(k.sum()), "avg stake $": round(st[k].mean(), 1), "total $": round(p.sum()), "per day $": round(p.sum() / ND, 1), "max drawdown $": round(dd(p)), "profit / drawdown": round(p.sum() / dd(p), 1), "worst day $": round(gday.min()), "losing days": int((gday < 0).sum()), "Jun1-Aug19": round(p[per == 0].sum()), "Aug20-Sep30": round(p[per == 1].sum()), "Oct1-Nov7": round(p[per == 2].sum())})
    R = pd.DataFrame(rows); f0 = R.loc[0, "max drawdown $"]; R["total $ at the SAME drawdown as flat"] = (R["total $"] * f0 / R["max drawdown $"]).round().astype(int); pd.set_option("display.width", 260); print("\n" + ttl); print(R.to_string(index=False))
