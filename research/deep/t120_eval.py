"""Compare the variants of t120_lab.py.  SELECT = Oct 2025 - May 2026 (the only period used to choose), CONFIRM = Jun - Nov 2026.
For each: AUC against the Binance TWAP label, and the share of the top 30 % / top 15 % strongest signals that point the right way
(all hours, and in the trading hours = weekends + weekday 16-24 UTC).   python research/deep/t120_eval.py"""
import glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
fs = sorted(glob.glob("/home/user/data/deep/t120_lab_*.parquet")); B = None
for f in fs:
    d = pd.read_parquet(f); B = d if B is None else B.merge(d.drop(columns=["ym", "y_twap", "x_twap"]), on="T", how="outer")
cols = [c for c in B.columns if c not in ("T", "ym", "y_twap", "x_twap")]; rk = lambda v: pd.Series(v).rank(pct=True).values - 0.5
def addens(nm, parts):
    if all(p in B.columns for p in parts): B[nm] = np.mean([rk(B[p].values) * 2 for p in parts], axis=0) * np.nanstd(B[parts[0]].values); cols.append(nm)
addens("ENS base r150+300+500", ["base_r150", "base_r300", "base_r500"]); addens("ENS seeds x3", ["seeds_r300_s1", "seeds_r300_s2", "seeds_r300_s3"]); addens("ENS seeds x3 + base", ["seeds_r300_s1", "seeds_r300_s2", "seeds_r300_s3", "base_r300"])
addens("ENS base + bin", ["base_r300", "bin_r300"]); addens("ENS base + big", ["base_r300", "big_r300"]); addens("ENS base + recent365", ["base_r300", "recent365_r300"]); addens("ENS base + small", ["base_r300", "small_r600"]); addens("ENS base+big+bin+slow", ["base_r300", "big_r300", "bin_r300", "slow_r600"])
t = pd.to_datetime(B["T"], unit="s"); good = ((t.dt.dayofweek >= 5) | (t.dt.hour >= 16)).values; x = B.x_twap.values; yv = (x > 0); nz = x != 0
def hit(s, m, q):
    a = np.abs(s); thr = pd.Series(a).groupby(B.ym.values).transform(lambda v: v.quantile(q)).values; k = m & nz & np.isfinite(s) & (a >= thr); return ((s[k] > 0) == yv[k]).mean(), k.sum()
rows = []
for c in cols:
    s = B[c].values; r = {"variant": c}
    for nm, m in (("SELECT", (B.ym.values <= "2026-05")), ("CONFIRM", (B.ym.values >= "2026-06"))):
        k = m & np.isfinite(s) & nz; r[f"{nm} auc"] = roc_auc_score(yv[k], s[k]); r[f"{nm} top30"] = hit(s, m, 0.70)[0]; r[f"{nm} top30 hrs"] = hit(s, m & good, 0.70)[0]; r[f"{nm} top15"] = hit(s, m, 0.85)[0]
    rows.append(r)
R = pd.DataFrame(rows).set_index("variant"); base = R.loc["base_r300"] if "base_r300" in R.index else None
pd.set_option("display.width", 250); print((R * 100).round(2).sort_values("SELECT top30", ascending=False).to_string())
if base is not None: print("\ndifference vs base_r300, in points:"); print(((R - base) * 100).round(2).sort_values("SELECT top30", ascending=False).to_string())
n1 = hit(B["base_r300"].values, B.ym.values <= "2026-05", 0.70)[1]; n2 = hit(B["base_r300"].values, B.ym.values >= "2026-06", 0.70)[1]
print(f"\ntop-30 % signals: {n1} in SELECT, {n2} in CONFIRM -> one standard error of a hit rate is about {100*0.5/np.sqrt(n1):.2f} / {100*0.5/np.sqrt(n2):.2f} points; differences smaller than about twice that are noise")

# paired check by day blocks: is a variant's gain over base_r300 bigger than chance?  (1,000 resamples of whole days, SELECT period only)
rng = np.random.default_rng(0); sel = (B.ym.values <= "2026-05") & nz; day = (B["T"].values // 86400); days = np.unique(day[sel]); b0 = B["base_r300"].values
def stats(s):   # per-day sums needed for the top-30 % hit rate and for the rank correlation with the move
    a = np.abs(s); thr = pd.Series(a).groupby(B.ym.values).transform(lambda v: v.quantile(0.70)).values; k = sel & np.isfinite(s) & (a >= thr); w = ((s > 0) == yv) & k
    g = pd.DataFrame({"d": day[sel], "k": k[sel], "w": w[sel], "ic": (rk(s)[sel] * np.sign(x[sel]))}).groupby("d").sum(); return g.reindex(days).fillna(0).values
S0 = stats(b0); draws = rng.integers(0, len(days), (1000, len(days))); out = []
for c in cols:
    if c == "base_r300": continue
    S1 = stats(B[c].values); d_hit = S1[draws, 1].sum(1) / S1[draws, 0].sum(1) - S0[draws, 1].sum(1) / S0[draws, 0].sum(1); d_ic = (S1[draws, 2].sum(1) - S0[draws, 2].sum(1)) / sel.sum()
    out.append({"variant": c, "top30 gain pts": 100 * d_hit.mean(), "lo": 100 * np.quantile(d_hit, 0.025), "hi": 100 * np.quantile(d_hit, 0.975), "P(gain>0)": (d_hit > 0).mean(), "P(rank-score gain>0)": (d_ic > 0).mean()})
print("\nSELECT period, gain over base_r300 with day-block resampling (95 % range):"); print(pd.DataFrame(out).set_index("variant").round(3).sort_values("top30 gain pts", ascending=False).to_string())
