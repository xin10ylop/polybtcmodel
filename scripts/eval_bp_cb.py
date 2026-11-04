"""Does the Coinbase premium improve the btc-predict 5-min forecast used in A's agreement filter?
AUC vs the Polymarket outcome at T-60 (tune / test), and the frozen A configuration with each filter."""
import os, sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
TUNE, TEST = se.TUNE, se.TEST
b = pd.read_parquet("/home/user/data/forecasts/bp_cb_minute.parquet")
b["T"] = ((pd.DatetimeIndex(b.ts).tz_localize(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s") + 60 + 60).astype(np.int64)   # forecast instant T-60 -> window T
m = ex.master.join(b.set_index("T")[[c for c in b.columns if c.startswith("z_")]], how="left")
for c in [c for c in m.columns if c.startswith("z_s5") or c.startswith("z_s1")]:
    for split, nm in ((TUNE, "tune"), (TEST, "test")):
        s = (m.date.values >= split[0]) & (m.date.values <= split[1]) & np.isfinite(m[c].values)
        print(f"{c:18s} {nm}: AUC vs outcome {roc_auc_score(m.up.values[s], m[c].values[s]):.4f}  n={s.sum()}")
sim = polysim.simulate(ex.mk, ex.g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=(0.80,), sls=(0.20,), tstops=(240,))
mm, sim = ex.align(m, sim); idx = np.arange(len(mm)); p = mm["m2_-45_close_p"].values; side = np.where(p > 0.5, 0, 1)
base = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05); pnl = sim["pnl"][idx, side, 0, 0, 0]
rows = []
for nm, z in (("no filter", None), ("agree bp core (frozen rule)", mm["z_s5_core"].values), ("agree bp core+cb", mm["z_s5_core+cb"].values), ("agree both", None)):
    if nm == "agree both": f = (np.sign(mm["z_s5_core"].values) == np.sign(p - 0.5)) & (np.sign(mm["z_s5_core+cb"].values) == np.sign(p - 0.5))
    else: f = np.ones(len(mm), bool) if z is None else (np.sign(z) == np.sign(p - 0.5))
    r = {"filter": nm}
    for split, sp in ((TUNE, "tune"), (TEST, "test")):
        s = base & f & (mm.date.values >= split[0]) & (mm.date.values <= split[1]); mt = se.metrics(np.where(s, pnl, np.nan), mm.date.values)
        r.update({f"{k}_{sp}": mt.get(k, np.nan) for k in ("n", "mean", "win", "sharpe", "tstat")})
    rows.append(r)
print(pd.DataFrame(rows).round(4).to_string(index=False))
# paired day-block bootstrap: core+cb filter vs core filter (difference of daily P&L sums over common days)
fa = (np.sign(mm["z_s5_core"].values) == np.sign(p - 0.5)); fb = (np.sign(mm["z_s5_core+cb"].values) == np.sign(p - 0.5))
for split, sp in ((TUNE, "tune"), (TEST, "test")):
    s = base & (mm.date.values >= split[0]) & (mm.date.values <= split[1])
    da = pd.Series(np.where(s & fa, pnl, 0.0)).groupby(mm.date.values).sum(); db = pd.Series(np.where(s & fb, pnl, 0.0)).groupby(mm.date.values).sum()
    diff = (db - da).values; rng = np.random.default_rng(0); boots = [np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(2000)]
    print(f"{sp}: daily P&L diff (core+cb minus core) mean {diff.mean():+.4f}/day  95% CI [{np.quantile(boots, .025):+.4f}, {np.quantile(boots, .975):+.4f}]")
