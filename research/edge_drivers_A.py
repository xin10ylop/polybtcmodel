"""Which decision-time parameters separate strategy A's per-trade P&L the most? (fixed fill 0.50 at T-45,
TP 0.80 / SL 0.20 / time-stop T+240, side = btcmovement2 close head sign). Deciles -> spread top-bottom."""
import os, sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
TUNE, TEST = se.TUNE, se.TEST
sim = polysim.simulate(ex.mk, ex.g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=(0.80,), sls=(0.20,), tstops=(240,))
m, sim = ex.align(ex.master.copy(), sim); idx = np.arange(len(m))
p = m["m2_-45_close_p"].values; side = np.where(p > 0.5, 0, 1); pnl = sim["pnl"][idx, side, 0, 0, 0]
pre = sim["pre_last"][idx, side]
params = {
    "model confidence |p-0.5| (m2 close @-45)": np.abs(p - 0.5),
    "btc-predict 5m z, signed by side": np.where(side == 0, 1, -1) * m["bp_-60_z_s5"].values,
    "btc-movement close_p, signed by side": np.where(side == 0, 1, -1) * (m["bm_-45_close_p"].values - 0.5),
    "m2 10s head, signed by side": np.where(side == 0, 1, -1) * (m["m2_-45_h10_pup_now"].values - 0.5),
    "m2 60s head, signed by side": np.where(side == 0, 1, -1) * (m["m2_-45_h60_pup_now"].values - 0.5),
    "pre-open market print of our side": pre,
    "5-min realized vol (bp)": m.vol_prev5m_bp.values,
    "1-day RV (log, m2 lrv_86400)": m["m2_-45_lrv_86400"].values,
    "seasonal activity (seas_now)": m["m2_-45_seas_now"].values,
    "perp lead (pls_5, signed by side)": np.where(side == 0, 1, -1) * m["m2_-45_pls_5"].values,
    "basis deviation (bdev_60, signed by side)": np.where(side == 0, 1, -1) * m["m2_-45_bdev_60"].values,
    "last 30s return r_30, signed by side": np.where(side == 0, 1, -1) * m["m2_-45_r_30"].values,
    "prev window return, signed by side": np.where(side == 0, 1, -1) * m["m2_-45_prev_ret_v"].values,
    "1-day trend (reg_trend_1d), signed": np.where(side == 0, 1, -1) * m["m2_-45_reg_trend_1d"].values,
    "lead-lag beta regime (reg_ll_beta_1h)": m["m2_-45_reg_ll_beta_1h"].values,
    "catch-up regime (reg_catchup_1h)": m["m2_-45_reg_catchup_1h"].values,
    "hour of day (ET)": m.hour_et.values,
    "market volume of the window (USDC)": m.volume.values,
}
rows = []
for nm, x in params.items():
    r = {"parameter": nm}
    for split, sp in ((TUNE, "tune"), (TEST, "test")):
        s = (m.date.values >= split[0]) & (m.date.values <= split[1]) & np.isfinite(x) & np.isfinite(pnl)
        q = pd.qcut(pd.Series(x[s]).rank(method="first"), 5, labels=False)
        g = pd.Series(pnl[s]).groupby(q.values).mean()
        r[f"q1_{sp}"] = g.iloc[0]; r[f"q5_{sp}"] = g.iloc[-1]; r[f"spread_{sp}"] = g.iloc[-1] - g.iloc[0]
        r[f"corr_{sp}"] = np.corrcoef(x[s], pnl[s])[0, 1]
    rows.append(r)
df = pd.DataFrame(rows); df["avg_spread"] = (df.spread_tune + df.spread_test) / 2
print(df.sort_values("avg_spread", ascending=False).round(4).to_string(index=False))
