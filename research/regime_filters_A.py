"""Regime filters for strategy A on the saved trade records (thresholds from TUNE medians only)."""
import os, numpy as np, pandas as pd
pd.set_option("display.width", 230)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
t = pd.read_csv(os.path.join(R, "trades_A_preopen.csv")); t = t[t.selected & t.filled].drop(columns=["vol_prev5m_bp", "hour_et"])
m = pd.read_parquet("/home/user/data/master.parquet", columns=["m2_-45_close_p", "bp_-60_z_s5", "m2_-45_seas_now", "m2_-45_reg_ll_beta_1h", "m2_-45_reg_trend_1d", "vol_prev5m_bp", "hour_et"])
t = t.join(m, on="T"); sg = np.where(t.side == "UP", 1.0, -1.0); p = t["m2_-45_close_p"].values
tune = ((t.date >= "2026-08-20") & (t.date <= "2026-09-30")).values; test = ((t.date >= "2026-10-01") & (t.date <= "2026-11-03")).values
seas, beta, trend = t["m2_-45_seas_now"].values, t["m2_-45_reg_ll_beta_1h"].values, sg * t["m2_-45_reg_trend_1d"].values
agree = np.sign(t["bp_-60_z_s5"].values) == np.sign(p - 0.5)
qs, qb = np.nanmedian(seas[tune]), np.nanmedian(beta[tune])
def mets(mask):
    out = {}
    for s, nm in ((tune, "tune"), (test, "test")):
        x = t.pnl.values[mask & s]; d = pd.Series(x).groupby(t.date.values[mask & s]).sum()
        out[f"n_{nm}"] = len(x); out[f"mean_{nm}"] = x.mean() if len(x) else np.nan; out[f"win_{nm}"] = np.mean(x > 0) if len(x) else np.nan
        out[f"sharpe_{nm}"] = d.mean() / (d.std() + 1e-12) * np.sqrt(365); out[f"per_day_{nm}"] = len(x) / max(d.shape[0], 1); out[f"negdays_{nm}"] = f"{int((d < 0).sum())}/{d.shape[0]}"
    return out
rows = []
for nm, f in {"frozen rule (no filter)": np.ones(len(t), bool), "+ bp agreement (live)": agree, "quiet + agreement": (seas < qs) & agree,
              "low beta + agreement": (beta < qb) & agree, "counter-trend + agreement": (trend < 0) & agree, "quiet + low beta + agreement": (seas < qs) & (beta < qb) & agree}.items():
    r = {"filter": nm}; r.update(mets(f)); rows.append(r)
print(pd.DataFrame(rows).round(4).to_string(index=False)); print("seas_now tune median", round(qs, 3), " beta tune median", round(qb, 3))
