"""Hypothesis: previous window's close pressure (push toward its own reference in the last ~10 s
before T) happens INSIDE our effective window (our reference is fixed at ~T-30). Model-free check."""
import numpy as np, pandas as pd
m = pd.read_parquet("/home/user/data/master.parquet")
paths = np.load("/home/user/data/truth/paths.npy"); tw = pd.read_parquet("/home/user/data/truth/windows.parquet")
ti = {t: i for i, t in enumerate(tw["T"].values)}
rows = np.array([ti[t] for t in m.index]); lp = np.log(paths[rows])      # col 120 == T, cols T-120 .. T+300
prev = m["ref30"].reindex(m.index - 300).values                            # previous window's lagged reference = c[T-330]
ok = np.isfinite(prev)
prev_ref = np.log(prev)
x45 = 1e4 * (lp[:, 120 - 46] - prev_ref)
mv_dec_ref = 1e4 * (lp[:, 120 - 30] - lp[:, 120 - 46]); mv_ref_T = 1e4 * (lp[:, 120] - lp[:, 120 - 30]); mv_T_end = 1e4 * (lp[:, 120 + 270] - lp[:, 120])
x_eff = m.x_eff_bps.values; vol = m.vol_prev5m_bp.values
z = x45 / np.maximum(vol, 1.0)
b = pd.cut(z, [-10, -1.5, -0.8, -0.4, -0.15, 0.15, 0.4, 0.8, 1.5, 10])
df = pd.DataFrame({"z_prev": b, "mv_dec_ref": mv_dec_ref, "mv_ref_T": mv_ref_T, "mv_T_end": mv_T_end, "x_eff": x_eff, "up": m.up.values, "month": m.month.values})[ok]
pd.set_option("display.width", 200)
print("displacement from previous window's reference at T-46 (5-min-vol units) -> mean subsequent moves (bp), P(up)")
print(df.groupby("z_prev", observed=True).agg(n=("up", "size"), mv_dec_ref=("mv_dec_ref", "mean"), mv_ref_T=("mv_ref_T", "mean"),
                                              mv_T_end=("mv_T_end", "mean"), x_eff=("x_eff", "mean"), p_up=("up", "mean")).round(3).to_string())
sel = (np.abs(z) > 0.4) & ok
for mo in sorted(set(df.month)):
    s = sel & (m.month.values == mo)
    print(mo, s.sum(), "P(up == (z_prev<0)) =", np.mean(m.up.values[s] == (z[s] < 0)).round(4), " corr(x_eff,-z) =", np.corrcoef(x_eff[s], -z[s])[0, 1].round(4))
print("\nmove [T-46,T-30] vs x_eff: corr", np.corrcoef(mv_dec_ref[ok], x_eff[ok])[0, 1].round(4), " | [T-30,T] vs [T,T+270]:", np.corrcoef(mv_ref_T[ok], mv_T_end[ok])[0, 1].round(4),
      " | [T-120,T-46] vs x_eff:", np.corrcoef(1e4 * (lp[ok, 120 - 46] - lp[ok, 0]), x_eff[ok])[0, 1].round(4))
