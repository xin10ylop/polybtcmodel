"""Pre-open Polymarket order flow (signed taker flow before T-45) as a side-confirmation / filter for A."""
import os, sys, glob, numpy as np, pandas as pd, pyarrow.parquet as pq, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
t = pd.read_csv(os.path.join(R, "trades_A_preopen.csv")); t = t[t.selected & t.filled].set_index("T")
mk = pd.read_parquet(polysim.MARKETS).set_index("window_start")
W = sorted(set(t.index.values) & set(mk.index.values)); rows = {}
for f in sorted(glob.glob(f"{polysim.TRADES}/*.parquet")):
    x = pq.read_table(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"], filters=[("window_start", "in", W)]).to_pandas()
    if len(x) == 0: continue
    x["rel"] = x.ts - x.window_start
    x = x[(x.rel >= -600) & (x.rel < -45)]
    upbuy = ((x.side == "BUY") & (x.outcome_index == 0)) | ((x.side == "SELL") & (x.outcome_index == 1))
    x["usd"] = x.price * x["size"]; x["signed"] = np.where(upbuy, x.usd, -x.usd)
    for w, s in x.groupby("window_start"):
        rows[int(w)] = dict(flow=s.signed.sum(), vol=s.usd.sum(), n=len(s), last_up=(s.price.where(s.outcome_index == 0, 1 - s.price)).iloc[-1])
F = pd.DataFrame(rows).T.reindex(t.index); t = t.join(F)
sg = np.where(t.side == "UP", 1.0, -1.0); flow_side = sg * t.flow.values; imb = flow_side / (t.vol.values + 1e-9)
last_side = np.where(t.side == "UP", t.last_up.values, 1 - t.last_up.values)
tune = ((t.date >= "2026-08-20") & (t.date <= "2026-09-30")).values; test = (t.date >= "2026-10-01").values
print("pre-open window [-600,-45): median USD volume", np.nanmedian(t.vol), " trades", np.nanmedian(t.n))
for nm, x in (("signed taker flow toward our side (USD)", flow_side), ("flow imbalance toward our side", imb), ("pre-open print of our side", last_side), ("pre-open USD volume", t.vol.values)):
    print(f"\n{nm}: corr with pnl tune {np.corrcoef(np.nan_to_num(x[tune]), t.pnl.values[tune])[0,1]:+.3f}  test {np.corrcoef(np.nan_to_num(x[test]), t.pnl.values[test])[0,1]:+.3f}")
    for s, sp in ((tune, "tune"), (test, "test")):
        ok = s & np.isfinite(x); q = pd.qcut(pd.Series(x[ok]).rank(method="first"), 4, labels=False)
        g = pd.Series(t.pnl.values[ok]).groupby(q.values).agg(["mean", "size"]); print(f"  {sp} quartile means (¢): {np.round(g['mean'].values*100, 2).tolist()}")
for nm, f in {"flow agrees (>0)": flow_side > 0, "flow disagrees": flow_side < 0, "imbalance > +0.2": imb > 0.2, "imbalance < -0.2": imb < -0.2}.items():
    r = []
    for s, sp in ((tune, "tune"), (test, "test")):
        x = t.pnl.values[f & s]; r.append(f"{sp}: n={len(x)} mean={x.mean()*100:+.2f}¢ win={np.mean(x>0):.3f}")
    print(f"{nm:20s} " + " | ".join(r))

# ---- combined with the btc-predict agreement filter (live rule) + paired day bootstrap
m = pd.read_parquet("/home/user/data/master.parquet", columns=["bp_-60_z_s5", "m2_-45_close_p"]).reindex(t.index)
agree = (np.sign(m["bp_-60_z_s5"].values) == np.sign(m["m2_-45_close_p"].values - 0.5))
old = (t.date < "2026-08-20").values
print("\n== with the live agreement filter ==")
rng = np.random.default_rng(0)
for nm, f in {"agreement only (live)": agree, "agreement + flow agrees": agree & (flow_side > 0), "agreement + flow not against (>= 0 or no prints)": agree & ~(flow_side < 0),
              "agreement + imbalance > +0.2": agree & (imb > 0.2)}.items():
    out = []
    for s, sp in ((old, "old"), (tune, "tune"), (test, "test")):
        x = t.pnl.values[f & s]; d_all = pd.Series(np.where(f & s, t.pnl.values, 0.0)).groupby(t.date.values).sum(); d_base = pd.Series(np.where(agree & s, t.pnl.values, 0.0)).groupby(t.date.values).sum()
        days = np.unique(t.date.values[s]); diff = (d_all - d_base).reindex(days).values; b = [np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(2000)]
        out.append(f"{sp}: n={len(x)} mean={x.mean()*100:+.2f}¢ win={np.mean(x>0):.3f} Δ/day={diff.mean()*100:+.1f}¢ p={np.mean(np.array(b)<=0):.2f}")
    print(f"{nm:48s} " + " | ".join(out))
