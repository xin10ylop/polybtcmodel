"""Polymarket's own pre-decision information per window (everything before T-45): the last UP-equivalent price, and the net taker
flow toward UP in $ (taker BUY UP or SELL DOWN minus taker BUY DOWN or SELL UP) over several lookbacks.  Saved to
/home/user/data/deep/pmflow_<asset>.parquet; prints the AUC of each vs the outcome by month, alone and next to the model.
  ASSET=BTC python research/deep/pm_flow.py"""
import os, sys, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, auc, DEEP  # noqa: E402
ASSET = os.environ.get("ASSET", "BTC"); f_out = f"{DEEP}/pmflow_{ASSET}.parquet"; DEC = -45
if not os.path.exists(f_out):
    tdir = "/home/user/data/polymarket/trades" if ASSET == "BTC" else "/home/user/data_eth/polymarket/trades"; rows = []
    for f in sorted(glob.glob(f"{tdir}/*.parquet")):
        t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t["rel"] = t.ts - t.window_start; t = t[t.rel < DEC].sort_values(["window_start", "ts"], kind="stable")
        if len(t) == 0: continue
        t["pu"] = np.where(t.outcome_index == 0, t.price, 1 - t.price); t["usd"] = t.price * t["size"]; t["toward_up"] = np.where((t.outcome_index == 0) == (t.side == "BUY"), 1.0, -1.0) * t.usd
        g = t.groupby("window_start"); d = pd.DataFrame({"pm_last": g.pu.last(), "pm_n": g.size(), "pm_usd": g.usd.sum()})
        for lb in (60, 120, 300, 900):
            s = t[t.rel >= DEC - lb].groupby("window_start"); d[f"pm_net_{lb}"] = s.toward_up.sum(); d[f"pm_tot_{lb}"] = s.usd.sum(); d[f"pm_vwap_{lb}"] = s.apply(lambda x: np.average(x.pu, weights=x.usd + 1e-9))
        rows.append(d)
    P = pd.concat(rows); P = P[~P.index.duplicated()]; P.index.name = "T"; P.to_parquet(f_out)
P = pd.read_parquet(f_out); W = load(ASSET, ["m2_-45_close_p", "bp_-60_z_s5"]); P = P.reindex(W.index); y = W.up.values
for lb in (60, 120, 300, 900): P[f"pm_imb_{lb}"] = P[f"pm_net_{lb}"] / (P[f"pm_tot_{lb}"] + 1e-9)
new = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_ph5book.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values if os.path.exists(f"{DEEP}/ourcase_{ASSET}_ph5book.parquet") else np.full(len(W), np.nan)
comp = {"last UP price before T-45": P.pm_last.values - 0.5, "vwap UP price, last 60 s": P.pm_vwap_60.values - 0.5, "vwap UP price, last 300 s": P.pm_vwap_300.values - 0.5}
for lb in (60, 120, 300, 900): comp[f"net taker flow toward UP, last {lb} s ($)"] = P[f"pm_net_{lb}"].values; comp[f"flow imbalance, last {lb} s"] = P[f"pm_imb_{lb}"].values
comp["NEW model"] = new; comp["frozen close head"] = W["m2_-45_close_p"].values - 0.5
rk = lambda v: pd.Series(v).rank(pct=True).values - 0.5
comp["NEW model + last price (rank sum)"] = np.where(np.isfinite(new) & np.isfinite(P.pm_last.values), rk(new) + rk(P.pm_last.values), np.nan)
comp["NEW model + flow imbalance 300 (rank sum)"] = np.where(np.isfinite(new) & np.isfinite(P.pm_imb_300.values), rk(new) + rk(P.pm_imb_300.values), np.nan)
rows = []
for nm, v in comp.items():
    r = {"signal": nm}
    for mo in ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]: r[mo] = auc(y[W.month.values == mo], v[W.month.values == mo])
    r["Jun-Nov"] = auc(y, v); r["coverage"] = np.isfinite(v).mean(); rows.append(r)
print(f"{ASSET}: AUC vs the outcome of Polymarket's own pre-decision information"); print(pd.DataFrame(rows).set_index("signal").round(3).to_string())
ok = np.isfinite(new) & np.isfinite(P.pm_last.values); print("\ncorrelation NEW model score vs last UP price: %.3f;  vs flow imbalance 300 s: %.3f" % (np.corrcoef(new[ok], P.pm_last.values[ok])[0, 1], np.corrcoef(new[ok & np.isfinite(P.pm_imb_300.values)], P.pm_imb_300.values[ok & np.isfinite(P.pm_imb_300.values)])[0, 1]))
print("last UP price before T-45: distribution", P.pm_last.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3).to_dict())
