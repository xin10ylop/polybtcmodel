"""Can the cheap side be picked?  Candidates: at t = 60 .. 260 s after the open, a token whose ask (price a taker really paid in the
next 15 s) is <= 10 cents.  Model probability of that token = the FROZEN close head at that time (trained through April 2026, so every
window here is out of sample; nothing is fitted in this script).  Trades are grouped by price and by the ratio model probability / price.
$1 per trade, taker fee included (0.07 p (1-p) per share), held to settlement.   ASSET=BTC python research/deep/longshot_model.py"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, SRC, DEEP  # noqa: E402
ASSET = os.environ.get("ASSET", "BTC"); TS = [60, 90, 120, 150, 180, 210, 240, 260]
W = load(ASSET, [f"m2_{t}_close_p" for t in TS]); Q = pd.read_parquet(f"{DEEP}/latequotes_{ASSET}.parquet"); rows = []
for t in TS:
    q = Q[Q.t == t].set_index("T").reindex(W.index); pu = W[f"m2_{t}_close_p"].values
    for s, nm in ((0, "up"), (1, "dn")):
        d = pd.DataFrame({"T": W.index.values, "t": t, "price": q[f"ask_{nm}"].values, "usd": q[f"usd_{nm}"].values, "p_model": pu if s == 0 else 1 - pu, "won": (W.up.values == 1) if s == 0 else (W.up.values == 0), "date": W.date.values, "period": W.period.values})
        rows.append(d[np.isfinite(d.price) & (d.price <= 0.10) & np.isfinite(d.p_model)])
C = pd.concat(rows, ignore_index=True); C["fee"] = 0.07 * C.price * (1 - C.price); C["ret"] = np.where(C.won, 1.0, 0.0) / (C.price + C.fee) - 1     # return on $1
twap_from = pd.Timestamp(SRC[ASSET]["twap_from"]); C["rule"] = np.where(C.date >= twap_from, "TWAP", "point"); C["ratio"] = C.p_model / C.price
C["pb"] = pd.cut(C.price, [0, 0.03, 0.05, 0.10], labels=["<=3c", "3-5c", "5-10c"]); C["rb"] = pd.cut(C.ratio, [0, 0.5, 1.0, 1.5, 2.0, 3.0, 1e9], labels=["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", ">3"])
C.to_parquet(f"{DEEP}/longshot_cands_{ASSET}.parquet", index=False)
def tab(g): return pd.Series({"n": len(g), "avg_price_c": 100 * g.price.mean(), "model_%": 100 * g.p_model.mean(), "won_%": 100 * g.won.mean(), "ret_per_$": g.ret.mean(), "t_stat": g.ret.mean() / (g.ret.std() / np.sqrt(len(g)) + 1e-12), "med_usd_avail": g.usd.median()})
print(f"{ASSET}: {len(C)} candidates (token ask <= 10c at t = {TS}); model = frozen close head, out of sample")
for rule in ("point", "TWAP"):
    g = C[C.rule == rule]
    if len(g) == 0: continue
    print(f"\n== {rule} rule: by price and by model probability / price =="); print(g.groupby(["pb", "rb"]).apply(tab).round(2).to_string())
g = C[(C.rule == "TWAP")]
print("\n== TWAP rule, model probability >= 2 x price: by time of purchase =="); print(g[g.ratio >= 2].groupby("t").apply(tab).round(2).to_string())
print("\n== TWAP rule, model probability >= 2 x price: by month =="); print(g[g.ratio >= 2].groupby(g.date.dt.strftime("%Y-%m")).apply(tab).round(2).to_string())
print("\n== TWAP rule, model probability >= 3 x price and price <= 5c: by month =="); print(g[(g.ratio >= 3) & (g.price <= 0.05)].groupby(g.date.dt.strftime("%Y-%m")).apply(tab).round(2).to_string())
