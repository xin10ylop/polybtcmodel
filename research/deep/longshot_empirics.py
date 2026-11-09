"""Late longshots: what happened to every real taker BUY of a token at <= 10 cents after T+120?
Each such print is a purchase somebody actually made (so the price was really available).  Per asset, per price bucket and time bucket:
number of buys, $ spent, share that won, return per $ spent before and after the taker fee (0.07 * p * (1 - p) per share), and the
break-even win rate.  Split by settlement rule (point price vs 60-s TWAP).  Saves the buys to /home/user/data/deep/longshot_<asset>.parquet
  ASSET=BTC python research/deep/longshot_empirics.py"""
import os, sys, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import SRC, DEEP  # noqa: E402
ASSET = os.environ.get("ASSET", "BTC"); f_out = f"{DEEP}/longshot_{ASSET}.parquet"
if not os.path.exists(f_out):
    tdir = "/home/user/data/polymarket/trades" if ASSET == "BTC" else "/home/user/data_eth/polymarket/trades"
    mk = pd.concat([pd.read_parquet(m) for m, _, _ in SRC[ASSET]["markets"]]).drop_duplicates("window_start").set_index("window_start"); mk = mk[mk.outcome_up_price.isin([0.0, 1.0])]; parts = []
    for f in sorted(glob.glob(f"{tdir}/*.parquet")):
        t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t["rel"] = t.ts - t.window_start
        t = t[(t.rel >= 120) & (t.rel < 300) & (t.side == "BUY") & (t.price <= 0.10) & t.window_start.isin(mk.index)]
        if len(t): parts.append(t[["window_start", "rel", "outcome_index", "price", "size"]])
    L = pd.concat(parts, ignore_index=True); up = mk.outcome_up_price.reindex(L.window_start).values == 1.0; L["won"] = np.where(L.outcome_index == 0, up, ~up); L.to_parquet(f_out, index=False)
L = pd.read_parquet(f_out); L["usd"] = L.price * L["size"]; L["fee"] = 0.07 * L.price * (1 - L.price) * L["size"]; L["pay"] = np.where(L.won, L["size"], 0.0)
twap_from = pd.Timestamp(SRC[ASSET]["twap_from"], tz="UTC").timestamp(); L["rule"] = np.where(L.window_start >= twap_from, "TWAP", "point")
L["pb"] = pd.cut(L.price, [0, 0.01, 0.02, 0.03, 0.05, 0.10], labels=["<=1c", "1-2c", "2-3c", "3-5c", "5-10c"]); L["tb"] = pd.cut(L.rel, [120, 180, 240, 270, 285, 300], right=False, labels=["T+120-180", "T+180-240", "T+240-270", "T+270-285", "T+285-300"])
def tab(g):
    return pd.Series({"buys": len(g), "windows": g.window_start.nunique(), "usd_spent": g.usd.sum(), "avg_price_c": 100 * g.usd.sum() / g["size"].sum(), "share_of_$_that_won": g.usd[g.won].sum() / g.usd.sum(),
                      "won_shares_%": 100 * g.pay.sum() / g["size"].sum(), "return_per_$": g.pay.sum() / g.usd.sum() - 1, "return_per_$_after_fee": g.pay.sum() / (g.usd.sum() + g.fee.sum()) - 1})
print(f"{ASSET}: {len(L)} taker buys at <= 10 cents after T+120, ${L.usd.sum():,.0f} spent, in {L.window_start.nunique()} windows")
for rule in ("point", "TWAP"):
    g = L[L.rule == rule]
    if len(g) == 0: continue
    print(f"\n== settlement rule: {rule} ({g.window_start.nunique()} windows) ==  won_shares_% is the win rate per share bought; break-even = avg_price_c")
    print("-- by price --"); print(g.groupby("pb").apply(tab).round(3).to_string())
    print("-- by time (price <= 3c only) --"); print(g[g.price <= 0.03].groupby("tb").apply(tab).round(3).to_string())
    print("-- by time (3c < price <= 10c) --"); print(g[g.price > 0.03].groupby("tb").apply(tab).round(3).to_string())
