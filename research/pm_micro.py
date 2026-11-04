"""Polymarket microstructure on the days downloaded so far: pre-open prices, adverse selection by
fill time, and the share price's response to Binance displacement from the lagged reference."""
import numpy as np, pandas as pd, datetime as dt, os, glob
mk = pd.read_parquet("/home/user/data/polymarket/markets.parquet").set_index("window_start")
files = sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet"))
tr = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
tr = tr[tr.window_start.isin(mk.index)]
tr["up"] = mk.loc[tr.window_start.values, "outcome_up_price"].values
tr = tr[tr.up.isin([0.0, 1.0])]
tr["rel"] = tr.ts - tr.window_start
tr["p_up"] = np.where(tr.outcome_index == 0, tr.price, 1 - tr.price)          # UP-equivalent price
# taker buys the token outcome_index at price; taker 'wins' if that token resolves 1
tr["taker_token_win"] = np.where(tr.outcome_index == 0, tr.up, 1 - tr.up)
tr["taker_buy_side_price"] = tr.price
print("days:", len(files), "trades:", len(tr), "markets:", tr.window_start.nunique())
b = pd.cut(tr.rel, [-1e9, -300, -120, -60, -45, -35, -25, -15, -5, 0, 5, 15, 30, 60, 120, 180, 240, 270, 300, 1e9])
g = tr.groupby(b, observed=True)
print("\n== by time relative to nominal open: n trades, mean |p_up-0.5|, taker win rate (BUY side), taker avg price ==")
buys = tr[tr.side == "BUY"]
gb = buys.groupby(pd.cut(buys.rel, b.cat.categories), observed=True)
out = pd.DataFrame({"n": g.size(), "mean_absdev": g.apply(lambda x: (x.p_up - 0.5).abs().mean()),
                    "buy_n": gb.size(), "buy_win": gb.taker_token_win.mean(), "buy_px": gb.price.mean(),
                    "buy_pnl_to_settle": gb.apply(lambda x: (x.taker_token_win - x.price).mean()),
                    "vol_usd": g.apply(lambda x: (x.price * x.size).sum())})
print(out.round(4).to_string())
print("\n== pre-open fills (rel < -45): distribution of UP-equivalent price ==")
pre = tr[tr.rel < -45]
print(pre.p_up.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).round(3))
print("share of pre-open (rel<-45) volume within [0.48,0.53]:", ((pre.p_up >= 0.48) & (pre.p_up <= 0.53)).mean().round(3))
print("\n== what fraction of markets have any trade before T-45? and before T-60? ==")
print((tr[tr.rel < -45].groupby("window_start").size().reindex(tr.window_start.unique()).fillna(0) > 0).mean(),
      (tr[tr.rel < -60].groupby("window_start").size().reindex(tr.window_start.unique()).fillna(0) > 0).mean())
print("\n== maker-side perspective: if I rest a BUY of the eventual winner side vs loser; use taker SELLs hitting bids ==")
sells = tr[tr.side == "SELL"]
gs = sells.groupby(pd.cut(sells.rel, b.cat.categories), observed=True)
print(pd.DataFrame({"sell_n": gs.size(), "maker_buyer_win": gs.taker_token_win.mean(), "sell_px": gs.price.mean()}).round(4).to_string())
# last pre-reference price vs outcome
last_pre = tr[tr.rel < -45].sort_values("ts").groupby("window_start").p_up.last()
w = mk.loc[last_pre.index]
print("\nlast UP-equiv price before T-45: mean", last_pre.mean().round(4), " corr with outcome:",
      np.corrcoef(last_pre.values, w.outcome_up_price.values)[0, 1].round(4), " P(up | last<0.49)=", w.outcome_up_price[last_pre < 0.49].mean().round(4),
      " P(up | last>0.51)=", w.outcome_up_price[last_pre > 0.51].mean().round(4))
