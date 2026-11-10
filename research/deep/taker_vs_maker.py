"""New rule A: market order (taker: pays the offer + the 0.07 p (1-p) fee, always filled) vs the bot's resting 0.50 bid (assumed filled). Same exit.
  python research/deep/taker_vs_maker.py"""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, DEEP
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; TPX = {"BTC": 0.85, "ETH": 0.80}
for ASSET in ("BTC", "ETH"):
    W = load(ASSET, ["m2_-45_close_p"]); S = sims_cached(ASSET, TPX[ASSET]).reindex(W.index); mon = W.month.values; n = len(W)
    sc = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_ph5.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values
    def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
    sel = np.zeros(n, bool)
    for mo in MONTHS:
        te = (mon == mo) & np.isfinite(sc); sel[te] = np.abs(sc[te]) >= np.quantile(np.abs(sc[before(mo) & np.isfinite(sc)]), 0.85)
    up = sc > 0; pnl = np.where(up, S.pnl_up.values, S.pnl_dn.values); m = sel & np.isfinite(pnl)
    D = pd.DataFrame({"T": W.index.values[m], "date": W.date.values[m], "s": np.where(up[m], 0, 1), "pnl": pnl[m]}).set_index("T")
    tdir = "/home/user/data/polymarket/trades" if ASSET == "BTC" else "/home/user/data_eth/polymarket/trades"; px = {}
    for f in sorted(glob.glob(f"{tdir}/*.parquet")):
        t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "window_start"]); t = t[t.window_start.isin(D.index)]
        if not len(t): continue
        t["rel"] = t.ts - t.window_start; t = t[(t.rel >= -300) & (t.rel < -15) & (t.side == "BUY")]; t = t[t.outcome_index.values == D.s.reindex(t.window_start).values]
        for w, g in t.groupby("window_start"):
            a = g[g.rel >= -45]; b = g[g.rel < -45].sort_values("ts")
            px[w] = (float(a.price.median()), 1) if len(a) else ((float(b.price.iloc[-1]), 0) if len(b) else (np.nan, -1))
    D["ask"] = [px.get(w, (np.nan, -1))[0] for w in D.index]; D["src"] = [px.get(w, (np.nan, -1))[1] for w in D.index]
    ok = D.ask.notna(); E = D[ok].copy(); E["fee"] = 0.07 * E.ask * (1 - E.ask); E["tk"] = E.pnl + 0.50 - E.ask - E.fee
    days = E.groupby("date")[["pnl", "tk"]].sum(); cnt = E.groupby("date").size(); rng = np.random.default_rng(1)
    def ci(col):
        b = [days[col].values[i].sum() / cnt.values[i].sum() for i in (rng.integers(0, len(days), len(days)) for _ in range(2000))]; return np.percentile(b, [2.5, 97.5]) * 100
    nd = 160
    print(f"\n{ASSET}: {len(D)} signals ({len(D)/nd:.0f}/day); market-buy price known for {ok.mean():.0%} ({(E.src==1).mean():.0%} from prints between T-45 and T-15)")
    print(f"  price a market buyer paid: median {E.ask.median():.3f}, mean {E.ask.mean():.3f}; share paying 0.50 or less {(E.ask <= 0.50).mean():.0%}, 0.51 {(E.ask.round(2) == 0.51).mean():.0%}, 0.52+ {(E.ask >= 0.515).mean():.0%}; fee mean {E.fee.mean()*100:.2f}c")
    print(f"  resting bid, every order assumed filled at 0.50: {E.pnl.mean()*100:+.2f} c/share, CI [{ci('pnl')[0]:+.2f}, {ci('pnl')[1]:+.2f}], win {(E.pnl > 0).mean():.1%}; at $5 stake (10 shares): ${E.pnl.sum()*10/nd:+.2f}/day")
    print(f"  market order (offer + fee), always filled:      {E.tk.mean()*100:+.2f} c/share, CI [{ci('tk')[0]:+.2f}, {ci('tk')[1]:+.2f}], win {(E.tk > 0).mean():.1%}; at $5 stake: ${E.tk.sum()*5/E.ask.mean()/nd:+.2f}/day")
    print("  by month, market order c/share:", (E.assign(mo=pd.to_datetime(E.date).dt.strftime('%b')).groupby('mo', sort=False).tk.mean() * 100).round(2).to_dict())
