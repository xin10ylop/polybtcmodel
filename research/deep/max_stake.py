"""New rule A: how much of a 0.50 bid of a given size would have been filled (prints that hit a 0.50 bid on the signal side while it rests), and the P&L by stake.
Upper bound: assumes we get ALL the volume that traded at <= 0.50 on our side in the resting period (no other bidders ahead of us)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, DEEP
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; TPX = {"BTC": 0.85, "ETH": 0.80}; STAKES = [5, 10, 25, 50, 100, 200, 300, 500, 1000, 2000]
for ASSET in ("BTC", "ETH"):
    W = load(ASSET, ["m2_-45_close_p"]); S = sims_cached(ASSET, TPX[ASSET]).reindex(W.index); mon = W.month.values; n = len(W)
    sc = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_ph5.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values
    def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
    sel = np.zeros(n, bool)
    for mo in MONTHS:
        te = (mon == mo) & np.isfinite(sc); sel[te] = np.abs(sc[te]) >= np.quantile(np.abs(sc[before(mo) & np.isfinite(sc)]), 0.85)
    up = sc > 0; pnl = np.where(up, S.pnl_up.values, S.pnl_dn.values); m = sel & np.isfinite(pnl)
    Z = np.load(f"{DEEP}/maker2_{ASSET}.npz"); pos = pd.Series(np.arange(len(Z["W"])), index=Z["W"]); b = int(np.argmin(np.abs(Z["bids"] - 0.50))); ed = list(Z["edges"])
    T = W.index.values[m]; side = np.where(up[m], 0, 1); p = pnl[m]; ix = pos.reindex(T).values; ok = np.isfinite(ix); T, side, p, ix = T[ok], side[ok], p[ok], ix[ok].astype(int)
    usd = Z["usd"][ix, side, b, :]                       # $ that hit a 0.50 bid on our side, by bucket
    nd = 160
    for nm, bk in (("bid rests T-45..T-30 (live setting)", [ed.index(-45)]), ("bid rests T-45..open", [ed.index(-45), ed.index(-30), ed.index(-15)])):
        cap = usd[:, bk].sum(1); f = cap > 0
        print(f"\n{ASSET} | {nm}: {len(T)} signals ({len(T)/nd:.0f}/day), something traded at <=0.50 on our side in {f.mean():.0%}")
        print(f"   $ available when anything traded: median {np.median(cap[f]):,.0f}, mean {cap[f].mean():,.0f}, 75th pct {np.percentile(cap[f],75):,.0f}, 90th {np.percentile(cap[f],90):,.0f}, 99th {np.percentile(cap[f],99):,.0f}")
        print("   stake   avg $ filled per signal   share of stake filled   $/day     extra $/day vs previous stake   c per filled share")
        prev = 0
        for st in STAKES:
            fl = np.minimum(st, cap); tot = (fl / 0.5 * p).sum() / nd; cps = (fl * p).sum() / max(fl.sum(), 1e-9) * 100
            print(f"   {st:>5}   {fl.mean():>10.1f}                {fl.sum()/(st*len(T)):>6.1%}              {tot:>8.1f}   {tot - prev:>8.1f}                        {cps:+.2f}"); prev = tot
