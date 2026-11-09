"""Shared loaders for the deep-dive studies (BTC and ETH): one frame per asset over Jun 1 - Nov 7 2026 with the outcome, the
frozen heads and the periods of the protocol (tune Aug 20 - Sep 30, test Oct 1 - Nov 3, unseen Jun / Jul / Nov 4-7)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
PER = {"Jun": ("2026-06-01", "2026-06-30"), "Jul": ("2026-07-01", "2026-07-31"), "Aug1-19": ("2026-08-01", "2026-08-19"), "tune": ("2026-08-20", "2026-09-30"),
       "test": ("2026-10-01", "2026-11-03"), "Nov4-7": ("2026-11-04", "2026-11-07")}
SRC = {"BTC": dict(masters=["/home/user/data/master_june.parquet", "/home/user/data/master_july_L0.parquet", "/home/user/data/master.parquet", "/home/user/data/master_nov.parquet"],
                   markets=[("/home/user/data/polymarket/markets_2026-06-01_2026-07-01.parquet", "2026-06-01", "2026-06-30"), ("/home/user/data/polymarket/markets_2026-07-01_2026-08-01.parquet", "2026-07-01", "2026-07-31"),
                            ("/home/user/data/polymarket/markets.parquet", "2026-08-01", "2026-11-03"), ("/home/user/data/polymarket/markets_nov.parquet", "2026-11-04", "2026-11-07")],
                   pkg="/home/user/polybtcmodel/polybtc", twap_from="2026-09-01"),
       "ETH": dict(masters=["/home/user/data_eth/master.parquet"], markets=[("/home/user/data_eth/polymarket/markets.parquet", "2026-06-01", "2026-11-07")],
                   pkg="/home/user/polyethmodel/polyeth", twap_from="2026-08-07")}

def load(asset, cols):
    parts = []
    for mp in SRC[asset]["masters"]:
        have = set(pq.ParquetFile(mp).schema.names); parts.append(pd.read_parquet(mp, columns=["date", "up"] + [c for c in dict.fromkeys(cols) if c in have]))
    W = pd.concat(parts); W = W[~W.index.duplicated(keep="last")].sort_index(); W["date"] = pd.to_datetime(W.date)
    W = W[(W.date >= "2026-06-01") & (W.date <= "2026-11-07")]
    W["period"] = ""
    for k, (a, b) in PER.items(): W.loc[(W.date >= a) & (W.date <= b), "period"] = k
    W["month"] = W.date.dt.strftime("%b")
    return W

def sims(asset, tp=0.85, t_entry=-45):
    """Per-window pnl per share of a 0.50 fill before T-45 with 'sell tp else settle' (tp=None: settle only), for UP and DOWN."""
    import sys, importlib
    sys.path.insert(0, SRC[asset]["pkg"])
    for m in ("polysim",):
        if m in sys.modules: del sys.modules[m]
    polysim = importlib.import_module("polysim"); out = []
    for markets, a, b in SRC[asset]["markets"]:
        polysim.MARKETS = markets; mk, g = polysim.load_windows(days=[str(d.date()) for d in pd.date_range(a, b)])
        sim = polysim.simulate(mk, g, t_entry=t_entry, entry_mode="fixed", pe=0.50, tps=(tp if tp else 1.01,), sls=(0.0,), tstops=(300,), slip=0.01); idx = np.arange(len(sim["W"]))
        d = {"T": sim["W"]}
        for sd, nm in ((0, "up"), (1, "dn")):
            k = sim["kind"][idx, sd, 0, 0, 0]; hit = (k == 0) & (tp is not None)
            d[f"pnl_{nm}"] = np.where(hit, (tp or 0.0) - 0.5, sim["settle"][idx, sd]); d[f"tp_{nm}"] = hit
        out.append(pd.DataFrame(d).set_index("T"))
    sys.path.remove(SRC[asset]["pkg"])
    s = pd.concat(out); return s[~s.index.duplicated(keep="last")]

def auc(y, s):
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(s); return roc_auc_score(y[m], s[m]) if m.sum() > 200 and len(np.unique(y[m])) == 2 else np.nan

DEEP = "/home/user/data/deep"
def sims_cached(asset, tp):
    import os
    f = f"{DEEP}/sim_{asset}_{tp}.parquet"
    if not os.path.exists(f): sims(asset, tp).to_parquet(f)
    return pd.read_parquet(f)

def capacity(asset):
    """$ printed at 0.49-0.51 (UP-equivalent) before the nominal open, per window: the most a resting 0.50 bid could have taken."""
    import os, glob
    f = f"{DEEP}/cap_{asset}.parquet"
    if not os.path.exists(f):
        tdir = "/home/user/data/polymarket/trades" if asset == "BTC" else "/home/user/data_eth/polymarket/trades"; cap = {}
        for fn in sorted(glob.glob(f"{tdir}/*.parquet")):
            t = pd.read_parquet(fn, columns=["ts", "outcome_index", "price", "size", "window_start"]); pu = np.where(t.outcome_index.values == 0, t.price.values, 1 - t.price.values)
            m = (t.ts.values < t.window_start.values) & (pu >= 0.49) & (pu <= 0.51); s = pd.Series(t.price.values[m] * t["size"].values[m]).groupby(t.window_start.values[m]).sum()
            for k, v in s.items(): cap[int(k)] = cap.get(int(k), 0.0) + float(v)
        pd.Series(cap, name="cap").rename_axis("T").to_frame().to_parquet(f)
    return pd.read_parquet(f).cap

def book(asset, T, level="1.0"):
    import glob
    d = "/home/user/data/extra/bookdepth" if asset == "BTC" else "/home/user/data_eth/extra/bookdepth"
    bd = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{d}/*.parquet"))]).sort_index(); bd = bd[~bd.index.duplicated()]      # early-2025 files have no 0.2 % band
    s = ((bd[f"n-{level}"] - bd[f"n+{level}"]) / (bd[f"n-{level}"] + bd[f"n+{level}"])).astype(float); idx = s.index.values.astype(np.int64); when = np.asarray(T, np.int64) - 47
    j = np.searchsorted(idx, when, side="right") - 1; ok = j >= 0; val = np.where(ok, s.values[np.maximum(j, 0)], np.nan); age = np.where(ok, when - idx[np.maximum(j, 0)], np.inf)
    return np.where(age <= 120, val, np.nan)

def sims_mode(asset, tp, tag, **kw):
    """Like sims() but with another entry mode of polysim.simulate (kw: entry_mode, t_cancel, bid, entry_from_trades, entry_taker, entry_fee ...).
    pnl = tp - entry if the sell limit is hit, else settlement - entry; NaN when the entry did not fill.  Cached under `tag`."""
    import os, sys, importlib
    f = f"{DEEP}/sim_{asset}_{tp}_{tag}.parquet"
    if os.path.exists(f): return pd.read_parquet(f)
    sys.path.insert(0, SRC[asset]["pkg"])
    if "polysim" in sys.modules: del sys.modules["polysim"]
    polysim = importlib.import_module("polysim"); out = []
    for markets, a, b in SRC[asset]["markets"]:
        polysim.MARKETS = markets; mk, g = polysim.load_windows(days=[str(d.date()) for d in pd.date_range(a, b)])
        sim = polysim.simulate(mk, g, t_entry=-45, pe=0.50, tps=(tp,), sls=(0.0,), tstops=(300,), slip=0.01, **kw); idx = np.arange(len(sim["W"])); d = {"T": sim["W"]}
        for sd, nm in ((0, "up"), (1, "dn")):
            k = sim["kind"][idx, sd, 0, 0, 0]; e = sim["entry_px"][idx, sd]; d[f"pnl_{nm}"] = np.where(np.isfinite(e), np.where(k == 0, tp - e, sim["settle"][idx, sd]), np.nan); d[f"entry_{nm}"] = e; d[f"pre_{nm}"] = sim["pre_last"][idx, sd]
        out.append(pd.DataFrame(d).set_index("T"))
    sys.path.remove(SRC[asset]["pkg"]); s = pd.concat(out); s = s[~s.index.duplicated(keep="last")]; s.to_parquet(f); return s
