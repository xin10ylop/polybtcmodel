"""Live feature path for the btc-predict core model (spot + perp + ETH per-second bars -> minute panel ->
model matrix -> z forecast). Uses btc-predict's own feature functions, so the live path equals training.

Per-second arrays (dict of numpy arrays, one entry per second from `t0`):
  spot: o h l c v qv n bv     um: o h l c v qv n bv nb lbv lsv mxq     eth: c v bv
"""
import os, sys, json, datetime as dt
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/btc-predict")
from btcpredict.features import seconds as secfeat, panel as P, matrix as F
from btcpredict.config import SEC
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__)); MODELS = os.path.join(HERE, "..", "models")
PRE = 900; HIST_MIN = 2880            # 48 h of minute history for the rolling / EWM minute features
SPOT_COLS = ["o", "h", "l", "c", "v", "qv", "n", "bv"]; UM_COLS = ["o", "h", "l", "c", "v", "qv", "n", "bv", "nb", "lbv", "lsv", "mxq"]; ETH_COLS = ["c", "v", "bv"]


def _ffill(a):
    a = np.asarray(a, float).copy(); m = np.isnan(a)
    if m.all(): return a
    idx = np.where(~m, np.arange(len(a)), 0); np.maximum.accumulate(idx, out=idx); out = a[idx]
    first = np.flatnonzero(~m)[0]; out[:first] = a[first]
    return out


def _prep(d, cols):
    out = {c: np.asarray(d[c], dtype=np.float64) for c in cols}
    out["lp"] = np.log(_ffill(out["c"]))
    return out


class BPLive:
    def __init__(self, keys=("s5", "s1")):
        self.boosters = {k: lgb.Booster(model_file=os.path.join(MODELS, f"bp_{k}_core.txt")) for k in keys}
        self.cols = {k: json.load(open(os.path.join(MODELS, f"bp_{k}_core_features.json"))) for k in keys}

    def features(self, spot, um, eth, t0):
        """spot/um/eth: per-second dicts starting at epoch t0 (t0 % 60 == 0), all same length N (seconds < now).
        Returns the minute model matrix (one row per completed minute bar) and bar open times."""
        N = len(spot["c"]); nmin = (N - PRE) // 60
        ends = PRE + 60 * np.arange(nmin) + 59
        ts = (pd.Timestamp(t0 + PRE, unit="s") + pd.to_timedelta(60 * np.arange(nmin), unit="s")).values
        S, Fm, E = _prep(spot, SPOT_COLS), _prep(um, UM_COLS), _prep(eth, ETH_COLS)
        out = secfeat.features_from_arrays(S, Fm, E, ends, ts, forward=False)
        for c in [c for c in out.columns if c.endswith(("_close", "_open", "_hi", "_lo")) and not c.startswith("e_") and c != "e_close"]:
            out[c] = out[c] + out["ref"]
        out = out.drop(columns=["ref"])
        fc = out.select_dtypes("float64").columns
        keep64 = [c for c in fc if c.endswith(("_close", "_open", "_hi", "_lo")) or c.startswith(("px_", "basis"))]
        out[[c for c in fc if c not in keep64]] = out[[c for c in fc if c not in keep64]].astype("float32")
        out["ts"] = pd.to_datetime(out["ts"])
        df = P.add_minute_features(out)
        X = pd.concat([F.model_matrix(df), F.extra_features(df)], axis=1)
        return X, df

    def forecast(self, spot, um, eth, t0):
        X, df = self.features(spot, um, eth, t0)
        row = X.iloc[-1:]
        return {k: float(b.predict(row[self.cols[k]].values.astype(np.float64))[0]) for k, b in self.boosters.items()}, df["ts"].iloc[-1]


def archive_arrays(day0, day1):
    """Per-second dicts from btc-predict's archive parquet files for days day0..day1 (inclusive)."""
    days = [day0 + dt.timedelta(days=i) for i in range((day1 - day0).days + 1)]
    out = {}
    for venue, cols in (("spot", SPOT_COLS), ("um", UM_COLS), ("eth", ETH_COLS)):
        parts = [pd.read_parquet(SEC / venue / f"{d}.parquet") for d in days]
        df = pd.concat(parts, ignore_index=True)
        out[venue] = {c: df[c].to_numpy(float) for c in cols}
    t0 = int(dt.datetime(day0.year, day0.month, day0.day, tzinfo=dt.UTC).timestamp())
    return out["spot"], out["um"], out["eth"], t0


if __name__ == "__main__":
    # parity: live path on archive seconds vs the training panel (panel_exp.parquet) for the same minutes
    import time
    bp = BPLive(); d1 = dt.date(2026, 11, 3); d0 = d1 - dt.timedelta(days=2)
    spot, um, eth, t0 = archive_arrays(d0, d1)
    # cut to an arbitrary instant to mimic live: forecast instant 2026-11-03 17:23:00
    inst = int(dt.datetime(2026, 11, 3, 17, 23, tzinfo=dt.UTC).timestamp()); n = inst - t0
    sl = lambda d: {k: v[:n] for k, v in d.items()}
    t = time.time(); z, ts_last = bp.forecast(sl(spot), sl(um), sl(eth), t0); print("live-path z", z, "last bar", ts_last, f"{time.time()-t:.1f}s")
    panel = pd.read_parquet("/home/user/btc-predict/data/features/panel_exp.parquet")
    panel = panel[(panel.ts >= "2026-10-30") & (panel.ts <= "2026-11-03 17:30")].reset_index(drop=True)
    X = pd.concat([F.model_matrix(panel), F.extra_features(panel)], axis=1)
    j = np.flatnonzero(panel.ts.values == np.datetime64(ts_last))[0]
    for k, b in bp.boosters.items():
        zt = float(b.predict(X[bp.cols[k]].iloc[[j]].values.astype(np.float64))[0]); print(f"{k}: training-path z {zt:.5f} vs live-path z {z[k]:.5f}")
    # feature-level diff
    Xl, dfl = bp.features(sl(spot), sl(um), sl(eth), t0)
    rowl = Xl.iloc[-1][bp.cols["s5"]].astype(float); rowt = X.iloc[j][bp.cols["s5"]].astype(float)
    diff = (rowl - rowt).abs() / (rowt.abs() + 1e-6); print("features with rel diff > 1%:", diff[diff > 0.01].round(3).to_dict())
