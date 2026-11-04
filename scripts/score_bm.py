"""Batch out-of-sample forecasts from the FROZEN btc-movement model (artifacts/models; v2 recipe,
trained Feb 2024 -> Feb 2026) for every Polymarket window on the same decision grid as score_m2:
  group A: pseudo window start at T+d (d <= -L), e = 0 ; group B: window start T-L, e = d+L.
Implemented by shifting series['start'] (btcmove.features.compute derives e from ts % 300).
Output: /home/user/data/forecasts/bm_*.parquet
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/btc-movement")
from btcmove import features as fe, series as se, regime as rg
from btcmove.engine import Engine
from btcmove.model import hlen
from pathlib import Path

DATA = Path("/home/user/btc-movement/data")
OUT = "/home/user/data/forecasts"; os.makedirs(OUT, exist_ok=True)
L = 30
PRE = [-120, -90, -75, -60, -50, -45, -40, -35, -30]
POST = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 210, 240, 260]
KEEP = ["basis_dev_60", "basis_dev_300", "basis_dev_3600", "zfret_1", "zfret_5", "zfret_10", "fofi_5", "fofi_30",
        "lew_120", "lrv_300", "lrv_3600", "zret_5", "zret_30"]


def load_seasonal():
    z = np.load(DATA / "seasonal.npz")
    return {k: (int(z[k]) if k == "ts0" else z[k]) for k in z.files}


def score_points(eng, seas, reg_table, ser, Tw, d0, es):
    lo = int(ser["start"])
    s2 = dict(ser); s2["start"] = lo - d0
    nT, nE = len(Tw), len(es)
    T = np.repeat(Tw, nE); e = np.tile(np.array(es, dtype=np.int64), nT)
    idx = (T + d0 + e) - lo
    F = fe.compute(s2, idx, seasonal=seas, with_targets=False)
    if reg_table is not None:
        F.update(rg.lookup(reg_table, T + d0 + e))
    df = pd.DataFrame({k: np.asarray(v) for k, v in F.items()})
    assert np.array_equal(df["e"].to_numpy().astype(int), e), "elapsed mismatch"
    df, S_all = eng.augment(df)
    out = {"T": T, "d": d0 + e, "d0": np.full(len(T), d0), "e": e, "S_bm": df["S"].to_numpy(np.float64), "K_bm": df["K"].to_numpy(np.float64)}
    S = out["S_bm"]; K = out["K_bm"]
    for H, hm in eng.models.items():
        s = S_all[H]
        p_raw, p_st = hm.predict_above(df, s)
        p_rc = eng.recal.apply(H, df["e"].to_numpy(), p_raw, p_struct=p_st) if eng.recal is not None else p_raw
        p_up, _ = hm.predict_cdf(df, np.full(len(df), 1e-12), s=s)
        h = hlen(df, H)
        within = (e + h) <= 300 if H != "T" else np.ones(len(df), bool)
        nm = {"10": "h10", "15": "h15", "30": "h30", "60": "h60", "T": "close"}[H]
        for k, v in {"p": p_rc, "praw": p_raw, "pst": p_st, "pup_now": p_up, "mu": s["mu"], "sd": s["sd"], "mz": s["mz"]}.items():
            out[f"{nm}_{k}"] = np.where(within, v, np.nan).astype(np.float32)
        if H in ("T", "60"):
            n = len(df)
            for dbp in (5, 10, 20):
                Pf = hm.predict_first(df, np.full(n, float(dbp)), s=s)[0]
                out[f"{nm}_fp{dbp}_up"] = np.where(within, Pf[:, 1], np.nan).astype(np.float32)
                out[f"{nm}_fp{dbp}_dn"] = np.where(within, Pf[:, 2], np.nan).astype(np.float32)
            for lv in (-10, -5, 5, 10):
                q = np.log(K / S) + lv * 1e-4
                side = (q > 0).astype(int); dist = np.abs(q)
                pt = hm.predict_touch(df, np.maximum(dist, 1e-9), side, s=s)[0]
                pt = np.where(dist <= 1e-9, 1.0, pt)
                out[f"{nm}_touch{lv:+d}"] = np.where(within, pt, np.nan).astype(np.float32)
    for c in KEEP:
        out[c] = df[c].to_numpy(np.float32)
    return pd.DataFrame(out)


def run_chunk(eng, seas, reg_table, t_lo, t_hi, lag=L, pre=PRE, post=POST):
    lo = t_lo - 2 * 86400 - 3600; hi = t_hi + 400
    ser = se.load_range(lo, hi, data_dir=DATA, futures=True)
    Tw = np.arange((t_lo + 299) // 300 * 300, t_hi, 300)
    parts = [score_points(eng, seas, reg_table, ser, Tw, -lag, [d + lag for d in post])]
    for d in pre:
        if d != -lag:
            parts.append(score_points(eng, seas, reg_table, ser, Tw, d, [0]))
    return pd.concat(parts, ignore_index=True).sort_values(["T", "d"]).reset_index(drop=True)


def consistency_test(eng, seas, reg_table):
    """Unshifted window (d0 = 0) vs the recorded live replay artifacts/replay_demo.jsonl."""
    import datetime as dt
    rep = [json.loads(l) for l in open("/home/user/btc-movement/artifacts/replay_demo.jsonl")]
    T = rep[0]["window_start"]
    lo = T - 3 * 86400; ser = se.load_range(lo, T + 400, data_dir=DATA, futures=True)
    es = [0, 5, 30, 60, 120, 200, 280]
    o = score_points(eng, seas, reg_table, ser, np.array([T]), 0, es)
    worst = 0.0
    for j, e in enumerate(es):
        r = [x for x in rep if x["elapsed_s"] == e][0]
        for H, nm in (("close", "close"), ("30s", "h30"), ("10s", "h10")):
            if H in r["horizons"] and np.isfinite(o[f"{nm}_praw"].values[j]):
                dv = abs(r["horizons"][H]["p_above_reference_raw"] - o[f"{nm}_praw"].values[j])
                worst = max(worst, dv)
                print(f"e={e} {nm}: replay raw={r['horizons'][H]['p_above_reference_raw']:.4f} batch raw={o[f'{nm}_praw'].values[j]:.4f}  K replay={r['reference_price']} batch={o['K_bm'].values[j]}")
    print("max |replay - batch| =", worst)


if __name__ == "__main__":
    import datetime as dt
    eng = Engine.load("/home/user/btc-movement/artifacts/models")
    seas = load_seasonal()
    reg_table = rg.load(str(DATA / "regime.npz")) if (DATA / "regime.npz").exists() else None
    if sys.argv[1] == "test":
        consistency_test(eng, seas, reg_table); sys.exit()
    d0, d1 = dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2])
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    d = d0
    while d <= d1:
        d2 = min(d + dt.timedelta(days=step - 1), d1)
        f = f"{OUT}/bm_{d}_{d2}.parquet"
        if not os.path.exists(f):
            t = time.time()
            t_lo = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.UTC).timestamp())
            t_hi = int(dt.datetime(d2.year, d2.month, d2.day, tzinfo=dt.UTC).timestamp()) + 86400
            o = run_chunk(eng, seas, reg_table, t_lo, t_hi)
            o.to_parquet(f, index=False)
            print(f"{d}..{d2}: {len(o)} rows {time.time()-t:.0f}s", flush=True)
        d = d2 + dt.timedelta(days=1)
    print("done")
