"""Features of the btcmovement2 / eth3 feature engine at the A decision time (T-45, pseudo window starting at T-45, e = 0) for a long
history, plus the labels of OUR target computed from the same Binance timeline:
  y_twap : TWAP60 of spot over [T+240, T+300) >= TWAP60 over [T-60, T)        (Polymarket's rule since Sept 2026 BTC / Aug 7 2026 ETH)
  y_pt   : spot close of second T+299 >= spot open of second T                 (the earlier point rule)
`phase` shifts the 5-minute grid by 0 / 60 / ... seconds (pseudo windows for training augmentation; Polymarket windows are phase 0).
  ASSET=BTC python scripts/dump_feats.py 2025-01-12 2026-11-07 0 [chunk_days]
Output: /home/user/data/deep/feats_<asset>_ph<phase>_<start>_<end>.parquet"""
import os, sys, time, datetime as dt, numpy as np, pandas as pd
ASSET = os.environ.get("ASSET", "BTC")
if ASSET == "BTC":
    sys.path.insert(0, "/home/user/polybtcmodel/scripts"); import score_m2 as S
    MODEL = "/home/user/btcmovement2/models/wf5_20260430"
else:
    sys.path.insert(0, "/home/user/polyethmodel/scripts"); import score_m3 as S
    MODEL = "/home/user/xin10ylop/eth3/models/wf5_20260430"
from btcmodel.timeline import load_timeline
from btcmodel.features import Base, FeatureEngine, W
from btcmodel.regime import regime_features
from btcmodel.predictor import Predictor
OUT = "/home/user/data/deep"; D = int(os.environ.get("DEC", "-45"))      # decision offset from the nominal open (features use seconds < T+D)

def ffill(a):
    a = np.asarray(a, np.float64); m = ~np.isfinite(a) | (a <= 0); idx = np.where(~m, np.arange(len(a)), 0); np.maximum.accumulate(idx, out=idx); return a[idx]

def chunk(tl, pr, c0, c1, phase):
    Tw = np.arange((c0 + W - 1) // W * W, c1, W); Tw = Tw[(Tw + phase + W + 60 <= tl.n) & (Tw >= 10 * 86400)]
    if len(Tw) == 0: return None
    d0 = phase + D; s = d0 % 300; tls = S.ShiftedTL(tl, s)
    a = max(((c0 - s - 2 * 86400) // 3600) * 3600 - s, 0)
    while (tls.t0_epoch + a) % 3600: a += 1
    b = min(tls.n, c1 - s + 900 + phase)
    base = Base(tls, a, b); fe = FeatureEngine(base, pr.season)
    T0_s = Tw + d0 - s; T0 = (T0_s - a).astype(np.int64); e = np.zeros(len(Tw), np.int64)
    X, aux = fe.compute(T0, e); R = regime_features(tls, T0_s.astype(np.int64), e)
    for k in R.columns: X[k] = R[k].values
    X = X.astype(np.float32); Tn = Tw + phase                                    # nominal window start (local index)
    lo, hi = Tn.min() - 400, Tn.max() + 400; c = ffill(tl["s_close"][lo:hi]); o = ffill(tl["s_open"][lo:hi]); i = Tn - lo
    cs = np.concatenate([[0.0], np.cumsum(c)]); mean = lambda u, v: (cs[v] - cs[u]) / (v - u)          # mean of c[u:v]
    tw0, tw1 = mean(i - 60, i), mean(i + 240, i + 300); k15 = mean(i - 60, i - 45); now = c[i + D - 1]
    out = pd.DataFrame({"T": tl.t0_epoch + Tn, "phase": np.int16(phase), "y_twap": (tw1 >= tw0).astype(np.int8), "x_twap": 1e4 * np.log(tw1 / tw0),
                        "y_pt": (c[i + 299] >= o[i]).astype(np.int8), "x_pt": 1e4 * np.log(c[i + 299] / o[i]), "x_dec": 1e4 * np.log(tw1 / now),
                        "f_disp15": ((1e4 * np.log(now / k15)) if D == -45 else np.full(len(i), np.nan)).astype(np.float32), "price": now})      # the part of the start TWAP already known at T-45 (not known earlier)
    return pd.concat([out, X.reset_index(drop=True)], axis=1)

if __name__ == "__main__":
    d0, d1 = dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2]); phase = int(sys.argv[3]); step = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    f = f"{OUT}/feats_{ASSET}_" + ("" if D == -45 else f"d{D}_") + f"ph{phase}_{d0}_{d1}.parquet"
    if os.path.exists(f): print("exists", f); sys.exit()
    tl = load_timeline(); pr = Predictor.load_dir(MODEL); parts = []; d = d0; t0 = time.time()
    while d <= d1:
        d2 = min(d + dt.timedelta(days=step - 1), d1); o = chunk(tl, pr, tl.index_of(d), tl.index_of(d2) + 86400, phase)
        if o is not None: parts.append(o)
        print(f"{ASSET} ph{phase} {d}..{d2}: {0 if o is None else len(o)} rows, {0 if o is None else o.shape[1]} cols, {time.time() - t0:.0f}s", flush=True); d = d2 + dt.timedelta(days=1)
    pd.concat(parts, ignore_index=True).to_parquet(f, index=False); print("wrote", f, flush=True)
