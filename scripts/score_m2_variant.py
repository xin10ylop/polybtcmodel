"""Diagnostic: btcmovement2 forecasts with the Binance perp fields degraded the way a live venue degrades them:
  VARIANT=lag1    perp price/flow fields delayed by 1 s (the OKX feed lags Binance by ~1 s)
  VARIANT=scale   volumes / 2.34, large-trade volumes / 7.5, max trade size / 2.12 (OKX vs Binance, 2026-11-04 20-21 UTC)
  VARIANT=both    both
Output: $M2_OUT/m2_<d0>_<d1>.parquet ; compare with /home/user/data/forecasts (research/feed_diagnostic.py)."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_m2 as S  # noqa: E402
from btcmodel.timeline import load_timeline  # noqa: E402
from btcmodel.predictor import Predictor  # noqa: E402
V = os.environ.get("VARIANT", "lag1"); LAG = 1 if V in ("lag1", "both") else 0; SC = V in ("scale", "both")
PERP = {"p_open", "p_high", "p_low", "p_close", "p_vwap", "p_buy", "p_sell", "p_ntr", "p_nb", "p_qv", "p_maxq", "p_lbv", "p_lsv"}
SCALE = {"p_buy": 1 / 2.34, "p_sell": 1 / 2.34, "p_qv": 1 / 2.34, "p_lbv": 1 / 7.5, "p_lsv": 1 / 7.5, "p_maxq": 1 / 2.12}
class Lazy:
    def __init__(self, tl, k, lo=0, hi=None): self.tl, self.k, self.lo = tl, k, lo; self.hi = tl.n if hi is None else hi
    def __len__(self): return self.hi - self.lo
    def __getitem__(self, s):
        if isinstance(s, slice): a, b, _ = s.indices(len(self)); return Lazy(self.tl, self.k, self.lo + a, self.lo + b)
        return np.asarray(self)[s]
    def __array__(self, dtype=None, copy=None):
        lo, hi = max(self.lo - LAG, 0), max(self.hi - LAG, 0)                     # delayed: value at t is the venue's value at t-LAG
        x = np.asarray(self.tl[self.k][lo:hi]).astype(np.float64)
        if len(x) < self.hi - self.lo: x = np.concatenate([np.full(self.hi - self.lo - len(x), x[0] if len(x) else 0.0), x])
        if SC and self.k in SCALE: x = x * SCALE[self.k]
        return x.astype(dtype) if dtype is not None else x
class VarTL:
    def __init__(self, tl): self.tl = tl; self.n = tl.n; self.t0_epoch = tl.t0_epoch
    def index_of(self, d): return self.tl.index_of(d)
    def __getitem__(self, k): return Lazy(self.tl, k) if k in PERP else self.tl[k]
if __name__ == "__main__":
    from datetime import date, timedelta
    tl = VarTL(load_timeline()); pr = Predictor.load_dir("/home/user/btcmovement2/models/wf5_20260430")
    d0, d1 = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]); step = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    d = d0
    while d <= d1:
        d2 = min(d + timedelta(days=step - 1), d1); f = f"{S.OUT}/m2_{d}_{d2}.parquet"
        if not os.path.exists(f):
            t = time.time(); o = S.score_chunk(tl, pr, tl.index_of(d), tl.index_of(d2) + 86400); o.to_parquet(f, index=False); print(f"{V} {d}..{d2}: {len(o)} rows {time.time()-t:.0f}s", flush=True)
        d = d2 + timedelta(days=1)
    print("done")
