"""btcmovement2 forecasts computed the way the LIVE trader would see the data if no perpetual feed were available:
the perp fields are synthesized from spot exactly as paper_trader.LiveTimeline.finalize_to does (price = spot x basis,
buy/sell volume from the spot taker-buy split, trade count from spot, no large-trade / max-size information).
This measures how much of the backtested edge survives the live data path (research question raised by the
parity audit of 2026-11-04). Output: $M2_OUT (default /home/user/data/forecasts_synth)/m2_<d0>_<d1>.parquet
  M2_OUT=/home/user/data/forecasts_synth python scripts/score_m2_synth.py 2026-08-01 2026-11-03 4"""
import os, sys, time, numpy as np
os.environ.setdefault("M2_OUT", "/home/user/data/forecasts_synth")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_m2 as S  # noqa: E402
from btcmodel.timeline import load_timeline  # noqa: E402
from btcmodel.predictor import Predictor  # noqa: E402
BASIS = float(os.environ.get("SYNTH_BASIS", "0.9996"))
PERP = {"p_open", "p_high", "p_low", "p_close", "p_vwap", "p_buy", "p_sell", "p_ntr", "p_nb", "p_qv", "p_maxq", "p_lbv", "p_lsv"}

class Lazy:
    """A column that is computed from spot on demand, supporting slicing like a memmap."""
    def __init__(self, tl, k, lo=0, hi=None):
        self.tl, self.k, self.lo = tl, k, lo; self.hi = tl.n if hi is None else hi
    def __len__(self): return self.hi - self.lo
    def __getitem__(self, s):
        if isinstance(s, slice):
            a, b, _ = s.indices(len(self)); return Lazy(self.tl, self.k, self.lo + a, self.lo + b)
        return np.asarray(self)[s]
    def __array__(self, dtype=None, copy=None):
        lo, hi = self.lo, self.hi; k = self.k; t = self.tl
        if k in ("p_open", "p_high", "p_low", "p_close", "p_vwap"): x = np.asarray(t["s_close"][lo:hi], dtype=np.float64) * BASIS
        elif k == "p_buy": x = np.asarray(t["s_tb_base"][lo:hi], dtype=np.float32)
        elif k == "p_sell": x = np.maximum(np.asarray(t["s_volume"][lo:hi], dtype=np.float32) - np.asarray(t["s_tb_base"][lo:hi], dtype=np.float32), 0)
        elif k == "p_ntr": x = np.asarray(t["s_count"][lo:hi])
        elif k == "p_nb": x = np.asarray(t["s_count"][lo:hi], dtype=np.float32) / 2
        elif k == "p_qv": x = np.asarray(t["s_close"][lo:hi], dtype=np.float64) * BASIS * np.asarray(t["s_volume"][lo:hi], dtype=np.float64)
        else: x = np.zeros(hi - lo, dtype=np.float32)          # p_maxq, p_lbv, p_lsv: unknown without a perp feed
        return x.astype(dtype) if dtype is not None else x

class SynthTL:
    def __init__(self, tl): self.tl = tl; self.n = tl.n; self.t0_epoch = tl.t0_epoch
    def index_of(self, d): return self.tl.index_of(d)
    def __getitem__(self, k): return Lazy(self.tl, k) if k in PERP else self.tl[k]

if __name__ == "__main__":
    from datetime import date, timedelta
    tl = SynthTL(load_timeline()); pr = Predictor.load_dir("/home/user/btcmovement2/models/wf5_20260430")
    d0, d1 = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]); step = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    d = d0
    while d <= d1:
        d2 = min(d + timedelta(days=step - 1), d1); f = f"{S.OUT}/m2_{d}_{d2}.parquet"
        if not os.path.exists(f):
            t = time.time(); o = S.score_chunk(tl, pr, tl.index_of(d), tl.index_of(d2) + 86400); o.to_parquet(f, index=False)
            print(f"{d}..{d2}: {len(o)} rows {time.time()-t:.0f}s", flush=True)
        d = d2 + timedelta(days=1)
    print("done")
