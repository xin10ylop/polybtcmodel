"""Cross-asset features at the A decision time for every minute boundary T' (= every grid phase used by dump_feats.py): simple returns,
flow imbalances, basis and realised variance of BOTH assets from their 1-s Binance timelines, last complete second = T'-46.
The BTC model reads the eth_* columns, the ETH model the btc_* columns (plus relative returns built in the model script).
  python scripts/dump_cross.py <eth_data_dir> 2026-05-25 2026-11-07
Output: /home/user/data/deep/xfeat_<start>_<end>.parquet"""
import sys, os, datetime as dt, numpy as np, pandas as pd
OUT = "/home/user/data/deep"; RK = (5, 10, 30, 60, 120, 300, 900, 3600); IK = (10, 60, 300)

class TL:
    def __init__(self, d):
        a, b, n = open(f"{d}/timeline/meta.txt").read().split()[:3]; self.d = d; self.n = int(n)
        self.t0 = int(dt.datetime.fromisoformat(a + "T00:00:00+00:00").timestamp())
    def get(self, k, lo, hi): return np.asarray(np.load(f"{self.d}/timeline/{k}.npy", mmap_mode="r")[lo:hi], dtype=np.float64)

def ffill(a):
    m = ~np.isfinite(a) | (a <= 0); idx = np.where(~m, np.arange(len(a)), 0); np.maximum.accumulate(idx, out=idx); return a[idx]

def feats(tl, Tq, pre):
    """Tq: epoch seconds of T' (sorted). Returns dict of arrays."""
    lo = Tq.min() - tl.t0 - 46 - 3700; hi = Tq.max() - tl.t0 - 46 + 2; assert lo >= 0 and hi <= tl.n, (lo, hi, tl.n)
    i = Tq - tl.t0 - 46 - lo; c = ffill(tl.get("s_close", lo, hi)); lc = np.log(c); pc = ffill(tl.get("p_close", lo, hi))
    cs = lambda a: np.concatenate([[0.0], np.cumsum(np.nan_to_num(a))]); out = {}
    for k in RK: out[f"{pre}_r_{k}"] = 1e4 * (lc[i] - lc[i - k])
    v, tb = cs(tl.get("s_volume", lo, hi)), cs(tl.get("s_tb_base", lo, hi)); pb, ps = cs(tl.get("p_buy", lo, hi)), cs(tl.get("p_sell", lo, hi))
    for k in IK:
        vv = v[i + 1] - v[i + 1 - k]; out[f"{pre}_simb_{k}"] = (2 * (tb[i + 1] - tb[i + 1 - k]) - vv) / (vv + 1e-12)
        b, s = pb[i + 1] - pb[i + 1 - k], ps[i + 1] - ps[i + 1 - k]; out[f"{pre}_pimb_{k}"] = (b - s) / (b + s + 1e-12)
    basis = np.log(pc) - lc; cb = cs(basis); out[f"{pre}_bdev_300"] = 1e4 * (basis[i] - (cb[i + 1] - cb[i + 1 - 300]) / 300); out[f"{pre}_bdev_60"] = 1e4 * (basis[i] - (cb[i + 1] - cb[i + 1 - 60]) / 60)
    r2 = cs(np.diff(lc, prepend=lc[0]) ** 2); out[f"{pre}_lrv_300"] = np.log((r2[i + 1] - r2[i + 1 - 300]) * 1e8 + 1e-3); out[f"{pre}_lrv_3600"] = np.log((r2[i + 1] - r2[i + 1 - 3600]) * 1e8 / 12 + 1e-3)
    return out

if __name__ == "__main__":
    eth = TL(sys.argv[1]); btc = TL("/home/user/data"); d0, d1 = dt.date.fromisoformat(sys.argv[2]), dt.date.fromisoformat(sys.argv[3]); parts = []; d = d0
    while d <= d1:
        d2 = min(d + dt.timedelta(days=13), d1); a = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.UTC).timestamp()); b = int(dt.datetime(d2.year, d2.month, d2.day, tzinfo=dt.UTC).timestamp()) + 86400
        Tq = np.arange(a, b, 60, dtype=np.int64); o = {"T": Tq}; o.update(feats(eth, Tq, "eth")); o.update(feats(btc, Tq, "btc"))
        parts.append(pd.DataFrame(o).astype({k: np.float32 for k in o if k != "T"})); print(d, d2, len(Tq), flush=True); d = d2 + dt.timedelta(days=1)
    f = f"{OUT}/xfeat_{d0}_{d1}.parquet"; pd.concat(parts, ignore_index=True).to_parquet(f, index=False); print("wrote", f)
