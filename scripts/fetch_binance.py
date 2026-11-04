"""Shared Binance archive downloader: one download per (symbol, day), written in every format the
three repos expect.  Keeps the originals intact (only calls their public functions).

  /home/user/data/raw/{spot1s,eth1s,umagg}/<day>.zip      raw archives (for btc-movement monthly builder)
  /home/user/data/days/<day>.npz                           btcmovement2 day files (its own compact_* code)
  /home/user/btc-predict/data/sec/{spot,um,eth}/<day>.parquet   btc-predict per-second bars
"""
import sys, os, io, time, datetime as dt, hashlib, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, requests

sys.path.insert(0, "/home/user/btcmovement2")
sys.path.insert(0, "/home/user/btc-predict")
os.environ.setdefault("BTC_DATA_DIR", "/home/user/data")
from btcmodel import data as m2data            # btcmovement2
from btcpredict.data import binance as bpb     # btc-predict
from btcpredict.data.binance import KLINE_COLS, _read_zip_csv, _to_ms, _day_start_ms, trades_to_1s, DAY_SECONDS

RAW = "/home/user/data/raw"
DAYS = "/home/user/data/days"
BP_SEC = "/home/user/btc-predict/data/sec"
BASE = "https://data.binance.vision/data"
S = requests.Session()
for _d in [f"{RAW}/spot1s", f"{RAW}/eth1s", f"{RAW}/umagg", DAYS, f"{BP_SEC}/spot", f"{BP_SEC}/um", f"{BP_SEC}/eth"]:
    os.makedirs(_d, exist_ok=True)

def get(url, retries=6):
    last = None
    for k in range(retries):
        try:
            r = S.get(url, timeout=180)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            blob = r.content
            c = S.get(url + ".CHECKSUM", timeout=60)
            if c.status_code == 200 and c.text.strip().split()[0] != hashlib.sha256(blob).hexdigest():
                raise IOError("checksum mismatch")
            return blob
        except Exception as e:
            last = e; time.sleep(2 ** k)
    raise RuntimeError(f"{url}: {last}")

def raw_path(kind, d):
    return f"{RAW}/{kind}/{d}.zip"

def load_raw(kind, url, d):
    p = raw_path(kind, d)
    if os.path.exists(p):
        return open(p, "rb").read()
    blob = get(url)
    if blob is None:
        return None
    tmp = p + ".part"
    open(tmp, "wb").write(blob); os.replace(tmp, p)
    return blob

def bp_spot_from_blob(blob, day):
    df = _read_zip_csv(blob, KLINE_COLS)
    ts = _to_ms(df["open_time"].to_numpy())
    sec = (ts - _day_start_ms(day)) // 1000
    ok = (sec >= 0) & (sec < DAY_SECONDS); sec = sec[ok]
    out = {}; n = DAY_SECONDS
    for col, name in [("open", "o"), ("high", "h"), ("low", "l"), ("close", "c")]:
        a = np.full(n, np.nan); a[sec] = df[col].to_numpy(dtype=float)[ok]; out[name] = a
    for col, name in [("volume", "v"), ("quote_volume", "qv"), ("count", "n"), ("taker_buy_volume", "bv")]:
        a = np.zeros(n); a[sec] = df[col].to_numpy(dtype=float)[ok]; out[name] = a
    nov = out["n"] == 0
    for k in ("o", "h", "l", "c"):
        out[k][nov] = np.nan
    return pd.DataFrame(out).astype("float64")

def bp_um_from_blob(blob, day):
    df = _read_zip_csv(blob, bpb.AGG_COLS)
    return trades_to_1s(_to_ms(df["transact_time"].to_numpy()) - _day_start_ms(day),
                        df["price"].to_numpy(dtype=float), df["quantity"].to_numpy(dtype=float),
                        df["is_buyer_maker"].to_numpy().astype(bool), n=DAY_SECONDS)

def write_parquet(df, path):
    tmp = path + ".tmp"; df.to_parquet(tmp, index=False); os.replace(tmp, path)

def do_day(d):
    res = []
    spot_url = f"{BASE}/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-{d}.zip"
    eth_url = f"{BASE}/spot/daily/klines/ETHUSDT/1s/ETHUSDT-1s-{d}.zip"
    um_url = f"{BASE}/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{d}.zip"
    need_m2 = not os.path.exists(f"{DAYS}/{d}.npz")
    need_bp_spot = not os.path.exists(f"{BP_SEC}/spot/{d}.parquet")
    need_bp_um = not os.path.exists(f"{BP_SEC}/um/{d}.parquet")
    need_bp_eth = not os.path.exists(f"{BP_SEC}/eth/{d}.parquet")
    spot = load_raw("spot1s", spot_url, d) if (need_m2 or need_bp_spot) else None
    if spot is None and (need_m2 or need_bp_spot):
        return f"{d}: spot missing"
    um = load_raw("umagg", um_url, d) if (need_m2 or need_bp_um) else None
    if need_m2:
        s = m2data.compact_spot_1s(spot, d)
        p = m2data.compact_perp_agg(um, d) if um is not None else {}
        path = f"{DAYS}/{d}.npz"
        np.savez_compressed(path + ".tmp.npz", **s, **p); os.replace(path + ".tmp.npz", path); res.append("m2")
    if need_bp_spot:
        write_parquet(bp_spot_from_blob(spot, d), f"{BP_SEC}/spot/{d}.parquet"); res.append("bps")
    if need_bp_um and um is not None:
        write_parquet(bp_um_from_blob(um, d), f"{BP_SEC}/um/{d}.parquet"); res.append("bpu")
    if need_bp_eth:
        eth = load_raw("eth1s", eth_url, d)
        if eth is not None:
            write_parquet(bp_spot_from_blob(eth, d), f"{BP_SEC}/eth/{d}.parquet"); res.append("bpe")
    return f"{d}: {','.join(res) or 'cached'}"

if __name__ == "__main__":
    a, b = dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    days = [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]
    # newest first so the Polymarket period is ready early
    days = days[::-1]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(do_day, d): d for d in days}
        for i, f in enumerate(as_completed(futs)):
            try:
                print(f.result(), f"[{i+1}/{len(days)} {time.time()-t0:.0f}s]", flush=True)
            except Exception as e:
                print(futs[f], "ERROR", repr(e), flush=True)
    print("done", flush=True)
