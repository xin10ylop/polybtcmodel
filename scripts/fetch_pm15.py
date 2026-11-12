"""Polymarket 15-minute Up/Down markets, history for the fill study: the market list with results for a date range, and the PRE-OPEN prints (T-200 .. T) of the
windows the T-120 model would signal on (|score| >= MIN_SCORE; the rest of the tape is not needed for a bid that rests before the open).
  ASSET=BTC python scripts/fetch_pm15.py 2026-06-01 2026-10-26   ->  /home/user/data/pm15/<asset>/markets_hist.parquet, /home/user/data/pm15/<asset>/pre_hist.parquet"""
import os, sys, json, datetime as dt, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/user/polybtcmodel/scripts"); import fetch_polymarket as fp
A = os.environ.get("ASSET", "BTC").lower(); SERIES = {"btc": 10192, "eth": 10191}[A]; OUT = f"/home/user/data/pm15/{A}"; MIN = float(os.environ.get("MIN_SCORE", "0.06"))
t0 = int(pd.Timestamp(sys.argv[1]).timestamp()) - 86400 - 3600; t1 = int(pd.Timestamp(sys.argv[2]).timestamp()); mf = f"{OUT}/markets_hist.parquet"
def chunk(t):
    a = dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"); b = dt.datetime.fromtimestamp(t + 6 * 3600 - 1, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"); rows = []
    for off in range(0, 1000, 100):
        d = fp.get_json(f"{fp.GAMMA}/events", dict(series_id=SERIES, limit=100, offset=off, start_date_min=a, start_date_max=b))
        for e in d:
            if not e.get("slug", "").startswith(f"{A}-updown-15m-"): continue
            m = e["markets"][0]; rows.append(dict(window_start=int(e["slug"].rsplit("-", 1)[1]), condition_id=m["conditionId"], outcome_up_price=float(json.loads(m["outcomePrices"])[0]) if m.get("outcomePrices") else np.nan, volume=float(m.get("volumeNum") or 0), uma_status=m.get("umaResolutionStatus")))
        if len(d) < 100: break
    return rows
if os.path.exists(mf): mk = pd.read_parquet(mf)
else:
    with ThreadPoolExecutor(4) as ex: mk = pd.DataFrame([r for rows in ex.map(chunk, range(t0, t1, 6 * 3600)) for r in rows]).drop_duplicates("window_start").sort_values("window_start").reset_index(drop=True)
    mk.to_parquet(mf, index=False)
print(len(mk), "markets", pd.to_datetime(mk.window_start.iloc[[0, -1]], unit="s").tolist(), flush=True)
s = pd.read_parquet("/home/user/data/deep/ourcase_BTC_d-120_ph3.parquet").set_index("T")["lgb_reg300"]; want = mk[mk.window_start.isin(s.index[s.abs() >= MIN]) & (mk.uma_status == "resolved")]; print(len(want), "windows with a signal", flush=True)
ws = dict(zip(want.condition_id, want.window_start)); ids = list(ws); done = [0]
def tr(batch):
    rows = [(c, ts, side, o, p, sz) for (c, ts, side, o, p, sz, _) in fp.fetch_trades(batch) if -200 <= ts - ws[c] <= 0]; done[0] += len(batch)
    if done[0] % 400 < len(batch): print(f"  {done[0]}/{len(ids)} markets", flush=True)
    return rows
with ThreadPoolExecutor(4) as ex: rows = [r for b in ex.map(tr, [ids[i:i + 10] for i in range(0, len(ids), 10)]) for r in b]
d = pd.DataFrame(rows, columns=["condition_id", "ts", "side", "outcome_index", "price", "size"]); d["window_start"] = d.condition_id.map(ws); d.drop(columns="condition_id").to_parquet(f"{OUT}/pre_hist.parquet", index=False); print("wrote", len(d), "pre-open prints for", d.window_start.nunique(), "windows")
