"""btc-predict forecasts for a range of days with the production models (trained through 2026-11-03, the ones the box runs),
through the live code path (polybtc/bp_live.py) on the archive seconds: one row per 5-minute window at the forecast instant
T-60 (the minute boundary A uses), in the bp_minute.parquet layout that scripts/build_master.py joins (ts = T-60; T = ts+60).
  python scripts/score_bp_days.py 2026-11-04 2026-11-06 /home/user/data/forecasts/bp_minute_nov.parquet
Days before the first must be in the archive (two days of bars are used per forecast)."""
import sys, os, time, datetime as dt
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
from bp_live import BPLive, archive_arrays
d0, d1, out = dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2]), sys.argv[3]
bp = BPLive(); rows = []; t_w = time.time()
day = d0
while day <= d1:
    spot, um, eth, t0 = archive_arrays(day - dt.timedelta(days=2), day)
    D = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp())
    for T in range(D + 300, D + 86400 + 1, 300):
        inst = T - 60; n = inst - t0
        sl = lambda d: {k: v[:n] for k, v in d.items()}
        try:
            z, ts_last = bp.forecast(sl(spot), sl(um), sl(eth), t0)
            rows.append(dict(ts=pd.Timestamp(inst, unit="s"), z_s5=z.get("s5", np.nan), z_s1=z.get("s1", np.nan)))
        except Exception as e:
            rows.append(dict(ts=pd.Timestamp(inst, unit="s"), z_s5=np.nan, z_s1=np.nan))
    print(f"{day}: {len(rows)} rows ({time.time()-t_w:.0f}s)", flush=True); day += dt.timedelta(days=1)
pd.DataFrame(rows).to_parquet(out, index=False); print("saved", out)
