"""Build btc-movement monthly npz (spot_1s, fut_1s) from the shared raw zips, then seasonal.npz and
regime.npz, using btc-movement's own build/features/regime code."""
import sys, glob, numpy as np, datetime as dt
from pathlib import Path
sys.path.insert(0, "/home/user/btc-movement")
from btcmove import build, features as fe, series as se, regime as rg
from btcmove.build import month_bounds
RAW = Path("/home/user/data/raw"); DATA = Path("/home/user/btc-movement/data")
months = sys.argv[1:]
for period in months:
    for kind, sub, fn in (("spot_1s", "spot1s", build.build_spot_month), ("fut_1s", "umagg", build.build_futures_month)):
        out = DATA / kind / f"{period}.npz"
        if out.exists():
            continue
        zips = sorted(RAW.joinpath(sub).glob(f"{period}-*.zip"))
        if not zips:
            print(period, kind, "no zips"); continue
        d = fn(zips, period)
        d = build.ffill_prices(d, None)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **d)
        print(period, kind, len(zips), "files", f"present={d['present'].mean():.5f}", flush=True)
periods = se.available_periods("spot_1s", DATA)
first_ts = month_bounds(periods[0])[0]
last = np.load(DATA / "spot_1s" / f"{periods[-1]}.npz")
data_end = month_bounds(periods[-1])[0] + int(np.where(last["present"])[0].max()) + 1
full = se.load_range(first_ts, data_end, data_dir=DATA, futures=False)
seas = fe.build_seasonal(full["c"], first_ts)
np.savez(DATA / "seasonal.npz", **{k: v for k, v in seas.items()})
del full
daily = rg.build_daily(periods, data_dir=str(DATA))
np.savez(DATA / "regime_daily.npz", **daily)
rg.save(rg.regime_table(daily), str(DATA / "regime.npz"))
print("seasonal + regime built; data_end", data_end)
