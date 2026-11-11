"""Older training history for the purpose-built models, without keeping the heavy per-second timeline.
For each calendar month: download the Binance archives for the month plus 11 days of lookback into a SEPARATE data root, build a small timeline,
dump the features and labels (scripts/dump_feats.py, same code as the main history), then delete the timeline and the days no longer needed.
  DEC=-120 python scripts/fetch_hist_feats.py 2022-01 2024-12         ->  /home/user/data/deep/feats_BTC_d-120_ph{0,120,240}_<month start>_<month end>.parquet
(the last days before the main history, 2025-01-01..2025-01-11, are fetched with the end month 2025-01)"""
import os, sys, glob, shutil, subprocess, time, datetime as dt
ROOT = "/home/user/data_hist"; os.environ["BTC_DATA_DIR"] = ROOT; sys.path.insert(0, "/home/user/btcmovement2")
from btcmodel import data as m2, timeline as tlm
from concurrent.futures import ThreadPoolExecutor
DEC = os.environ.get("DEC", "-120"); PHASES = os.environ.get("PHASES", "0,120,240").split(","); MAIN_START = dt.date(2025, 1, 12); LOOK = 11
def month_iter(a, b):
    y, m = map(int, a.split("-")); y1, m1 = map(int, b.split("-"))
    while (y, m) <= (y1, m1): yield dt.date(y, m, 1); y, m = (y + 1, 1) if m == 12 else (y, m + 1)
def day(d):
    for k in range(4):
        try: return m2.build_day(d)
        except Exception as e: err = e; time.sleep(3 * (k + 1))
    print(f"   MISSING {d}: {err}", flush=True)
for m0 in month_iter(sys.argv[1], sys.argv[2]):
    m1 = min((m0.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1), MAIN_START - dt.timedelta(days=1)); t0 = time.time()
    outs = [f"/home/user/data/deep/feats_BTC_d{DEC}_ph{p}_{m0}_{m1}.parquet" for p in PHASES]
    if all(os.path.exists(f) for f in outs): print("have", m0, flush=True); continue
    lo = m0 - dt.timedelta(days=LOOK); days = [lo + dt.timedelta(days=i) for i in range((m1 - lo).days + 1)]
    for f in glob.glob(f"{ROOT}/days/*.npz"):
        if dt.date.fromisoformat(os.path.basename(f)[:10]) < lo: os.remove(f)
    with ThreadPoolExecutor(4) as ex: list(ex.map(day, days))
    shutil.rmtree(f"{ROOT}/timeline", ignore_errors=True); tlm.build_timeline(lo, m1)
    ps = [subprocess.Popen([sys.executable, "-W", "ignore", "/home/user/polybtcmodel/scripts/dump_feats.py", str(m0), str(m1), p, "40"], env=dict(os.environ, ASSET="BTC", DEC=DEC, HIST_DATA_DIR=ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) for p in PHASES]
    for p in ps:
        e = p.communicate()[1]
        if p.returncode: print("   DUMP FAILED", e.decode()[-400:], flush=True)
    shutil.rmtree(f"{ROOT}/timeline", ignore_errors=True); print(f"{m0}..{m1}: {sum(os.path.exists(f) for f in outs)}/{len(outs)} files, {time.time() - t0:.0f}s", flush=True)
