"""Compare the degraded-feed forecasts (scripts/score_m2_variant.py, score_m2_synth.py) with the archive forecasts on the
same chunk: mean |dp| and direction flips of the A (d=-45) and B (d=0) close head, plus the A / B decisions that change."""
import sys, numpy as np, pandas as pd
import glob, datetime as dt
chunk = sys.argv[1]   # e.g. 2026-09-10_2026-09-13 (variant chunk); the archive forecasts are loaded by T range whatever their chunking
a, b = chunk.split("_"); t0 = int(dt.datetime.fromisoformat(a + "T00:00:00+00:00").timestamp()); t1 = int(dt.datetime.fromisoformat(b + "T00:00:00+00:00").timestamp()) + 86400
def load(d_):
    fs = [f for f in sorted(glob.glob(f"{d_}/m2_*.parquet"))]
    x = pd.concat([pd.read_parquet(f, columns=["T", "d", "close_p"]) for f in fs], ignore_index=True); return x[(x["T"] >= t0) & (x["T"] < t1)].drop_duplicates(["T", "d"])
real = load("/home/user/data/forecasts")
for name, d_ in (("synthetic (no perp)", "/home/user/data/forecasts_synth"), ("lag1", "/home/user/data/forecasts_lag1"), ("scale", "/home/user/data/forecasts_scale"), ("both", "/home/user/data/forecasts_both")):
    try: v = load(d_)
    except Exception as e: print(f"{name:20s}: missing {e}"); continue
    m = real.merge(v, on=["T", "d"], suffixes=("", "_v"))
    out = []
    for d, lab, thr in ((-45, "A", 0.05), (0, "B", None)):
        x = m[m.d == d]; dp = (x.close_p_v - x.close_p).abs(); flips = (np.sign(x.close_p - 0.5) != np.sign(x.close_p_v - 0.5)).mean()
        s = f"{lab}: mean|dp| {dp.mean():.4f} max {dp.max():.3f} flips {flips*100:.0f}%"
        if thr: s += f", A signals {(np.abs(x.close_p-0.5)>=thr).mean()*100:.0f}% -> {(np.abs(x.close_p_v-0.5)>=thr).mean()*100:.0f}% of windows"
        out.append(s)
    print(f"{name:20s}: " + " | ".join(out))
