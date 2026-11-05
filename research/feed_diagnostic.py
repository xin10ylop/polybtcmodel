"""Compare the degraded-feed forecasts (scripts/score_m2_variant.py, score_m2_synth.py) with the archive forecasts on the
same chunk: mean |dp| and direction flips of the A (d=-45) and B (d=0) close head, plus the A / B decisions that change."""
import sys, numpy as np, pandas as pd
chunk = sys.argv[1]   # e.g. 2026-09-10_2026-09-13
real = pd.read_parquet(f"/home/user/data/forecasts/m2_{chunk}.parquet", columns=["T", "d", "close_p"])
for name, d_ in (("synthetic (no perp)", "/home/user/data/forecasts_synth"), ("lag1", "/home/user/data/forecasts_lag1"), ("scale", "/home/user/data/forecasts_scale"), ("both", "/home/user/data/forecasts_both")):
    try: v = pd.read_parquet(f"{d_}/m2_{chunk}.parquet", columns=["T", "d", "close_p"])
    except Exception: print(f"{name:20s}: missing"); continue
    m = real.merge(v, on=["T", "d"], suffixes=("", "_v"))
    out = []
    for d, lab, thr in ((-45, "A", 0.05), (0, "B", None)):
        x = m[m.d == d]; dp = (x.close_p_v - x.close_p).abs(); flips = (np.sign(x.close_p - 0.5) != np.sign(x.close_p_v - 0.5)).mean()
        s = f"{lab}: mean|dp| {dp.mean():.4f} max {dp.max():.3f} flips {flips*100:.0f}%"
        if thr: s += f", A signals {(np.abs(x.close_p-0.5)>=thr).mean()*100:.0f}% -> {(np.abs(x.close_p_v-0.5)>=thr).mean()*100:.0f}% of windows"
        out.append(s)
    print(f"{name:20s}: " + " | ".join(out))
