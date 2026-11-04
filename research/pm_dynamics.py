"""Polymarket share price vs Binance displacement from the (lagged) reference; implied per-window lag;
end-of-window mispricing check."""
import os
import numpy as np, pandas as pd, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim
mk, g = polysim.load_windows()
W, grid = polysim.path_grid(g, -60, 300)          # UP-equiv last price per second, cols = rel -60..300
tw = pd.read_parquet("/home/user/data/truth/windows.parquet").set_index("T")
paths = np.load("/home/user/data/truth/paths.npy")  # [n_truth, 421], cols = T-120 .. T+300
tidx = {t: i for i, t in enumerate(tw.index)}
sel = np.array([w in tidx for w in W]); W = W[sel]; grid = grid[sel]
P = np.stack([paths[tidx[w]] for w in W]); lp = np.log(P)
up = (mk.loc[W, "outcome_up_price"].values == 1.0)
print("windows", len(W))
# implied reference lag per window: which k in [-60, 0] best explains the share price path via a logistic in displacement?
# simple version: correlation between share-price changes and binance returns at lag
rel = np.arange(-60, 301)
def disp(k):   # binance displacement (bp) at each rel time from the ref sampled at T+k
    ref = lp[:, 120 + k]
    return 1e4 * (lp[:, 120 + rel] - ref[:, None])
best = []
for k in range(-60, 1, 2):
    x = disp(k)
    m = (rel >= 30) & (rel <= 270)
    c = np.nanmean([np.corrcoef(x[i, m], grid[i, m])[0, 1] for i in range(0, len(W), 3) if np.isfinite(grid[i, m]).all()])
    best.append((round(c, 4), k))
best.sort(reverse=True); print("corr(share price, displacement from ref at T+k) best:", best[:6])
k = best[0][1]
x = disp(k)
# mapping share price <- displacement, by remaining time bucket
print("== mean UP price by displacement (bp) and elapsed bucket ==")
rows = []
for lo, hi in [(0, 30), (30, 60), (60, 120), (120, 180), (180, 240), (240, 270), (270, 300)]:
    m = (rel >= lo) & (rel < hi)
    xb = pd.cut(x[:, m].ravel(), [-100, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 100])
    s = pd.Series(grid[:, m].ravel()).groupby(xb, observed=True).mean()
    rows.append(s.rename(f"e{lo}-{hi}"))
print(pd.concat(rows, axis=1).round(3).to_string())
# end-of-window: binance crosses the reference after T+270 -> does the market misprice?
ref = lp[:, 120 + k]
x270 = 1e4 * (lp[:, 120 + 270] - ref); x295 = 1e4 * (lp[:, 120 + 295] - ref)
cross = (np.sign(x270) != np.sign(x295)) & (np.abs(x270) > 2) & (np.abs(x295) > 2)
print(f"\nwindows where binance crosses ref between T+270 and T+295 (|x|>2bp both): {cross.sum()} ({cross.mean():.3f})")
p295 = grid[:, 60 + 295]
print("  outcome agrees with sign at T+270:", np.mean((x270[cross] > 0) == up[cross]).round(3), " with sign at T+295:", np.mean((x295[cross] > 0) == up[cross]).round(3))
print("  mean UP price at T+295 when x270>0 (should be ~1 if market knows lag):", np.nanmean(p295[cross & (x270 > 0)]).round(3), " when x270<0:", np.nanmean(p295[cross & (x270 < 0)]).round(3))
# generic: price at T+295 vs outcome
print("  all windows: mean |p295 - outcome|:", np.nanmean(np.abs(p295 - up)).round(3), " P(|p295-outcome|>0.5):", np.nanmean(np.abs(p295 - up) > 0.5).round(4))
# when did the share price first leave [0.47,0.53]? relative to T (implied reference fix timing + reaction)
first = np.array([rel[np.argmax(np.abs(grid[i] - 0.5) > 0.04)] if (np.abs(grid[i] - 0.5) > 0.04).any() else np.nan for i in range(len(W))])
print("\nfirst time |p-0.5|>0.04: percentiles", np.nanpercentile(first, [5, 10, 25, 50, 75, 90]))
# pre-entry: share price at T-45 and T-60
for t in (-60, -45, -35):
    v = grid[:, 60 + t]
    print(f"  price at T{t}: mean {np.nanmean(v):.4f} sd {np.nanstd(v):.4f}  P(|p-0.5|>0.02)={np.nanmean(np.abs(v-0.5)>0.02):.3f}  NaN {np.isnan(v).mean():.3f}")
