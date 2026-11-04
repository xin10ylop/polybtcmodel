"""Trade-management layer on top of strategy A: after entry, use the btcmovement2 close head re-evaluated
through the window (offsets d = 0..240, data < T+d) to (a) exit early when P(our side) < q_exit ("model stop",
sold at the first print after T+d+2 s, taker) and/or (b) cancel the 0.80 TP and hold to the time-stop when
P(our side) > q_run ("let winners run"). Compared with the frozen exits on old / tune / test."""
import os, sys, glob, numpy as np, pandas as pd, pyarrow.parquet as pq, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 230)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
OFFS = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 210, 240]
cols = ["up", "date", "m2_-45_close_p", "bp_-60_z_s5"] + [f"m2_{d}_close_p" for d in OFFS]
m = pd.read_parquet("/home/user/data/master.parquet", columns=cols)
p = m["m2_-45_close_p"].values; z = m["bp_-60_z_s5"].values
sel = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05) & np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5)) & (m.date.values >= "2026-08-01")
m = m[sel]; mk = pd.read_parquet(polysim.MARKETS).set_index("window_start"); mk = mk[mk.outcome_up_price.isin([0.0, 1.0])]
W = sorted(set(m.index.values) & set(mk.index.values)); g = {}
for f in sorted(glob.glob(f"{polysim.TRADES}/*.parquet")):
    t = pq.read_table(f, columns=["ts", "outcome_index", "price", "window_start"], filters=[("window_start", "in", W)]).to_pandas()
    if len(t) == 0: continue
    t["rel"] = (t.ts - t.window_start).astype(np.int32); t["p_up"] = np.where(t.outcome_index == 0, t.price, 1 - t.price).astype(np.float32)
    t = t.sort_values(["window_start", "ts"], kind="stable")
    for w, sub in t.groupby("window_start", sort=False): g[int(w)] = (sub.rel.values, sub.p_up.values)
W = [w for w in W if w in g]; m = m.loc[W]; print("trades:", len(W))
side_up = (m["m2_-45_close_p"].values > 0.5)
PM = np.column_stack([m[f"m2_{d}_close_p"].values for d in OFFS]); PM = np.where(side_up[:, None], PM, 1 - PM)   # P(our side) at each offset
TP, SL, TS, PE, SLIP, FEE = 0.80, 0.05, 240, 0.50, 0.01, 0.07
def fee(x): return FEE * x * (1 - x)

def run(q_exit=None, d_min=0, q_run=None, d_run_min=0):
    pnl = np.full(len(W), np.nan); kind = np.full(len(W), -1)
    for i, w in enumerate(W):
        rel, pup = g[w]; ps = pup if side_up[i] else 1 - pup
        mm = (rel >= -45) & (rel <= 300); t, pr = rel[mm], ps[mm]
        if len(pr) == 0: continue
        rm = np.maximum.accumulate(pr); rn = np.minimum.accumulate(pr)
        # let-run: if the model is confident at some checkpoint, the TP is cancelled from then on
        run_from = None
        if q_run is not None:
            hits = [d for d, pm in zip(OFFS, PM[i]) if d >= d_run_min and np.isfinite(pm) and pm > q_run]
            run_from = hits[0] if hits else None
        i_tp = int(np.searchsorted(rm, TP, side="left"))
        if run_from is not None and i_tp < len(pr) and t[i_tp] >= run_from + 2:
            i_tp = len(pr)                                           # TP cancelled before it would have filled
        i_sl = int(np.searchsorted(-rn, -SL, side="left")); i_ts = int(np.searchsorted(t, TS, side="right"))
        i_mx = len(pr); mx_px = None
        if q_exit is not None:
            hits = [d for d, pm in zip(OFFS, PM[i]) if d >= d_min and np.isfinite(pm) and pm < q_exit]
            if hits:
                j = int(np.searchsorted(t, hits[0] + 2, side="left"))
                if j < len(pr): i_mx = j
        first = min(i_tp, i_sl, i_ts, i_mx)
        if first == i_tp and i_tp < len(pr): px, k = TP, 0
        elif first == i_sl and i_sl < len(pr): px, k = pr[i_sl] - SLIP, 1
        elif first == i_mx and i_mx < len(pr): px, k = pr[i_mx] - SLIP, 3
        else: px, k = (pr[i_ts - 1] if i_ts > 0 else PE) - SLIP, 2
        px = float(np.clip(px, 0.001, 0.999)); f = 0.0 if k == 0 else fee(px)
        pnl[i] = px - PE - f; kind[i] = k
    return pnl, kind
per = {"old": m.date.values < "2026-08-20", "tune": (m.date.values >= "2026-08-20") & (m.date.values <= "2026-09-30"), "test": m.date.values >= "2026-10-01"}
def stats(pnl, kind, mask):
    x = pnl[mask]; d = pd.Series(x).groupby(m.date.values[mask]).sum(); k = kind[mask]
    return dict(mean=np.nanmean(x), win=np.nanmean(x > 0), sharpe=d.mean() / (d.std() + 1e-12) * np.sqrt(365), maxdd=float((d.cumsum().cummax() - d.cumsum()).max()),
                avg_loss=np.nanmean(x[x < 0]), p_tp=np.mean(k == 0), p_mx=np.mean(k == 3), p_time=np.mean(k == 2))
base_pnl, base_kind = run()
rows = []
def add(label, pnl, kind):
    r = {"rule": label}
    for nm, mask in per.items():
        s = stats(pnl, kind, mask); r.update({f"{k}_{nm}": v for k, v in s.items() if k in ("mean", "win", "sharpe", "avg_loss", "p_mx")})
        # paired day bootstrap vs baseline
        d = pd.Series(pnl[mask] - base_pnl[mask]).groupby(m.date.values[mask]).sum().values; rng = np.random.default_rng(0)
        b = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]; r[f"dpd_{nm}"] = d.mean(); r[f"p_{nm}"] = float(np.mean(np.array(b) <= 0))
    rows.append(r)
add("frozen exits (TP .80 / SL .05 / T+240)", base_pnl, base_kind)
for q in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
    for dmin in (0, 30, 60, 120):
        pnl, kind = run(q_exit=q, d_min=dmin); add(f"model stop q<{q} from T+{dmin}", pnl, kind)
for qr in (0.85, 0.90, 0.95):
    for drm in (60, 120):
        pnl, kind = run(q_run=qr, d_run_min=drm); add(f"let run if P>{qr} from T+{drm}", pnl, kind)
for q, dmin, qr, drm in ((0.25, 30, 0.90, 60), (0.30, 30, 0.90, 60), (0.30, 60, 0.95, 120), (0.25, 0, 0.95, 60)):
    pnl, kind = run(q_exit=q, d_min=dmin, q_run=qr, d_run_min=drm); add(f"stop q<{q}@{dmin} + run P>{qr}@{drm}", pnl, kind)
df = pd.DataFrame(rows); df.to_csv(os.path.join(R, "A_dynamic_exit.csv"), index=False)
c = ["rule", "mean_old", "mean_tune", "win_tune", "sharpe_tune", "avg_loss_tune", "dpd_tune", "p_tune", "mean_test", "win_test", "sharpe_test", "avg_loss_test", "p_mx_test", "dpd_test", "p_test"]
print(df[c].round(4).to_string(index=False))
