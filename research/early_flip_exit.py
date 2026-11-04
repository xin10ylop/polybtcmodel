"""Strategy A + a pre-open / at-open model check: the position is bought at 0.50 by T-40 on the T-45
forecast; at later checkpoints d (T-40 .. T+30) the close head is re-evaluated with the data available
then. If P(our side) at a checkpoint drops below q, the position is sold at the first print after
T+d+2 s (taker). Compared with the frozen exits on every period, paired day bootstrap vs. baseline.
MASTER_PATH / POLY_MARKETS / POLY_DAYS select the data set (main = Aug-Nov, or July)."""
import os, sys, glob, numpy as np, pandas as pd, pyarrow.parquet as pq, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
LABEL = os.environ.get("LABEL", "main")
OFFS = [-40, -35, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30]
cols = ["up", "date", "m2_-45_close_p", "bp_-60_z_s5"] + [f"m2_{d}_close_p" for d in OFFS]
m = pd.read_parquet(os.environ.get("MASTER_PATH", "/home/user/data/master.parquet"), columns=cols)
p = m["m2_-45_close_p"].values; z = m["bp_-60_z_s5"].values
sel = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05) & np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5))
m = m[sel]; mk = pd.read_parquet(polysim.MARKETS).set_index("window_start"); mk = mk[mk.outcome_up_price.isin([0.0, 1.0])]
W = sorted(set(m.index.values) & set(mk.index.values)); g = {}
days = os.environ.get("POLY_DAYS", "").split(",") if os.environ.get("POLY_DAYS") else None
for f in sorted(glob.glob(f"{polysim.TRADES}/*.parquet")):
    if days and os.path.basename(f)[:10] not in days: continue
    t = pq.read_table(f, columns=["ts", "outcome_index", "price", "window_start"], filters=[("window_start", "in", W)]).to_pandas()
    if len(t) == 0: continue
    t["rel"] = (t.ts - t.window_start).astype(np.int32); t["p_up"] = np.where(t.outcome_index == 0, t.price, 1 - t.price).astype(np.float32)
    t = t.sort_values(["window_start", "ts"], kind="stable")
    for w, sub in t.groupby("window_start", sort=False): g[int(w)] = (sub.rel.values, sub.p_up.values)
W = [w for w in W if w in g]; m = m.loc[W]; print(LABEL, "trades:", len(W), m.date.min(), m.date.max())
side_up = (m["m2_-45_close_p"].values > 0.5); up = m.up.values; won_side = np.where(side_up, up == 1, up == 0)
PM = np.column_stack([m[f"m2_{d}_close_p"].values for d in OFFS]); PM = np.where(side_up[:, None], PM, 1 - PM)   # P(our side) at each checkpoint
TP, SL, TS, PE, SLIP, FEE = 0.80, 0.05, 240, 0.50, 0.01, 0.07
def fee(x): return FEE * x * (1 - x)
print("\ninformation value: outcome hit rate of the T-45 side when the checkpoint forecast agrees / disagrees (P(our side) < 0.5)")
for j, d in enumerate(OFFS):
    ok = np.isfinite(PM[:, j]); flip = ok & (PM[:, j] < 0.5)
    print(f"  T{d:+4d}s: flipped {flip.mean()*100:4.1f}% of trades; hit rate when not flipped {won_side[ok & ~flip].mean():.3f}, when flipped {won_side[flip].mean() if flip.any() else float('nan'):.3f}")
def run(q_exit=None, checks=()):
    pnl = np.full(len(W), np.nan); kind = np.full(len(W), -1); J = [OFFS.index(d) for d in checks]
    for i, w in enumerate(W):
        rel, pup = g[w]; ps = pup if side_up[i] else 1 - pup
        mm = (rel >= -45) & (rel <= 300); t, pr = rel[mm], ps[mm]
        if len(pr) == 0: continue
        rm = np.maximum.accumulate(pr); rn = np.minimum.accumulate(pr)
        i_tp = int(np.searchsorted(rm, TP, side="left")); i_sl = int(np.searchsorted(-rn, -SL, side="left")); i_ts = int(np.searchsorted(t, TS, side="right"))
        i_mx = len(pr)
        if q_exit is not None:
            hits = [OFFS[j] for j in J if np.isfinite(PM[i, j]) and PM[i, j] < q_exit]
            if hits:
                jj = int(np.searchsorted(t, hits[0] + 2, side="left"))
                if jj < len(pr): i_mx = jj
        first = min(i_tp, i_sl, i_ts, i_mx)
        if first == i_tp and i_tp < len(pr): px, k = TP, 0
        elif first == i_sl and i_sl < len(pr): px, k = pr[i_sl] - SLIP, 1
        elif first == i_mx and i_mx < len(pr): px, k = pr[i_mx] - SLIP, 3
        else: px, k = (pr[i_ts - 1] if i_ts > 0 else PE) - SLIP, 2
        px = float(np.clip(px, 0.001, 0.999)); f = 0.0 if k == 0 else fee(px)
        pnl[i] = px - PE - f; kind[i] = k
    return pnl, kind
dates = m.date.values
if LABEL == "main": per = {"old": dates < "2026-08-20", "tune": (dates >= "2026-08-20") & (dates <= "2026-09-30"), "test": dates >= "2026-10-01"}
else: per = {LABEL: np.ones(len(dates), bool)}
def stats(pnl, kind, mask):
    x = pnl[mask]; d = pd.Series(x).groupby(dates[mask]).sum(); k = kind[mask]
    return dict(mean=np.nanmean(x), win=np.nanmean(x > 0), sharpe=d.mean() / (d.std() + 1e-12) * np.sqrt(365), avg_loss=np.nanmean(x[x < 0]), p_mx=np.mean(k == 3), usd=float((x / PE * 100).sum()))
base_pnl, base_kind = run(); rows = []
def add(label, pnl, kind):
    r = {"rule": label}
    for nm, mask in per.items():
        s = stats(pnl, kind, mask); r.update({f"{k}_{nm}": v for k, v in s.items()})
        d = pd.Series(pnl[mask] - base_pnl[mask]).groupby(dates[mask]).sum().values; rng = np.random.default_rng(0)
        b = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]; r[f"dpd_{nm}"] = d.mean(); r[f"p_{nm}"] = float(np.mean(np.array(b) <= 0))
    rows.append(r)
add("frozen exits", base_pnl, base_kind)
for q in (0.5, 0.45, 0.4, 0.35):
    for name, checks in (("T-10", (-10,)), ("T0", (0,)), ("T+10", (10,)), ("T-25..0", (-25, -20, -15, -10, -5, 0)), ("T-10,0,+10", (-10, 0, 10)), ("0..+30", (0, 5, 10, 15, 20, 30)), ("all -40..+30", tuple(OFFS))):
        pnl, kind = run(q_exit=q, checks=checks); add(f"exit if P<{q} at {name}", pnl, kind)
df = pd.DataFrame(rows); df.to_csv(os.path.join(R, f"A_early_flip_exit_{LABEL}.csv"), index=False)
c = ["rule"] + [f"{k}_{nm}" for nm in per for k in ("mean", "win", "avg_loss", "p_mx", "usd", "dpd", "p")]
print("\n(mean/avg_loss per share; p_mx = share of trades closed by the model check; usd at $100 per trade; dpd = daily P&L difference vs frozen per $1 share; p = bootstrap P(no improvement))")
print(df[c].round(4).to_string(index=False))
