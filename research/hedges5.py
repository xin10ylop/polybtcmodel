"""Five hedging structures on top of A (and one on B), all with resting limit orders on the other token:
  H1 hedge after TP      : A (buy 0.50 at T-45, sell 0.85 then settle); once our side prints >= 0.85, bid the other side at Y until the close
  H2 pair completion     : A without TP; bid the other side at X for the whole window -> both sides owned for 0.50+X < 1, settlement pays 1.00 (locked 0.50-X)
  H3 pair or TP, first   : as H2 but the 0.85 sell limit stays live; whichever fills first decides (lock, or +0.35), else settle
  H4 B pair completion   : B fill at p (model side, last print at T, 10 s); bid the other side at 1-p-margin until the close; else B's TP .85 then settle
  H5 stop replacement    : A; instead of a stop, bid the other side at H (0.55-0.70) -> if filled the loss is locked at 0.50+H-1 with no fee; else TP .85 / settle
P&L per window combines the legs; 'win' = window P&L > 0. Selection on tune (>= 150 trades), read on test and Aug 1-19 (LABEL=july for July).
"""
import os, sys, glob, itertools, numpy as np, pandas as pd, pyarrow.parquet as pq, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 260)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
LABEL = os.environ.get("LABEL", "main"); master, mk, g = ex.master, ex.mk, ex.g
PERS = {"old": se.OLD, "tune": se.TUNE, "test": se.TEST} if LABEL == "main" else {"july": ("2026-07-01", "2026-07-31")}
SEL = "tune" if LABEL == "main" else "july"
rows = []
def per_stats(pnl, ent, dates):
    out = {}
    for nm, per in PERS.items():
        s = (dates >= per[0]) & (dates <= per[1]); x = np.where(s, pnl, np.nan); mt = se.metrics(x, dates); f = s & np.isfinite(pnl)
        d = pd.Series(pnl[f] / ent[f] * 100).groupby(dates[f]).sum() if f.any() else pd.Series(dtype=float)
        out[nm] = dict(n=mt.get("n", 0), mean=mt.get("mean", np.nan), win=mt.get("win", np.nan), usd=float(d.sum()) if len(d) else 0.0,
                       maxdd=float((d.cumsum().cummax() - d.cumsum()).max()) if len(d) else np.nan, tstat=mt.get("tstat", np.nan), stake=float(np.nanmean(ent[f])) if f.any() else np.nan)
    return out
def add(idea, variant, pnl, ent, dates, extra=None):
    st = per_stats(pnl, ent, dates); r = {"idea": idea, "variant": variant}
    for nm, d in st.items(): r.update({f"{k}_{nm}": v for k, v in d.items()})
    if extra: r.update(extra)
    rows.append(r)
def paths(W):
    gg = {}; days = os.environ.get("POLY_DAYS", "").split(",") if os.environ.get("POLY_DAYS") else None
    for f in sorted(glob.glob(f"{polysim.TRADES}/*.parquet")):
        if days and os.path.basename(f)[:10] not in days: continue
        t = pq.read_table(f, columns=["ts", "outcome_index", "price", "window_start"], filters=[("window_start", "in", W)]).to_pandas()
        if not len(t): continue
        t["rel"] = (t.ts - t.window_start).astype(np.int32); t["p_up"] = np.where(t.outcome_index == 0, t.price, 1 - t.price).astype(np.float32); t = t.sort_values(["window_start", "ts"], kind="stable")
        for w, sub in t.groupby("window_start", sort=False): gg[int(w)] = (sub.rel.values, sub.p_up.values)
    return gg

# ---------------- A windows
p45 = master["m2_-45_close_p"].values; z5 = master["bp_-60_z_s5"].values
selA = np.isfinite(p45) & (np.abs(p45 - 0.5) >= 0.05) & np.isfinite(z5) & (np.sign(z5) == np.sign(p45 - 0.5))
mA = master[selA]; WA = sorted(set(mA.index.values) & set(mk.index.values)); gA = paths(WA); WA = [w for w in WA if w in gA]; mA = mA.loc[WA]
su = mA["m2_-45_close_p"].values > 0.5; upA = mA.up.values; wonA = np.where(su, upA == 1, upA == 0); dA = mA.date.values
print(LABEL, "A windows:", len(WA), flush=True)
def a_paths(i):
    rel, pup = gA[WA[i]]; ps = pup if su[i] else 1 - pup; mm = (rel >= -45) & (rel <= 300); return rel[mm], ps[mm]
TP = 0.85
def a_leg_settle(i, t, pr):
    """A as adopted: TP 0.85 (first print >= TP), else settle. Returns (pnl, time of TP or None)."""
    j = int(np.searchsorted(np.maximum.accumulate(pr), TP)) if len(pr) else len(pr)
    if j < len(pr): return TP - 0.5, j
    return (1.0 if wonA[i] else 0.0) - 0.5, None
# baseline
base = np.array([a_leg_settle(i, *a_paths(i))[0] for i in range(len(WA))]); add("A adopted (reference)", "sell .85 else settle", base, np.full(len(WA), 0.5), dA)

# H1: hedge after TP print
for Y in (0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
    pnl = np.full(len(WA), np.nan); ent = np.full(len(WA), 0.5)
    for i in range(len(WA)):
        t, pr = a_paths(i); a_pnl, j = a_leg_settle(i, t, pr); pnl[i] = a_pnl
        if j is not None:
            other = 1 - pr[j + 1:]
            if len(other) and (other <= Y + 1e-9).any():
                pnl[i] += (0.0 if wonA[i] else 1.0) - Y; ent[i] = 0.5 + Y
    add("H1 hedge after TP", f"other side bid {Y:.2f} after the .85 print", pnl, ent, dA)
    f = ent > 0.5; add("H1 hedge after TP", f"  (only windows where the hedge filled, {Y:.2f})", np.where(f, pnl, np.nan), ent, dA)
print("H1 done", flush=True)

# H2: pair completion without TP ; H3: pair or TP, whichever first ; H5: stop replacement
for X in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
    pnl2 = np.full(len(WA), np.nan); ent2 = np.full(len(WA), 0.5); pnl3 = pnl2.copy(); ent3 = ent2.copy(); lock2 = np.zeros(len(WA), bool); lock3 = np.zeros(len(WA), bool); tp3 = np.zeros(len(WA), bool)
    for i in range(len(WA)):
        t, pr = a_paths(i)
        after = t >= -40                                                     # hedge bid live from our fill
        other = np.where(after, 1 - pr, np.inf); k = int(np.argmax(other <= X + 1e-9)) if (other <= X + 1e-9).any() else None
        j = int(np.searchsorted(np.maximum.accumulate(pr), TP)) if len(pr) else len(pr); j = j if j < len(pr) else None
        settle = (1.0 if wonA[i] else 0.0) - 0.5
        # H2
        if k is not None: pnl2[i] = 0.5 - X; ent2[i] = 0.5 + X; lock2[i] = True
        else: pnl2[i] = settle
        # H3
        if k is not None and (j is None or k <= j): pnl3[i] = 0.5 - X; ent3[i] = 0.5 + X; lock3[i] = True
        elif j is not None: pnl3[i] = TP - 0.5; tp3[i] = True
        else: pnl3[i] = settle
    add("H2 pair completion (no TP)", f"other side bid {X:.2f} -> lock {0.5-X:+.2f}", pnl2, ent2, dA, {"lock_share": float(lock2.mean())})
    add("H3 pair or TP first", f"other side bid {X:.2f} (lock {0.5-X:+.2f}) or TP .85", pnl3, ent3, dA, {"lock_share": float(lock3.mean()), "tp_share": float(tp3.mean())})
print("H2/H3 done", flush=True)
for H in (0.55, 0.60, 0.65, 0.70, 0.75):
    pnl = np.full(len(WA), np.nan); ent = np.full(len(WA), 0.5); locked = np.zeros(len(WA), bool)
    for i in range(len(WA)):
        t, pr = a_paths(i); after = t >= -40
        other = np.where(after, 1 - pr, np.inf); k = int(np.argmax(other <= H + 1e-9)) if (other <= H + 1e-9).any() else None
        j = int(np.searchsorted(np.maximum.accumulate(pr), TP)) if len(pr) else len(pr); j = j if j < len(pr) else None
        if k is not None and (j is None or k <= j): pnl[i] = 1.0 - 0.5 - H; ent[i] = 0.5 + H; locked[i] = True
        elif j is not None: pnl[i] = TP - 0.5
        else: pnl[i] = (1.0 if wonA[i] else 0.0) - 0.5
    add("H5 stop replacement", f"buy other side at {H:.2f} when offered -> loss locked at {1-0.5-H:+.2f}; TP .85 / settle otherwise", pnl, ent, dA, {"lock_share": float(locked.mean())})
print("H5 done", flush=True)

# ---------------- H4: B pair completion
sim = polysim.simulate(mk, g, t_entry=0, entry_mode="maker_last", t_cancel=10, tps=(0.85,), sls=(0.0,), tstops=(300,), slip=0.01); mB, sim = ex.align(master, sim)
mod = mB["m2_0_close_p"].values; mkt = sim["pre_last"][:, 0]; dis = mod - mkt; sideB = np.where(dis > 0, 0, 1); idx = np.arange(len(mB))
selB = np.isfinite(dis) & (np.abs(dis) >= 0.12) & np.isfinite(sim["fill_rel"][idx, sideB])
WB = [int(w) for w in mB.index.values[selB]]; gB = paths(WB); keep = [i for i, w in enumerate(mB.index.values) if selB[i] and int(w) in gB]
entB = sim["entry_px"][idx, sideB]; fillB = sim["fill_rel"][idx, sideB]; upB = mB.up.values; dB = mB.date.values
print(LABEL, "B windows:", len(keep), flush=True)
baseB = np.full(len(mB), np.nan)
for i in keep:
    rel, pup = gB[int(mB.index.values[i])]; ps = pup if sideB[i] == 0 else 1 - pup; mm = rel > fillB[i]; t, pr = rel[mm], ps[mm]
    j = int(np.searchsorted(np.maximum.accumulate(pr), TP)) if len(pr) else len(pr)
    won = (upB[i] == 1) if sideB[i] == 0 else (upB[i] == 0)
    baseB[i] = (TP - entB[i]) if j < len(pr) else (1.0 if won else 0.0) - entB[i]
add("B adopted-exits (reference)", "B fill, sell .85 else settle", baseB, entB, dB)
for margin in (0.05, 0.10, 0.15, 0.20, 0.25):
    pnl = np.full(len(mB), np.nan); ent = entB.copy(); locked = np.zeros(len(mB), bool)
    for i in keep:
        rel, pup = gB[int(mB.index.values[i])]; ps = pup if sideB[i] == 0 else 1 - pup; mm = rel > fillB[i]; t, pr = rel[mm], ps[mm]
        X = 1 - entB[i] - margin
        other = 1 - pr; k = int(np.argmax(other <= X + 1e-9)) if (other <= X + 1e-9).any() else None
        j = int(np.searchsorted(np.maximum.accumulate(pr), TP)) if len(pr) else len(pr); j = j if j < len(pr) else None
        won = (upB[i] == 1) if sideB[i] == 0 else (upB[i] == 0)
        if k is not None and (j is None or k <= j): pnl[i] = margin; ent[i] = entB[i] + X; locked[i] = True
        elif j is not None: pnl[i] = TP - entB[i]
        else: pnl[i] = (1.0 if won else 0.0) - entB[i]
    add("H4 B pair completion", f"other side bid at 1-p-{margin:.2f} (lock {margin:+.2f}) or TP .85", pnl, ent, dB, {"lock_share": float(np.mean(locked[keep]))})
print("H4 done", flush=True)

df = pd.DataFrame(rows); df.to_csv(os.path.join(R, f"hedges5_{LABEL}.csv"), index=False)
cols = ["idea", "variant"] + [f"{k}_{nm}" for nm in PERS for k in ("n", "mean", "win", "usd", "maxdd")] + [c for c in ("lock_share", "tp_share") if c in df]
print("\n== all variants (mean = per window, in $1-share units; usd at $100 of stake per window) ==")
print(df[cols].round(3).to_string(index=False))
