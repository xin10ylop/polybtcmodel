"""A whole month never used for anything, evaluated once with the CURRENT frozen rules (A: TP 0.85 then settle;
B: delta 0.12, TP 0.85 then settle; D: T+90, q 0.90; plus the previous-exit shadows A0 / B0).
  MONTH=2026-06 python research/oos_month.py
Inputs (built like July's, nothing in the main data set touched):
  scripts/fetch_polymarket_range.py <m-01> <next-m-01>            -> /home/user/data/polymarket/markets_<m-01>_<next-m-01>.parquet + trades/<day>.parquet
  TRUTH_DIR=/home/user/data/truth_<mon> scripts/truth.py <m-01> <m-last>
  M2_OUT=/home/user/data/forecasts_<mon>_L0 M2_REF_LAG=0 scripts/score_m2.py <m-01> <m-last>   (point-price resolution before Sept: L = 0)
  scripts/train_bp_<mon>.py  (btc-predict fold trained only on data before the month) -> forecasts/bp_minute_<mon>.parquet
  TRUTH_DIR=... POLY_MARKETS=... BP_MINUTE=... FORECASTS_DIR=... MASTER_OUT=/home/user/data/master_<mon>.parquet scripts/build_master.py
Writes reports/oos_<mon>.json and reports/trades_{A,B,D}_<mon>.csv.
"""
import os, sys, json, calendar, datetime as dt
import numpy as np, pandas as pd
MONTH = os.environ.get("MONTH", "2026-06"); y, mo = int(MONTH[:4]), int(MONTH[5:]); nd = calendar.monthrange(y, mo)[1]
mon = dt.date(y, mo, 1).strftime("%B").lower(); first = f"{MONTH}-01"; last = f"{MONTH}-{nd:02d}"
nxt = (dt.date(y, mo, 1) + dt.timedelta(days=nd)).isoformat()
os.environ.setdefault("MASTER_PATH", f"/home/user/data/master_{mon}.parquet")
os.environ.setdefault("POLY_MARKETS", f"/home/user/data/polymarket/markets_{first}_{nxt}.parquet")
os.environ.setdefault("POLY_DAYS", ",".join(f"{MONTH}-{d:02d}" for d in range(1, nd + 1)))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex, final_strategy as fs   # noqa: E402
pd.set_option("display.width", 220)
master, mk, g = ex.master, ex.mk, ex.g
print(f"{MONTH} master: {len(master)} windows, {master.date.min()} .. {master.date.max()}; polymarket windows with prints: {len(g)}")

CFG_A = {"kind": "A", "decision_offset": -45, "fill_by": -45, "entry_price": 0.50, "signal": "m2_-45_close_p", "agree": "bp_-60_z_s5", "thr": 0.05, "tp": 0.85, "sl": None, "tstop": None, "settle": True}
CFG_A0 = dict(CFG_A, tp=0.80, sl=0.05, tstop=240, settle=False)
CFG_B, CFG_B0, CFG_D = dict(fs.CFG_B), dict(fs.CFG_B0), dict(fs.CFG_D)

def trades_A(cfg, master, mk, g):   # same as final_report_data.trades_A2 (live rule: m2 close head + btc-predict agreement, no bm stack)
    settle = cfg.get("settle", False)
    sim = polysim.simulate(mk, g, t_entry=cfg["fill_by"], entry_mode="fixed", pe=cfg["entry_price"], tps=(cfg["tp"],), sls=(0.0 if settle else cfg["sl"],), tstops=(300 if settle else cfg["tstop"],), slip=cfg.get("slip", 0.01))
    m, sim = ex.align(master, sim)
    if settle:
        idx = np.arange(len(m))
        for sd in (0, 1):
            k = sim["kind"][idx, sd, 0, 0, 0]; sim["pnl"][idx, sd, 0, 0, 0] = np.where(k == 0, cfg["tp"] - cfg["entry_price"], sim["settle"][idx, sd]); sim["kind"][idx, sd, 0, 0, 0] = np.where(k == 0, 0, 2)
    p = m[cfg["signal"]].values; side = np.where(p > 0.5, 0, 1)
    sel = np.isfinite(p) & (np.abs(p - 0.5) >= cfg["thr"])
    if cfg.get("agree"):
        z = m[cfg["agree"]].values; sel &= np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5))
    return m, sim, p, side, sel, {}

def run(cfg, fn, label):
    m, sim, p, side, sel, _ = fn(cfg, master, mk, g)
    idx = np.arange(len(m)); pnl = np.where(sel, sim["pnl"][idx, side, 0, 0, 0], np.nan)
    k = sim["kind"][idx, side, 0, 0, 0]; entry = sim["entry_px"][idx, side]; tex = sim["texit"][idx, side, 0, 0, 0]
    s = (m.date.values >= first) & (m.date.values <= last)
    r = se.metrics(np.where(s, pnl, np.nan), m.date.values)
    r["signals"] = int((sel & s).sum()); r["fill_rate"] = float(np.isfinite(pnl[sel & s]).mean()) if (sel & s).sum() else np.nan
    f = s & np.isfinite(pnl)
    r["p_tp"] = float(np.mean(k[f] == 0)) if f.sum() else np.nan; r["p_sl"] = float(np.mean(k[f] == 1)) if f.sum() else np.nan; r["avg_entry"] = float(np.nanmean(entry[f])) if f.sum() else np.nan
    r["trades_per_day"] = r["n"] / max(r.get("days", 1), 1)
    rec = pd.DataFrame({"T": m.index.values[f], "date": m.date.values[f], "side": np.where(side[f] == 0, "UP", "DOWN"), "signal": p[f], "entry_px": entry[f],
                        "exit_kind": k[f], "exit_rel": tex[f], "pnl": pnl[f], "up": m.up.values[f]})
    usd = (rec.pnl / rec.entry_px * 100).groupby(rec.date).sum() if len(rec) else pd.Series(dtype=float)
    eq = usd.cumsum(); r["usd_total"] = float(eq.iloc[-1]) if len(eq) else 0.0; r["usd_maxdd"] = float((eq.cummax() - eq).max()) if len(eq) else 0.0
    r["usd_worst_day"] = float(usd.min()) if len(usd) else 0.0; r["neg_days"] = int((usd < 0).sum()); r["usd_per_day"] = float(usd.mean()) if len(usd) else 0.0
    print(f"\n=== {label} {json.dumps(cfg)} ===")
    print(pd.Series(r).reindex(["signals", "fill_rate", "n", "days", "trades_per_day", "avg_entry", "mean", "ret", "win", "p_tp", "p_sl", "sharpe", "maxdd", "tstat", "usd_total", "usd_per_day", "usd_maxdd", "usd_worst_day", "neg_days"]).round(4).to_string())
    if len(rec):
        print("by side:"); print(rec.groupby("side").pnl.agg(["size", "mean", lambda x: np.mean(x > 0)]).round(4).rename(columns={"<lambda_0>": "win"}).to_string())
        wk = rec.assign(week=pd.to_datetime(rec.date).dt.isocalendar().week).groupby("week").pnl.agg(["size", "mean", lambda x: np.mean(x > 0)]).round(4).rename(columns={"<lambda_0>": "win"})
        print("by ISO week:"); print(wk.to_string())
    return r, rec, usd

out = {}; daily = {}
for nm, cfg, fn in (("A", CFG_A, trades_A), ("A0", CFG_A0, trades_A), ("B", CFG_B, fs.trades_B), ("B0", CFG_B0, fs.trades_B), ("D", CFG_D, fs.trades_B)):
    r, rec, usd = run(cfg, fn, f"{nm} ({MONTH})"); out[nm] = r; daily[nm] = usd
    if nm in ("A", "B", "D"): rec.to_csv(os.path.join(HERE, "..", "reports", f"trades_{nm}_{mon}.csv"), index=False)
comb = pd.concat([daily[k].rename(k) for k in ("A", "B", "D")], axis=1).fillna(0); comb["A+B+D"] = comb.A + comb.B + comb.D
eq = comb["A+B+D"].cumsum()
print(f"\n== {MONTH} A+B+D at $100 per trade: total ${eq.iloc[-1]:.0f} (${comb['A+B+D'].mean():.0f}/day), max dd ${(eq.cummax()-eq).max():.0f}, worst day ${comb['A+B+D'].min():.0f}, negative days {(comb['A+B+D']<0).sum()}/{len(comb)}, corr(A,B) {comb.A.corr(comb.B):.2f}")
print(comb.round(0).to_string())
out["combined_daily_usd"] = comb.round(2).to_dict()
json.dump(out, open(os.path.join(HERE, "..", "reports", f"oos_{mon}.json"), "w"), indent=1, default=float); print("saved reports/oos_%s.json" % mon)
