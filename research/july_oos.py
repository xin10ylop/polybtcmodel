"""July 2026: a month never used for anything (before the first Polymarket day we downloaded), evaluated
once with the frozen rules. Inputs are built by scripts/fetch_polymarket_range.py (markets + trades),
scripts/score_m2.py (btcmovement2 wf5_20260430, frozen in April), scripts/train_bp_july.py (btc-predict
fold trained on Feb-Jun only), scripts/truth.py and scripts/build_master.py with TRUTH_DIR/POLY_MARKETS/
BP_MINUTE/MASTER_OUT pointing at the July files. Nothing in the main data set is modified.
Caveat: in July the markets resolved on the point price (the 60 s TWAP rule started in September), so the
models' effective-window alignment is a slight mismatch; the exits use July's actual prints either way.
"""
import os, sys, json, datetime as dt
import numpy as np, pandas as pd
os.environ.setdefault("MASTER_PATH", "/home/user/data/master_july.parquet")
os.environ.setdefault("POLY_MARKETS", "/home/user/data/polymarket/markets_2026-07-01_2026-08-01.parquet")
os.environ.setdefault("POLY_DAYS", ",".join((dt.date(2026, 7, 1) + dt.timedelta(days=i)).isoformat() for i in range(31)))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex, final_strategy as fs   # noqa: E402
pd.set_option("display.width", 220)
master, mk, g = ex.master, ex.mk, ex.g
JULY = ("2026-07-01", "2026-07-31")
print(f"July master: {len(master)} windows, {master.date.min()} .. {master.date.max()}; polymarket windows with prints: {len(g)}")

CFG_A = {"kind": "A", "decision_offset": -45, "fill_by": -45, "entry_price": 0.50, "signal": "m2_-45_close_p", "agree": "bp_-60_z_s5", "thr": 0.05, "tp": 0.80, "sl": 0.05, "tstop": 240}
CFG_B = dict(fs.CFG_B)

def trades_A(cfg, master, mk, g):
    sim = polysim.simulate(mk, g, t_entry=cfg["fill_by"], entry_mode="fixed", pe=cfg["entry_price"], tps=(cfg["tp"],), sls=(cfg["sl"],), tstops=(cfg["tstop"],), slip=cfg.get("slip", 0.01))
    m, sim = ex.align(master, sim)
    p = m[cfg["signal"]].values; side = np.where(p > 0.5, 0, 1)
    sel = np.isfinite(p) & (np.abs(p - 0.5) >= cfg["thr"])
    if cfg.get("agree"):
        z = m[cfg["agree"]].values; sel &= np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5))
    return m, sim, p, side, sel, {}

def run(cfg, fn, label):
    m, sim, p, side, sel, _ = fn(cfg, master, mk, g)
    idx = np.arange(len(m)); pnl = np.where(sel, sim["pnl"][idx, side, 0, 0, 0], np.nan)
    k = sim["kind"][idx, side, 0, 0, 0]; entry = sim["entry_px"][idx, side]
    s = (m.date.values >= JULY[0]) & (m.date.values <= JULY[1])
    r = se.metrics(np.where(s, pnl, np.nan), m.date.values)
    f = s & np.isfinite(pnl)
    r["signals"] = int((sel & s).sum()); r["fill_rate"] = float(np.isfinite(pnl[sel & s]).mean()) if (sel & s).sum() else np.nan
    r["p_tp"] = float(np.mean(k[f] == 0)); r["p_sl"] = float(np.mean(k[f] == 1)); r["avg_entry"] = float(np.nanmean(entry[f]))
    r["trades_per_day"] = r["n"] / max(r.get("days", 1), 1)
    rec = pd.DataFrame({"T": m.index.values[f], "date": m.date.values[f], "side": np.where(side[f] == 0, "UP", "DOWN"), "signal": p[f], "entry_px": entry[f],
                        "exit_kind": k[f], "pnl": pnl[f], "up": m.up.values[f]})
    usd = (rec.pnl / rec.entry_px * 100).groupby(rec.date).sum()
    eq = usd.cumsum(); r["usd_total"] = float(eq.iloc[-1]); r["usd_maxdd"] = float((eq.cummax() - eq).max()); r["usd_worst_day"] = float(usd.min()); r["neg_days"] = int((usd < 0).sum())
    print(f"\n=== {label} {json.dumps(cfg)} ===")
    print(pd.Series(r)[["signals", "fill_rate", "n", "days", "trades_per_day", "avg_entry", "mean", "ret", "win", "p_tp", "p_sl", "sharpe", "maxdd", "tstat", "usd_total", "usd_maxdd", "usd_worst_day", "neg_days"]].round(4).to_string())
    print("by side:"); print(rec.groupby("side").pnl.agg(["size", "mean", lambda x: np.mean(x > 0)]).round(4).rename(columns={"<lambda_0>": "win"}).to_string())
    wk = rec.assign(week=pd.to_datetime(rec.date).dt.isocalendar().week).groupby("week").pnl.agg(["size", "mean", lambda x: np.mean(x > 0)]).round(4).rename(columns={"<lambda_0>": "win"})
    print("by ISO week:"); print(wk.to_string())
    return r, rec, usd

out = {}
rA, recA, uA = run(CFG_A, trades_A, "A frozen rule (m2 + btc-predict agreement, TP 0.80 / SL 0.05 / T+240)")
rA0, _, _ = run(dict(CFG_A, agree=None), trades_A, "A without the agreement filter (information)")
rB, recB, uB = run(CFG_B, fs.trades_B, "B frozen rule (delta 0.12, TP 0.80 / SL 0.10 / T+240)")
comb = pd.concat([uA.rename("A"), uB.rename("B")], axis=1).fillna(0); comb["A+B"] = comb.A + comb.B
eq = comb["A+B"].cumsum()
print(f"\n== combined July at $100 per trade: total ${eq.iloc[-1]:.0f}, max dd ${(eq.cummax()-eq).max():.0f}, worst day ${comb['A+B'].min():.0f}, negative days {(comb['A+B']<0).sum()}/{len(comb)}, corr(A,B) {comb.A.corr(comb.B):.2f}")
print(comb.round(0).to_string())
out = {"A": rA, "A_no_agree": rA0, "B": rB, "combined_daily_usd": comb.round(2).to_dict()}
OUT = os.path.join(HERE, "..", "reports"); json.dump(out, open(f"{OUT}/july_oos.json", "w"), indent=1, default=float)
recA.to_csv(f"{OUT}/trades_A_july.csv", index=False); recB.to_csv(f"{OUT}/trades_B_july.csv", index=False); print("saved", OUT)
