"""Final strategies, evaluated on TUNE (selection) and TEST (untouched), with trade-level records.

A  'pre-open directional':  decide at T-45 from the stacked 5-minute direction forecast; buy the chosen side
   at 0.50 before the reference is fixed (user's execution assumption: always filled); exits TP/SL/time-stop.
B  'open mispricing':       at the nominal open T (data < T), compare the btcmovement2 close-head P(up)
   (effective window, reference at T-30) with the last Polymarket print; if they disagree by >= delta,
   rest a bid at the last print for `cancel` seconds (maker, fills only if hit); exits TP/SL/time-stop.
"""
import os
import numpy as np, pandas as pd, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
TUNE, TEST, OLD = se.TUNE, se.TEST, se.OLD


def trades_A(cfg, master, mk, g):
    d = cfg["decision_offset"]
    sim = polysim.simulate(mk, g, t_entry=cfg.get("fill_by", -45), entry_mode="fixed", pe=cfg.get("entry_price", 0.5),
                           tps=(cfg["tp"],), sls=(cfg["sl"],), tstops=(cfg["tstop"],), slip=cfg.get("slip", 0.01))
    m, sim = ex.align(master, sim)
    p, coef = ex.stacked_signal(m, d, cols=cfg["stack_cols"])
    side = np.where(p > 0.5, 0, 1); idx = np.arange(len(m))
    sel = np.isfinite(p) & (np.abs(p - 0.5) >= cfg["thr"])
    return m, sim, p, side, sel, coef


def trades_B(cfg, master, mk, g):
    d = cfg["decision_offset"]
    sim = polysim.simulate(mk, g, t_entry=d, entry_mode="maker_last", t_cancel=d + cfg["cancel"],
                           tps=(cfg["tp"],), sls=(cfg["sl"],), tstops=(cfg["tstop"],), slip=cfg.get("slip", 0.01))
    m, sim = ex.align(master, sim)
    mod = m[f"m2_{d}_close_p"].values; mkt = sim["pre_last"][:, 0]
    dis = mod - mkt; p = 0.5 + dis
    side = np.where(dis > 0, 0, 1)
    sel = np.isfinite(dis) & (np.abs(dis) >= cfg["delta"])
    if cfg.get("max_entry") is not None:            # do not chase: skip if our side already trades above max_entry
        sel &= sim["pre_last"][np.arange(len(m)), side] <= cfg["max_entry"]
    return m, sim, p, side, sel, {}


def evaluate(cfg, master=None, mk=None, g=None, verbose=True, label=""):
    master = ex.master if master is None else master; mk = ex.mk if mk is None else mk; g = ex.g if g is None else g
    m, sim, p, side, sel, coef = (trades_A if cfg["kind"] == "A" else trades_B)(cfg, master, mk, g)
    idx = np.arange(len(m))
    pnl = np.where(sel, sim["pnl"][idx, side, 0, 0, 0], np.nan)
    k = sim["kind"][idx, side, 0, 0, 0]; entry = sim["entry_px"][idx, side]; tex = sim["texit"][idx, side, 0, 0, 0]
    rec = pd.DataFrame({"T": m.index.values, "date": m.date.values, "month": m.month.values, "side": np.where(side == 0, "UP", "DOWN"),
                        "signal": p, "selected": sel, "filled": np.isfinite(pnl), "entry_px": np.where(sel, entry, np.nan), "exit_kind": np.where(np.isfinite(pnl), k, -1),
                        "exit_rel": np.where(np.isfinite(pnl), tex, np.nan), "pnl": pnl, "settle_pnl": np.where(np.isfinite(pnl), sim["settle"][idx, side], np.nan),
                        "vol_prev5m_bp": m.vol_prev5m_bp.values, "hour_et": m.hour_et.values, "weekend": m.weekend.values, "up": m.up.values})
    out = {}
    for split, nm in ((OLD, "old(Aug1-19)"), (TUNE, "tune"), (TEST, "test")):
        s = (m.date.values >= split[0]) & (m.date.values <= split[1])
        r = se.metrics(np.where(s, pnl, np.nan), m.date.values)
        r["signals"] = int((sel & s).sum()); r["fill_rate"] = float(np.isfinite(pnl[sel & s]).mean()) if (sel & s).sum() else np.nan
        f = s & np.isfinite(pnl)
        r["settle_mean"] = float(np.nanmean(sim["settle"][idx, side][f])) if f.sum() else np.nan
        r["p_tp"] = float(np.mean(k[f] == 0)) if f.sum() else np.nan; r["p_sl"] = float(np.mean(k[f] == 1)) if f.sum() else np.nan
        r["avg_entry"] = float(np.nanmean(entry[f])) if f.sum() else np.nan; r["trades_per_day"] = r["n"] / max(r.get("days", 1), 1)
        out[nm] = r
    res = pd.DataFrame(out).T
    if verbose:
        print(f"\n=== {label} {json.dumps(cfg)} ===\n" + (f"stack coef: {coef}\n" if coef else ""))
        print(res[["signals", "fill_rate", "n", "trades_per_day", "avg_entry", "mean", "ret", "win", "p_tp", "p_sl", "settle_mean", "sharpe", "maxdd", "tstat"]].round(4).to_string())
    return res, rec


def robustness(cfg, label=""):
    """Monthly, regime slices and cost sensitivity on TEST."""
    res, rec = evaluate(cfg, verbose=False)
    t = rec[(rec.date >= TEST[0]) & (rec.date <= TEST[1]) & rec.filled]
    print(f"\n--- {label}: TEST by month ---")
    print(t.groupby("month").pnl.agg(["size", "mean", "sum"]).round(4).to_string())
    vq = np.nanquantile(rec.vol_prev5m_bp[(rec.date >= TUNE[0]) & (rec.date <= TUNE[1])], [1 / 3, 2 / 3])
    sl = {"vol low": t.vol_prev5m_bp < vq[0], "vol mid": (t.vol_prev5m_bp >= vq[0]) & (t.vol_prev5m_bp < vq[1]), "vol high": t.vol_prev5m_bp >= vq[1],
          "weekend": t.weekend == 1, "weekday": t.weekend == 0, "ET 0-8h": t.hour_et < 8, "ET 8-16h": (t.hour_et >= 8) & (t.hour_et < 16), "ET 16-24h": t.hour_et >= 16,
          "side UP": t.side == "UP", "side DOWN": t.side == "DOWN"}
    rows = []
    for k_, f in sl.items():
        r = se.metrics(t.pnl.values[f.values], t.date.values[f.values]); r["slice"] = k_; rows.append(r)
    print(pd.DataFrame(rows).set_index("slice")[["n", "mean", "win", "sharpe", "tstat"]].round(4).to_string())
    print("exit kinds (test):", t.exit_kind.map({0: "TP", 1: "SL", 2: "time"}).value_counts(normalize=True).round(3).to_dict(),
          " median exit second:", float(np.nanmedian(t.exit_rel)))
    for slip in (0.0, 0.02, 0.03):
        c2 = dict(cfg, slip=slip); r2, _ = evaluate(c2, verbose=False)
        print(f"slippage {slip}: test mean {r2.loc['test', 'mean']:.4f} tstat {r2.loc['test', 'tstat']:.2f}")
    return rec


CFG_A = {"kind": "A", "decision_offset": -45, "fill_by": -45, "entry_price": 0.50, "stack_cols": ["m2_-45_close_p", "bm_-45_close_p", "bp_-60_z_s5"],
         "thr": 0.05, "tp": 0.75, "sl": 0.20, "tstop": 240}
CFG_B = {"kind": "B", "decision_offset": 0, "cancel": 10, "delta": 0.12, "tp": 0.80, "sl": 0.10, "tstop": 240}
CFG_B2 = {"kind": "B", "decision_offset": 0, "cancel": 10, "delta": 0.08, "tp": 0.80, "sl": 0.10, "tstop": 240}

if __name__ == "__main__":
    for nm, cfg in (("A", CFG_A), ("B", CFG_B), ("B2", CFG_B2)):
        evaluate(cfg, label=nm)
    for nm, cfg in (("B", CFG_B),):
        robustness(cfg, nm)
