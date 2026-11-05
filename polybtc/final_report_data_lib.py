import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polysim, experiments as ex, final_strategy as fs
def trades_A2(cfg, master, mk, g):
    settle = cfg.get("settle", False)
    sim = polysim.simulate(mk, g, t_entry=cfg["fill_by"], entry_mode="fixed", pe=cfg["entry_price"], tps=(cfg["tp"],), sls=(0.0 if settle else cfg["sl"],), tstops=(300 if settle else cfg["tstop"],), slip=cfg.get("slip", 0.01))
    m, sim = ex.align(master, sim)
    if settle:   # no stop, no time-stop: whatever did not reach the take-profit is held to settlement (1.00 / 0.00, no fee)
        idx = np.arange(len(m))
        for sd in (0, 1):
            k = sim["kind"][idx, sd, 0, 0, 0]; sim["pnl"][idx, sd, 0, 0, 0] = np.where(k == 0, cfg["tp"] - cfg["entry_price"], sim["settle"][idx, sd]); sim["kind"][idx, sd, 0, 0, 0] = np.where(k == 0, 0, 2)
    p = m[cfg["signal"]].values; side = np.where(p > 0.5, 0, 1)
    sel = np.isfinite(p) & (np.abs(p - 0.5) >= cfg["thr"])
    if cfg.get("agree"):
        z = m[cfg["agree"]].values; sel &= np.isfinite(z) & (np.sign(z) == np.sign(p - 0.5))
    return m, sim, p, side, sel, {}
fs.trades_A = trades_A2
