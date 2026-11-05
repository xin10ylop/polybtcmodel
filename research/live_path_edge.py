"""Edge of A / B / D with forecasts computed on the live data path without a perpetual feed (scripts/score_m2_synth.py)
versus the archive path, same periods, same rules. MASTER_PATH selects the master table."""
import os, sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import final_strategy as fs  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
print("master:", os.environ.get("MASTER_PATH", "/home/user/data/master.parquet"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import final_report_data_lib  # noqa: E402,F401
CFG_A = dict(fs.CFG_A, signal="m2_-45_close_p", agree="bp_-60_z_s5", fill_by=-45, entry_price=0.50)
for nm, cfg in (("A", CFG_A), ("B", fs.CFG_B), ("D", fs.CFG_D)):
    res, rec = fs.evaluate(cfg, verbose=False)
    print(f"{nm}: " + " | ".join(f"{p}: n={int(r.n)} mean {r['mean']*100:+.2f}c win {r.win:.3f}" for p, r in res.iterrows()))
