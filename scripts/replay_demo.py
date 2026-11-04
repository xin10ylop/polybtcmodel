"""Replay the two strategies on a few stored windows from the trade records produced by
polybtc/final_report_data.py (reports/trades_*.csv)."""
import os, sys, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
for nm in ("B_open_mispricing", "A_preopen"):
    f = os.path.join(R, f"trades_{nm}.csv")
    if not os.path.exists(f):
        continue
    t = pd.read_csv(f); t = t[t.selected & t.filled]
    print(f"\n== {nm}: {len(t)} filled trades; last 8 ==")
    print(t.tail(8)[["T", "date", "side", "signal", "entry_px", "exit_kind", "exit_rel", "pnl", "settle_pnl", "up"]].round(4).to_string(index=False))
