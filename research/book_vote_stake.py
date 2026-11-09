"""Stake size vs capacity for frozen A and A + book vote.  Capacity of a window = $ notional printed at 0.49-0.51 (UP-equivalent)
BEFORE the nominal open (rel < 0); a $S stake per trade is assumed to fill min(S, capacity).  Monthly P&L at S = 100 / 150 / 200,
unconstrained (x S/100) and capacity-constrained.  Uses the per-trade records logic of research/book_vote_monthly.py."""
import sys, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
sys.path.insert(0, "/home/user/polybtcmodel/polybtc"); import polysim, bookdepth  # noqa: E402
exec(open("/home/user/polybtcmodel/research/book_vote_monthly.py").read().split("def tab(")[0].split("import polysim, bookdepth")[1])   # builds R (trade records)
cap = {}
for f in sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "outcome_index", "price", "size", "window_start"])
    pu = np.where(t.outcome_index.values == 0, t.price.values, 1 - t.price.values); pre = t.ts.values < t.window_start.values
    m = pre & (pu >= 0.49) & (pu <= 0.51)
    s = pd.Series(t.price.values[m] * t["size"].values[m]).groupby(t.window_start.values[m]).sum()
    for k, v in s.items(): cap[int(k)] = cap.get(int(k), 0.0) + float(v)
R["cap"] = R["T"].map(cap).fillna(0.0)
print("capacity at 0.49-0.51 before the open, A+book windows: median $%.0f;  share of windows with >= $100: %.0f%%, >= $150: %.0f%%, >= $200: %.0f%%" % (
    R[R.AB].cap.median(), 100 * (R[R.AB].cap >= 100).mean(), 100 * (R[R.AB].cap >= 150).mean(), 100 * (R[R.AB].cap >= 200).mean()))
rows = []
for mo, grp in list(R.groupby("month")) + [("total", R)]:
    r = {"month": mo}
    for lab, sel in (("A", grp.A), ("A+book", grp.AB)):
        g = grp[sel]
        for S in (100, 150, 200):
            for cap_lab, stake in (("", S), (" capped", np.minimum(S, g.cap))):      # capped: fills only what printed at 0.49-0.51 pre-open
                usd = g.pnl * 2 * stake                                              # pnl per share x shares (stake / 0.50)
                daily = usd.groupby(g.date).sum(); eq = daily.cumsum()
                r[f"{lab} ${S}{cap_lab}"] = usd.sum(); r[f"{lab} ${S}{cap_lab} maxdd"] = (eq.cummax() - eq).max() if len(eq) else 0.0
    rows.append(r)
M = pd.DataFrame(rows).set_index("month").round(0)
for lab in ("A", "A+book"):
    print(f"\n== {lab}: monthly P&L and max drawdown (daily closes) by stake; 'capped' = fill min(stake, $ printed at 0.49-0.51 before the open) ==")
    print(M[[c for c in M.columns if c.startswith(lab + " $")]].to_string())
M.to_csv("/home/user/polybtcmodel/reports/A_book_vote_stake.csv")
