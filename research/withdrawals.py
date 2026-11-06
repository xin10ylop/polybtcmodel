"""Withdrawal policy on the A+B+D book at $10 stakes: the account starts with a bankroll, every closed trade adds or subtracts
cash, and a sweep rule moves cash above a buffer out of the account (daily / weekly / never). With fixed stakes the strategy's
earnings do not depend on the rule; what the rule changes is how much sits in the account, how low it gets, and whether it
ever drops below what three open $10 positions need (~$30, i.e. trading would stop). Periods: Aug 1-19, tune, test, June, July.
  python research/withdrawals.py"""
import os, numpy as np, pandas as pd
pd.set_option("display.width", 220); R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports"); STAKE = 10
def load(st_main, st_short):
    t = pd.read_csv(f"{R}/trades_{st_main}.csv"); t = t[(t.selected == 1) & (t.filled == 1)].copy(); t["per"] = np.select([t.date < "2026-08-20", t.date < "2026-10-01"], ["Aug1-19", "tune"], "test")
    parts = [t]
    for mon, lab in (("june", "June"), ("july", "July")):
        f = f"{R}/trades_{st_short}_{mon}.csv"
        if os.path.exists(f): x = pd.read_csv(f); x["per"] = lab; parts.append(x)
    d = pd.concat(parts, ignore_index=True); d["usd"] = d.pnl / d.entry_px * STAKE; return d
book = pd.concat([load("A_preopen", "A"), load("B_open_mispricing", "B"), load("D_high_confidence_settle", "D")], ignore_index=True)
daily = book.groupby(["per", "date"]).usd.sum()
rows = []
for per in ("Aug1-19", "tune", "test", "June", "July"):
    s = daily.loc[per].sort_index(); days = list(s.index)
    for start in (300, 500):
        for rule, buf in (("never", None), ("daily above $300", 300), ("daily above $500", 500), ("weekly above $300", 300), ("weekly above $500", 500), ("weekly above $400", 400)):
            bal = float(start); withdrawn = 0.0; low = bal; stopped = 0; path = []
            for i, d in enumerate(days):
                bal += s[d]; low = min(low, bal); path.append(bal)
                if bal < 3 * STAKE: stopped += 1                       # not enough cash for the book's three positions
                sweep = buf is not None and (rule.startswith("daily") or pd.Timestamp(d).dayofweek == 6)   # weekly = Sunday
                if sweep and bal > buf: withdrawn += bal - buf; bal = buf
            p = pd.Series(path); dd = (p.cummax() - p).max()
            rows.append(dict(period=per, days=len(days), start=start, rule=rule, earned=round(s.sum()), withdrawn=round(withdrawn), final=round(bal), lowest=round(low), acct_dd=round(dd), days_below_30=stopped))
df = pd.DataFrame(rows); print("A+B+D at $10 per trade; earned = the strategy's cash result for the period (independent of the rule); withdrawn + (final - start) = earned\n")
print(df.to_string(index=False)); df.to_csv(f"{R}/withdrawals.csv", index=False); print("saved reports/withdrawals.csv")
print("\nworst day and max account drawdown per period at $10 stakes:"); print(daily.groupby("per").agg(worst_day=lambda x: round(x.min()), total=lambda x: round(x.sum())).to_string())
