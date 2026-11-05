"""Daily take-profit / stop-loss for strategy A: stop opening new A trades for the UTC day once the day's REALISED P&L
(trades already closed at the decision instant T-45) reaches +X (take-profit) or -Y (stop). Open positions run to their
own exit. Evaluated on every period with the current exits (sell 0.85, else settle): Aug 1-19, tune, test (reports/
trades_A_preopen.csv), June and July (reports/trades_A_{june,july}.csv from research/oos_month.py). $100 per trade.
  python research/daily_tp_A.py
"""
import os, numpy as np, pandas as pd
pd.set_option("display.width", 220)
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "reports")
def load_main():
    t = pd.read_csv(f"{R}/trades_A_preopen.csv"); t = t[(t.selected == 1) & (t.filled == 1)].copy()
    t["per"] = np.select([t.date < "2026-08-20", t.date < "2026-10-01"], ["Aug1-19", "tune"], "test"); return t
def load_month(mon, lab):
    f = f"{R}/trades_A_{mon}.csv"
    if not os.path.exists(f): return None
    t = pd.read_csv(f)
    if "exit_rel" not in t: return None
    t["per"] = lab; return t
parts = [load_main()] + [x for x in (load_month("june", "June"), load_month("july", "July")) if x is not None]
t = pd.concat(parts, ignore_index=True); t["usd"] = t.pnl / t.entry_px * 100; t["t_exit"] = t["T"] + t.exit_rel.fillna(300)
t = t.sort_values("T").reset_index(drop=True)

def run(df, tp=None, sl=None):
    """per-day realised-P&L rule; returns the daily $ series and the share of days stopped"""
    out = {}; stopped = 0
    for d, g in df.groupby("date", sort=True):
        g = g.sort_values("T"); taken = []; stop = False
        for r in g.itertuples():
            if stop: continue
            realised = sum(x.usd for x in taken if x.t_exit <= r.T - 45)
            if (tp is not None and realised >= tp) or (sl is not None and realised <= -sl): stop = True; continue
            taken.append(r)
        stopped += stop; out[d] = sum(x.usd for x in taken)
    s = pd.Series(out); return s, stopped / max(len(out), 1)

def metrics(s):
    eq = s.cumsum(); return dict(per_day=s.mean(), total=s.sum(), sd=s.std(), sharpe=s.mean() / s.std() * np.sqrt(365) if s.std() > 0 else np.nan, worst=s.min(), maxdd=(eq.cummax() - eq).max(), neg_days=int((s < 0).sum()), days=len(s))
rows = []
for per, df in t.groupby("per", sort=False):
    base, _ = run(df)
    for lab, kw in [("no limit", {})] + [(f"TP +{x}", dict(tp=x)) for x in (100, 150, 200, 250, 300, 400, 500, 750, 1000)] + [(f"SL -{y}", dict(sl=y)) for y in (150, 200, 300, 400, 500)] + [(f"TP +{x} / SL -{y}", dict(tp=x, sl=y)) for x, y in ((300, 300), (400, 300), (500, 300), (500, 500), (750, 500))]:
        s, st = run(df, **kw); m = metrics(s); m.update(period=per, rule=lab, stopped_share=st, d_vs_base=(s - base).mean())
        rows.append(m)
df = pd.DataFrame(rows)
print("A only, $100 per trade (200 shares at 0.50), realised-P&L rule checked at each decision; per_day/total/worst/maxdd in $; d_vs_base = mean daily $ change vs no limit\n")
for per in df.period.unique():
    x = df[df.period == per]
    print(f"-- {per} ({int(x.days.iloc[0])} days) --")
    print(x[["rule", "per_day", "d_vs_base", "total", "sd", "sharpe", "worst", "maxdd", "neg_days", "stopped_share"]].round(2).to_string(index=False)); print()
# summary across periods: which TP level is best by total $ and by Sharpe
piv = df.pivot(index="rule", columns="period", values="per_day").round(0); piv["all_periods_mean"] = piv.mean(axis=1).round(0)
print("== mean $/day by rule and period =="); print(piv.to_string())
piv2 = df.pivot(index="rule", columns="period", values="sharpe").round(1); piv2["mean"] = piv2.mean(axis=1).round(1)
print("\n== daily Sharpe by rule and period =="); print(piv2.to_string())
df.round(3).to_csv(f"{R}/A_daily_tp.csv", index=False); print("\nsaved reports/A_daily_tp.csv")
