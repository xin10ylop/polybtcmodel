"""Protecting A from regime changes: (1) principled detectors available at decision time: the perp->spot lead-lag beta over the
last hour (reg_ll_beta_1h), the catch-up speed (reg_catchup_1h), the 1-day variance ratio (reg_vr300_1d) and the 1-day/7-day
vol ratio (reg_rv_1d_7d); A's win rate and edge by quintile of each (quintiles from the tune period), on every period incl. June,
July and Nov 4-7.  (2) performance gates: pause A after a rolling-N win rate below X, keep scoring on paper, resume when the paper
win rate is back above Y.  python research/regime_gate_A.py"""
import os, numpy as np, pandas as pd
pd.set_option("display.width", 230); R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
REG = ["reg_ll_beta_1h", "reg_catchup_1h", "reg_vr300_1d", "reg_rv_1d_7d", "reg_trend_1d", "lrv_3600"]
def load(f, master, lab=None):
    t = pd.read_csv(f)
    if "selected" in t: t = t[(t.selected == 1) & (t.filled == 1)].copy()
    m = pd.read_parquet(master, columns=[f"m2_-45_{c}" for c in REG]); m.columns = REG
    t = t.join(m, on="T"); t["per"] = lab if lab else np.select([t.date < "2026-08-20", t.date < "2026-10-01"], ["Aug1-19", "tune"], "test"); return t
t = pd.concat([load(f"{R}/trades_A_preopen.csv", "/home/user/data/master.parquet"), load(f"{R}/trades_A_june.csv", "/home/user/data/master_june.parquet", "June"),
               load(f"{R}/trades_A_july.csv", "/home/user/data/master_july_L0.parquet", "July"), load(f"{R}/trades_A_november.csv", "/home/user/data/master_nov.parquet", "Nov4-7")], ignore_index=True)
t = t.sort_values("T").reset_index(drop=True); t["usd"] = t.pnl / t.entry_px * 100; t["win"] = (t.pnl > 0).astype(int)
PERS = ["Aug1-19", "tune", "test", "June", "July", "Nov4-7"]
print("== A's win rate by quintile of each regime feature (quintile edges from the tune period) ==")
for c in REG:
    edges = t.loc[t.per == "tune", c].quantile([.2, .4, .6, .8]).values
    t[f"q_{c}"] = np.searchsorted(edges, t[c].values)
    piv = t.pivot_table(index=f"q_{c}", columns="per", values="win", aggfunc="mean").reindex(columns=PERS).round(2)
    piv["all"] = t.groupby(f"q_{c}").win.mean().round(3); piv["n"] = t.groupby(f"q_{c}").size()
    print(f"\n-- {c}: quintile edges {np.round(edges, 4).tolist()} --"); print(piv.to_string())
print("\n== performance gates: trade only while the rolling win rate of the last N A signals (traded or paper) is >= X; resume when it is back >= Y ==")
def gate(df, N, X, Y):
    w = df.win.values; on = True; taken = np.zeros(len(w), bool); hist = []
    for i in range(len(w)):
        rate = np.mean(hist[-N:]) if len(hist) >= N else 1.0
        if on and rate < X: on = False
        elif not on and rate >= Y: on = True
        taken[i] = on; hist.append(w[i])
    return taken
rows = []
for per in PERS:
    d = t[t.per == per]; base = d.groupby("date").usd.sum()
    rows.append(dict(per=per, gate="none", usd_day=round(base.mean()), win=round(d.win.mean(), 3), traded=1.0))
    for N, X, Y in ((20, .45, .60), (30, .50, .62), (30, .45, .60), (50, .52, .62), (15, .40, .60)):
        tk = gate(d, N, X, Y); x = d[tk]; s = x.groupby("date").usd.sum().reindex(base.index, fill_value=0)
        rows.append(dict(per=per, gate=f"N{N} off<{X} on>={Y}", usd_day=round(s.mean()), d_vs_none=round((s - base).mean()), win=round(x.win.mean(), 3) if len(x) else np.nan, traded=round(tk.mean(), 2)))
g = pd.DataFrame(rows); print(g.pivot(index="gate", columns="per", values=["usd_day", "win", "traded"]).reindex(columns=pd.MultiIndex.from_product([["usd_day", "win", "traded"], PERS])).to_string())
g.to_csv(f"{R}/A_regime_gate.csv", index=False); print("\nsaved reports/A_regime_gate.csv")
