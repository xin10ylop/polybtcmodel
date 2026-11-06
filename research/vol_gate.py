"""Volatility gating: trade A only in calm windows, B / D only in volatile ones? vol_prev5m_bp (realised vol of the 5 minutes
before the window, known at decision time) is attached to every backtest trade; thresholds are the tune-period terciles /
quintiles, read on Aug 1-19, test, June and July. $100 per trade, current exits.  python research/vol_gate.py"""
import os, numpy as np, pandas as pd
pd.set_option("display.width", 230); R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
def main(name):
    t = pd.read_csv(f"{R}/trades_{name}.csv"); t = t[(t.selected == 1) & (t.filled == 1)].copy()
    t["per"] = np.select([t.date < "2026-08-20", t.date < "2026-10-01"], ["Aug1-19", "tune"], "test"); return t
def month(st, mon, lab, master):
    f = f"{R}/trades_{st}_{mon}.csv"
    if not os.path.exists(f): return None
    t = pd.read_csv(f); m = pd.read_parquet(master, columns=["vol_prev5m_bp"]); t["vol_prev5m_bp"] = t["T"].map(m.vol_prev5m_bp); t["per"] = lab; return t
S = {"A": ("A_preopen", "A"), "B": ("B_open_mispricing", "B"), "D": ("D_high_confidence_settle", "D")}
data = {}
for st, (main_name, short) in S.items():
    parts = [main(main_name)]
    for mon, lab, mp in (("june", "June", "/home/user/data/master_june.parquet"), ("july", "July", "/home/user/data/master_july_L0.parquet")):
        x = month(short, mon, lab, mp)
        if x is not None: parts.append(x)
    d = pd.concat(parts, ignore_index=True); d["usd"] = d.pnl / d.entry_px * 100; data[st] = d
tune_vol = data["A"][data["A"].per == "tune"].vol_prev5m_bp
q = {k: float(tune_vol.quantile(v)) for k, v in (("p20", .2), ("p33", 1 / 3), ("p50", .5), ("p67", 2 / 3), ("p80", .8))}
print("thresholds from the tune period's windows (vol_prev5m_bp):", {k: round(v, 1) for k, v in q.items()})
PERS = ["Aug1-19", "tune", "test", "June", "July"]
print("\n== P&L per trade (cents/share) and trades/day by volatility tercile of the tune period (calm < %.1f bp < mid < %.1f bp < volatile)" % (q["p33"], q["p67"]))
for st, d in data.items():
    d["vq"] = pd.cut(d.vol_prev5m_bp, [-1, q["p33"], q["p67"], 1e9], labels=["calm", "mid", "volatile"])
    g = d.groupby(["per", "vq"], observed=True).agg(n=("pnl", "size"), cents=("pnl", lambda x: 100 * x.mean()), win=("pnl", lambda x: (x > 0).mean()), usd=("usd", "sum"))
    g["usd_per_day"] = g.usd / d.groupby("per").date.nunique().reindex(g.index.get_level_values(0)).values
    print(f"\n-- {st} --"); print(g.round(2).unstack("vq").reindex(PERS).to_string())
def daily(d, mask, per):
    x = d[mask & (d.per == per).values]; return x.groupby("date").usd.sum().reindex(sorted(d[d.per == per].date.unique()), fill_value=0.0)
def summ(s): eq = s.cumsum(); return dict(per_day=round(s.mean()), total=round(s.sum()), sharpe=round(s.mean() / s.std() * np.sqrt(365), 1) if s.std() > 0 else np.nan, maxdd=round((eq.cummax() - eq).max()), worst=round(s.min()))
print("\n== gates (daily $ at $100/trade; 'd' = change in $/day vs ungated) ==")
rows = []
for st, d in data.items():
    gates = {"ungated": np.ones(len(d), bool)}
    if st == "A":
        for k in ("p80", "p67", "p50"): gates[f"A only if vol < {k} ({q[k]:.1f}bp)"] = d.vol_prev5m_bp.values < q[k]
    else:
        for k in ("p20", "p33", "p50"): gates[f"{st} only if vol > {k} ({q[k]:.1f}bp)"] = d.vol_prev5m_bp.values > q[k]
    base = {p: daily(d, np.ones(len(d), bool), p) for p in PERS}
    for gname, gm in gates.items():
        r = {"strategy": st, "gate": gname}
        for p in PERS:
            s = daily(d, gm, p); m = summ(s); r[f"{p} $/day"] = m["per_day"]; r[f"{p} d"] = round((s - base[p]).mean()); r[f"{p} sharpe"] = m["sharpe"]; r[f"{p} dd"] = m["maxdd"]
        rows.append(r)
out = pd.DataFrame(rows); print(out.to_string(index=False)); out.to_csv(f"{R}/vol_gate.csv", index=False); print("saved reports/vol_gate.csv")
