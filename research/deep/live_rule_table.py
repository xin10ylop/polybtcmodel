"""P&L of the rule the bots run (GO_LIVE.md): purpose-built model WITHOUT book inputs (ourcase_<asset>_ph5, lgb_reg300), trade the top 15 % of
|score| (threshold from the months before), buy at 0.50 (assumed filled), sell 0.85 (BTC) / 0.80 (ETH) else settle.  BTC $100, ETH $25 per trade.
Next to the old rule A.  python research/deep/live_rule_table.py"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, capacity, DEEP
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; STAKE = {"BTC": 100, "ETH": 25}; TPX = {"BTC": 0.85, "ETH": 0.80}; recs = []
for ASSET in ("BTC", "ETH"):
    W = load(ASSET, ["m2_-45_close_p", "bp_-60_z_s5"]); S = sims_cached(ASSET, TPX[ASSET]).reindex(W.index); mon = W.month.values; n = len(W); cap = capacity(ASSET).reindex(W.index).fillna(0).values
    sc = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_ph5.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values
    def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
    sel = np.zeros(n, bool)
    for mo in MONTHS:
        te = (mon == mo) & np.isfinite(sc); sel[te] = np.abs(sc[te]) >= np.quantile(np.abs(sc[before(mo) & np.isfinite(sc)]), 0.85)
    pm = W["m2_-45_close_p"].values; z5 = W["bp_-60_z_s5"].values
    for nm, (s_, upside) in {"old A": (np.isfinite(pm) & (np.abs(pm - 0.5) >= 0.05) & np.isfinite(z5) & (np.sign(z5) == np.sign(pm - 0.5)), pm > 0.5), "new A (live model)": (sel, sc > 0)}.items():
        pnl = np.where(upside, S.pnl_up.values, S.pnl_dn.values); m = s_ & np.isfinite(pnl)
        recs.append(pd.DataFrame({"T": W.index.values[m], "date": W.date.values[m], "month": mon[m], "asset": ASSET, "rule": nm, "win": pnl[m] > 0, "usd": pnl[m] * 2 * STAKE[ASSET], "usd_cap": pnl[m] * 2 * np.minimum(STAKE[ASSET], cap[m])}))
R = pd.concat(recs); days = pd.date_range("2026-06-01", "2026-11-07"); R.to_parquet(f"{DEEP}/live_rule_trades.parquet", index=False)
def stats(g, col="usd"):
    g = g.sort_values("T"); eq = np.concatenate([[0], g[col].cumsum().values]); d = g.groupby("date")[col].sum().reindex(days, fill_value=0.0)
    return f"trades {len(g):5d}  win {g.win.mean():.1%}  total ${g[col].sum():>8,.0f}  per day ${g[col].sum() / len(days):>4,.0f}  max drawdown ${(np.maximum.accumulate(eq) - eq).max():>6,.0f}  worst day ${d.min():>7,.0f}  losing days {int((d < 0).sum())}/160"
for rule in ("old A", "new A (live model)"):
    g = R[R.rule == rule]; print(f"\n=== {rule} (every order filled at 0.50) ===")
    mt = g.pivot_table(index="month", columns="asset", values="usd", aggfunc="sum").reindex(MONTHS); mt["account"] = mt.sum(1); print(mt.round(0).to_string())
    for scope, gg in (("BTC $100", g[g.asset == "BTC"]), ("ETH $25", g[g.asset == "ETH"]), ("ACCOUNT", g)): print(f"   {scope:9s} {stats(gg)}")
    print(f"   ACCOUNT, stakes limited to what traded near 0.50 before the open: {stats(g, 'usd_cap')}")
