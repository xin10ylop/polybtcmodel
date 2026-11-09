"""How much of each rule's edge survives a less generous entry than 'always filled at 0.50 before T-45'?
  fixed      : the assumption used everywhere (filled at 0.50)
  lastprint  : pay the last print of our side before T-45 (what the market showed)
  maker 5s / 15s / 45s : a 0.50 bid resting from T-45, cancelled at T-40 / T-30 / T; trades only if a print of our side at <= 0.50 occurs
  taker      : buy the first print after T-45 + 1 cent + taker fee
Exits unchanged (sell limit, else settle).  ASSET=BTC python research/deep/fill_sensitivity.py"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, sims_mode, PER, DEEP  # noqa: E402
ASSET = os.environ.get("ASSET", "BTC"); TP = {"BTC": 0.85, "ETH": 0.80}[ASSET]
W = load(ASSET, ["m2_-45_close_p", "bp_-60_z_s5"]); pm = W["m2_-45_close_p"].values; z5 = W["bp_-60_z_s5"].values; per = W.period.values
RULES = {"FROZEN A": (np.isfinite(pm) & (np.abs(pm - 0.5) >= 0.05) & np.isfinite(z5) & (np.sign(z5) == np.sign(pm - 0.5)), np.sign(pm - 0.5))}
RULES.update({k: v for k, v in pd.read_pickle(f"{DEEP}/deep_rules_{ASSET}.pkl").items() if k.startswith("new |") and "bk" not in k})
MODES = {"fixed 0.50": None, "last print": dict(entry_mode="lastprint"), "maker 0.50, 5 s": dict(entry_mode="maker", t_cancel=-40, bid=0.50), "maker 0.50, 15 s": dict(entry_mode="maker", t_cancel=-30, bid=0.50),
         "maker 0.50, 45 s": dict(entry_mode="maker", t_cancel=0, bid=0.50), "maker 0.51, 15 s": dict(entry_mode="maker", t_cancel=-30, bid=0.51), "taker +1c +fee": dict(entry_mode="taker", entry_fee=True)}
rows = []
for mn, kw in MODES.items():
    S = (sims_cached(ASSET, TP) if kw is None else sims_mode(ASSET, TP, mn.replace(" ", "").replace(",", "_").replace("+", "p"), **kw)).reindex(W.index)
    for rn, (sel, side) in RULES.items():
        pnl = np.where(side > 0, S.pnl_up.values, S.pnl_dn.values); ent = np.where(side > 0, S.entry_up.values, S.entry_dn.values) if "entry_up" in S else np.full(len(W), 0.5)
        for p_ in list(PER) + ["ALL"]:
            m0 = sel & ((per == p_) if p_ != "ALL" else (per != "")); m = m0 & np.isfinite(pnl); nd = len(pd.date_range(*PER[p_])) if p_ != "ALL" else 160
            if m.sum() < 10: continue
            rows.append(dict(rule=rn, mode=mn, period=p_, signals=int(m0.sum()), fill_rate=m.sum() / max(m0.sum(), 1), entry=np.nanmean(ent[m]), win=(pnl[m] > 0).mean(), edge_c=pnl[m].mean() * 100, usd100_day=pnl[m].sum() / np.nanmean(ent[m]) * 100 / nd))
E = pd.DataFrame(rows); E.to_csv(f"/home/user/polybtcmodel/reports/deep_fill_sensitivity_{ASSET}.csv", index=False)
for rn in RULES:
    print(f"\n=== {rn} ===")
    a = E[(E.rule == rn) & (E.period == "ALL")].set_index("mode")[["signals", "fill_rate", "entry", "win", "edge_c", "usd100_day"]]; print(a.round(3).to_string())
    print("edge (cents) by period:"); print(E[E.rule == rn].pivot(index="mode", columns="period", values="edge_c").reindex(list(MODES))[list(PER)].round(1).to_string())
