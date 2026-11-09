"""Deep dive 1: where A's win rate comes from, and what each model component is worth at the decision time (BTC and ETH).
 1. win-rate decomposition of the frozen A trades: direction accuracy, P(sell limit hit | right / wrong), the zero-edge win rate 0.5/TP.
 2. AUC vs the Polymarket outcome by month of every component available at T-45.
 3. marginal edge (settle-only, i.e. pure direction, cents per share) by confidence bucket of the m2 head, with / without the second model agreeing.
 4. cross-asset: the other asset's heads as a vote on this asset's A trades."""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims, auc, PER  # noqa: E402
OFFS = (-120, -90, -75, -60, -50, -45); COMP = ["close_p", "close_pm", "close_pd", "close_mu", "h60_p", "h30_p", "h10_p", "h60_mu", "h30_mu", "h10_mu", "r_10", "r_30", "pimb_10", "bdev_60", "bdev_300", "prev_ret_v"]
cols = [f"m2_{o}_close_p" for o in OFFS] + [f"m2_-45_{c}" for c in COMP] + [f"bp_{o}_z_{k}" for o in (-120, -60) for k in ("s5", "s1", "s2", "s10s", "s15s", "f1")] + ["x_eff_bps", "vol_prev5m_bp"]
D = {a: load(a, cols) for a in ("BTC", "ETH")}; TPX = {"BTC": 0.85, "ETH": 0.80}
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
for asset, W in D.items():
    print(f"\n{'=' * 30} {asset} {'=' * 30}")
    y = W.up.values; p = W["m2_-45_close_p"].values; z5 = W["bp_-60_z_s5"].values; side = np.where(p > 0.5, 1, -1); right = (y == 1) == (p > 0.5)
    selA = np.isfinite(p) & (np.abs(p - 0.5) >= 0.05) & np.isfinite(z5) & (np.sign(z5) == side)
    S = sims(asset, TPX[asset]).reindex(W.index); pnl = np.where(p > 0.5, S.pnl_up.values, S.pnl_dn.values); tp = np.where(p > 0.5, S.tp_up.values, S.tp_dn.values).astype(float)
    print(f"\n-- 1. frozen A (thr 0.05 + second model agrees), sell {TPX[asset]} else settle: decomposition (zero-edge win rate = {0.5 / TPX[asset]:.3f}) --")
    rows = []
    for per in PER:
        m = selA & (W.period.values == per) & np.isfinite(pnl)
        rows.append(dict(period=per, n=int(m.sum()), dir_acc=right[m].mean(), tp_if_right=tp[m & right].mean(), tp_if_wrong=tp[m & ~right].mean(), win=(pnl[m] > 0).mean(), edge_c=pnl[m].mean() * 100,
                         settle_only_c=(right[m].mean() - 0.5) * 100))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\n-- 2. AUC vs the outcome by month, components at the decision time --")
    comp = {f"m2 close head @{o}": W[f"m2_{o}_close_p"].values for o in OFFS}
    comp["m2 close head mean(-60,-50,-45)"] = np.nanmean([W[f"m2_{o}_close_p"].values for o in (-60, -50, -45)], axis=0)
    comp["m2 close head mean(-120..-45)"] = np.nanmean([W[f"m2_{o}_close_p"].values for o in OFFS], axis=0)
    for c in COMP[1:]: comp[f"m2@-45 {c}"] = W[f"m2_-45_{c}"].values
    for c in W.columns:
        if c.startswith("bp_"): comp[c.replace("bp_", "second model ")] = W[c].values
    lz = lambda q: np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
    comp["m2 logit + 0.5*z5 (fixed mix)"] = lz(p) + 0.5 * np.where(np.isfinite(z5), z5, 0)
    rows = []
    for nm, v in comp.items():
        r = {"component": nm}
        for mo in MONTHS: mm = W.month.values == mo; r[mo] = auc(y[mm], v[mm])
        r["all"] = auc(y, v); rows.append(r)
    print(pd.DataFrame(rows).set_index("component").round(3).to_string())
    print("\n-- 3. settle-only edge (cents per share = 100*(dir accuracy - 0.5)) by confidence bucket of the m2 head; second model agree / disagree --")
    edges = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.5]; conf = np.abs(p - 0.5); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        for lab, cond in (("agree", np.sign(z5) == side), ("disagree", np.sign(z5) == -side)):
            r = {"bucket": f"{lo:.2f}-{hi:.2f}", "second": lab}
            for per in list(PER) + ["ALL"]:
                m = np.isfinite(p) & (conf >= lo) & (conf < hi) & np.isfinite(z5) & cond & ((W.period.values == per) if per != "ALL" else True)
                r[per] = f"{(right[m].mean() - 0.5) * 100:+.1f} ({int(m.sum())})" if m.sum() >= 30 else "-"
            rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    other = "ETH" if asset == "BTC" else "BTC"; O = D[other].reindex(W.index); po = O["m2_-45_close_p"].values; zo = O["bp_-60_z_s5"].values
    print(f"\n-- 4. cross-asset vote on the frozen A trades: {other}'s heads agree / disagree with the side (win / edge cents / n) --")
    rows = []
    for lab, v in ((f"{other} m2 head sign", np.sign(po - 0.5)), (f"{other} second-model z5 sign", np.sign(zo)), (f"{other} both", np.where(np.sign(po - 0.5) == np.sign(zo), np.sign(zo), 0))):
        for ag, cond in (("agrees", v == side), ("disagrees", v == -side)):
            r = {"vote": f"{lab} {ag}"}
            for per in PER:
                m = selA & (W.period.values == per) & np.isfinite(pnl) & cond
                r[per] = f"{(pnl[m] > 0).mean():.3f} / {pnl[m].mean() * 100:+.1f} / {int(m.sum())}" if m.sum() >= 20 else "-"
            rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
