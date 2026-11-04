"""Pre-open mispricing ("B before the open"): between T-25 and T-5 the book has often left 0.50 on pre-open
flow while most of the 60 s reference window is already known. The close head re-evaluated at T+d
(data < T+d, effective window [T-30, T+270)) is compared with the last print before T+d; when they disagree
by >= delta we buy the model's side at the prevailing price (resting bid at the last print, cancelled after
10 s or at the open; or taker). Two regimes are reported separately:
  fade   : the market is tilted AGAINST the model's side (e.g. UP at 0.56 while the model says DOWN) -> we buy the cheap side
  follow : the market is tilted toward the model's side but less than the model
Grid over d, delta, a minimum market displacement, entry mode, exits. Selection on tune; test and Aug 1-19 reported.
Note: before Sept 2026 the markets settled on the point price, so Aug 1-19 is a different regime for this signal."""
import os, sys, numpy as np, pandas as pd, warnings, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
warnings.filterwarnings("ignore"); pd.set_option("display.width", 260)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
master, mk, g = ex.master, ex.mk, ex.g
TPS = (0.60, 0.65, 0.70, 0.80); SLS = (0.05, 0.10, 0.20); TSS = (240,)
rows = []
def metrics_for(pnl, ent, dates, per):
    s = (dates >= per[0]) & (dates <= per[1]); x = np.where(s, pnl, np.nan); mt = se.metrics(x, dates); f = s & np.isfinite(pnl)
    return dict(n=mt.get("n", 0), mean=mt.get("mean", np.nan), win=mt.get("win", np.nan), usd=float((pnl[f] / ent[f] * 100).sum()) if f.any() else 0.0, sharpe=mt.get("sharpe", np.nan), entry=float(np.nanmean(ent[f])) if f.any() else np.nan, tstat=mt.get("tstat", np.nan))
for d in (-25, -20, -15, -10, -5):
    col = f"m2_{d}_close_p"
    if col not in master: print("missing", col); continue
    for mode, kw in (("bid@last 10s", dict(entry_mode="maker_last", t_cancel=d + 10)), ("bid@last to open", dict(entry_mode="maker_last", t_cancel=0)), ("taker", dict(entry_mode="taker"))):
        sim = polysim.simulate(mk, g, t_entry=d, tps=TPS, sls=SLS, tstops=TSS, slip=0.01, **kw); m, sim = ex.align(master, sim)
        mod = m[col].values; mkt = sim["pre_last"][:, 0]; dis = mod - mkt; side = np.where(dis > 0, 0, 1); idx = np.arange(len(m)); dates = m.date.values
        tilt = mkt - 0.5                                               # market tilt toward UP
        fade = np.sign(dis) == -np.sign(tilt)                          # model's side is the cheap side
        for delta, mind, regime in itertools.product((0.06, 0.08, 0.10, 0.12, 0.15), (0.0, 0.03, 0.05, 0.08), ("all", "fade", "follow")):
            base = np.isfinite(dis) & (np.abs(dis) >= delta) & (np.abs(tilt) >= mind)
            if regime == "fade": base &= fade
            elif regime == "follow": base &= ~fade & (np.abs(tilt) > 0)
            if base.sum() < 100: continue
            for a, tp in enumerate(TPS):
                for b, sl in enumerate(SLS):
                    pnl = np.where(base, sim["pnl"][idx, side, a, b, 0], np.nan); ent = sim["entry_px"][idx, side]
                    r = dict(d=d, mode=mode, delta=delta, min_tilt=mind, regime=regime, tp=tp, sl=sl, signals_tune=int((base & (dates >= se.TUNE[0]) & (dates <= se.TUNE[1])).sum()))
                    for nm, per in (("old", se.OLD), ("tune", se.TUNE), ("test", se.TEST)):
                        mt = metrics_for(pnl, ent, dates, per); r.update({f"{k}_{nm}": v for k, v in mt.items()})
                    rows.append(r)
        print(f"d={d} {mode}: {len(rows)} rows so far", flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(R, "A_preopen_fade_grid.csv"), index=False)
ok = df[df.n_tune >= 150].copy(); ok["score"] = ok.mean_tune * np.sqrt(ok.n_tune)
cols = ["d", "mode", "regime", "delta", "min_tilt", "tp", "sl", "n_tune", "entry_tune", "mean_tune", "win_tune", "usd_tune", "n_test", "mean_test", "win_test", "usd_test", "tstat_test", "n_old", "mean_test"]
cols = ["d", "mode", "regime", "delta", "min_tilt", "tp", "sl", "n_tune", "entry_tune", "mean_tune", "win_tune", "usd_tune", "n_test", "mean_test", "win_test", "usd_test", "tstat_test", "n_old", "mean_old"]
print("\n== top 25 cells by tune score (mean * sqrt(n)), with their test and Aug 1-19 results =="); print(ok.sort_values("score", ascending=False).head(25)[cols].round(3).to_string(index=False))
for regime in ("fade", "follow"):
    sub = ok[ok.regime == regime]
    print(f"\n== {regime}: top 10 by tune score =="); print(sub.sort_values("score", ascending=False).head(10)[cols].round(3).to_string(index=False))
print("\n== the user's case: market tilted >= 0.05 against the model's side (fade), bid at last print 10 s, by d and delta, frozen-B exits 0.80/0.10 ==")
sub = df[(df.regime == "fade") & (df.min_tilt == 0.05) & (df.mode == "bid@last 10s") & (df.tp == 0.80) & (df.sl == 0.10)]
print(sub[cols].round(3).to_string(index=False))
print("\n== for reference, frozen B at the open (d=0, delta 0.12, bid@last 10 s, 0.80/0.10): see reports/summary.json (tune +4.5c / test +5.6c per share) ==")
