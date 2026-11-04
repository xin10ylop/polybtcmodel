"""Score the refit btcmovement2 models (walk-forward through 2026-07-31) at d=-45 for every window Aug-Nov,
and compare with the frozen April model on tune/test with the frozen A configuration."""
import os, sys, pickle, numpy as np, pandas as pd, warnings, time
os.environ["BTC_DATA_DIR"] = "/home/user/data"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
sys.path.insert(0, "/home/user/btcmovement2")
import score_m2 as S
from btcmodel.timeline import load_timeline
from btcmodel.predictor import Predictor, HEAD_H
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
TAG = sys.argv[1] if len(sys.argv) > 1 else "_wf0731"
tl = load_timeline()
models = {h: pickle.load(open(f"/home/user/data/models/models_{h}_valid{TAG}.pkl", "rb")) for h in HEAD_H}
season = pickle.load(open("/home/user/data/cache/season.pkl", "rb"))
pr = Predictor(models, season)
from datetime import date, timedelta
parts = []; d = date(2026, 8, 1)
while d <= date(2026, 11, 3):
    d2 = min(d + timedelta(days=5), date(2026, 11, 3)); t = time.time()
    c0, c1 = tl.index_of(d), tl.index_of(d2) + 86400
    Tw = np.arange((c0 + 299) // 300 * 300, c1, 300); Tw = Tw[(Tw + 360 <= tl.n)]
    o = S.score_points(tl, pr, Tw, -45, [0], c0, c1); o = o[["T", "d", "close_p", "h60_pup_now", "h30_pup_now"]]; parts.append(o)
    print(d, d2, len(o), f"{time.time()-t:.0f}s", flush=True); d = d2 + timedelta(days=1)
new = pd.concat(parts, ignore_index=True).set_index("T"); new.columns = [f"m2new_-45_{c}" if c != "d" else "d" for c in new.columns]
new.drop(columns="d").to_parquet(f"/home/user/data/forecasts/m2new{TAG}_m45.parquet")
import polysim, strategy_eval as se, experiments as ex  # noqa: E402
TUNE, TEST = se.TUNE, se.TEST
m = ex.master.join(new.drop(columns="d"), how="left")
for c in ("m2_-45_close_p", "m2new_-45_close_p"):
    for split, nm in ((TUNE, "tune"), (TEST, "test")):
        s = (m.date.values >= split[0]) & (m.date.values <= split[1]) & np.isfinite(m[c].values)
        print(f"{c:20s} {nm}: AUC {roc_auc_score(m.up.values[s], m[c].values[s]):.4f} n={s.sum()}")
sim = polysim.simulate(ex.mk, ex.g, t_entry=-45, entry_mode="fixed", pe=0.50, tps=(0.80,), sls=(0.20,), tstops=(240,))
mm, sim = ex.align(m, sim); idx = np.arange(len(mm)); agree = np.sign(mm["bp_-60_z_s5"].values)
rows = []
for c in ("m2_-45_close_p", "m2new_-45_close_p"):
    p = mm[c].values; side = np.where(p > 0.5, 0, 1); pnl = sim["pnl"][idx, side, 0, 0, 0]
    for thr in (0.04, 0.05, 0.06):
        base = np.isfinite(p) & (np.abs(p - 0.5) >= thr) & (agree == np.sign(p - 0.5)); r = {"signal": c, "thr": thr}
        for split, sp in ((TUNE, "tune"), (TEST, "test")):
            s = base & (mm.date.values >= split[0]) & (mm.date.values <= split[1]); mt = se.metrics(np.where(s, pnl, np.nan), mm.date.values)
            r.update({f"{k}_{sp}": mt.get(k, np.nan) for k in ("n", "mean", "win", "sharpe", "tstat")})
        rows.append(r)
print(pd.DataFrame(rows).round(4).to_string(index=False))
