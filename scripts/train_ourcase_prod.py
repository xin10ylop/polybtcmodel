"""Production boosters of the purpose-built model (research/deep/ourcase_model.py: engine features at the decision time + book depth,
standardized TWAP-move regression, 300 rounds), trained on EVERYTHING available, for the decision times used by the real-fill rule.
Saved to models/ourcase_<asset>_d<D>.txt with a json: feature order, the scale regression, and the |score| quantiles that define the
top 5 % / 15 % tiers (from the walk-forward out-of-sample scores of the last three months, i.e. not from this in-sample fit).
  ASSET=BTC python scripts/train_ourcase_prod.py -120 -90 -45"""
import os, sys, glob, json, numpy as np, pandas as pd, lightgbm as lgb
ASSET = os.environ.get("ASSET", "BTC"); BOOK = os.environ.get("BOOK", "1") == "1"      # BOOK=0: engine features only = the LIVE model (the 1-5 % depth bands cannot be rebuilt from a live feed)
DEEP = "/home/user/data/deep"; OUT = "/home/user/polybtcmodel/models" if ASSET == "BTC" else "/home/user/polyethmodel/models"; os.makedirs(OUT, exist_ok=True)
LAB = ["T", "phase", "y_twap", "x_twap", "y_pt", "x_pt", "x_dec", "price"]; DROP = ("e", "x_bps", "x_v", "wmax_v", "wmin_v", "ncross", "fup", "tsc")
bd_dir = "/home/user/data/extra/bookdepth" if ASSET == "BTC" else "/home/user/data_eth/extra/bookdepth"
bd = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{bd_dir}/*.parquet"))]).sort_index(); bd = bd[~bd.index.duplicated()]
for lv in ("0.2", "1.0", "2.0", "5.0"): bd[f"bk_imb{lv}"] = (bd[f"n-{lv}"] - bd[f"n+{lv}"]) / (bd[f"n-{lv}"] + bd[f"n+{lv}"])
bd["bk_outer"] = ((bd["n-1.0"] - bd["n-0.2"]) - (bd["n+1.0"] - bd["n+0.2"])) / ((bd["n-1.0"] - bd["n-0.2"]) + (bd["n+1.0"] - bd["n+0.2"]))
bd["bk_ldepth"] = np.log(bd["n-1.0"] + bd["n+1.0"]); bd["bk_ldepth_rel"] = bd.bk_ldepth - bd.bk_ldepth.rolling(2880, min_periods=200).median()
BK = ["bk_imb0.2", "bk_imb1.0", "bk_imb2.0", "bk_imb5.0", "bk_outer", "bk_ldepth_rel"]; idx = bd.index.values.astype(np.int64)
def asof(when):
    j = np.searchsorted(idx, when, side="right") - 1; ok = j >= 0; v = np.where(ok[:, None], bd[BK].values[np.maximum(j, 0)], np.nan); v[np.where(ok, when - idx[np.maximum(j, 0)], 1e9) > 120] = np.nan; return v
for DEC in [int(a) for a in sys.argv[1:]]:
    dtag = "" if DEC == -45 else f"d{DEC}_"; files = sorted(glob.glob(f"{DEEP}/feats_{ASSET}_{dtag}ph*.parquet")); D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    FEATS = [c for c in D.columns if c not in LAB and D[c].std() > 0 and c not in DROP]; now = asof(D["T"].values + DEC - 2); ago = asof(D["T"].values + DEC - 302)
    if BOOK:
        for k, c in enumerate(BK): D[c] = now[:, k].astype(np.float32)
        D["bk_dimb1.0_300"] = (now[:, 1] - ago[:, 1]).astype(np.float32); FEATS += BK + ["bk_dimb1.0_300"]
    x = D.x_twap.values; A = np.column_stack([np.ones(len(D)), D["lrv_300"].values, D["lrv_3600"].values]); coef = np.linalg.lstsq(A, np.log(np.abs(x) + 0.3), rcond=None)[0]; z = np.clip(x / np.exp(A @ coef), -4, 4)
    b = lgb.train(dict(objective="huber", alpha=1.5, learning_rate=0.02, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0, max_bin=63, verbose=-1, num_threads=3, seed=1),
                  lgb.Dataset(D[FEATS].values.astype(np.float32), z), num_boost_round=300)
    oos = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_{dtag}{('ph5book' if DEC == -45 else 'ph3book') if BOOK else ('ph5' if DEC == -45 else 'ph3')}.parquet"); a = np.abs(oos[oos.ym >= "2026-08"]["lgb_reg300"].dropna().values)
    meta = dict(asset=ASSET, decision_offset=DEC, rows=int(len(D)), phases=sorted(int(p) for p in D.phase.unique()), trained_through=str(pd.to_datetime(D["T"].max(), unit="s")), features=FEATS, scale_coef=[float(c) for c in coef],
                tier_abs_score={"top5": float(np.quantile(a, 0.95)), "top15": float(np.quantile(a, 0.85)), "top30": float(np.quantile(a, 0.70))}, tier_source="walk-forward OOS scores 2026-08..2026-11",
                params="lgb huber 300 rounds, lr 0.02, 15 leaves, min_data 400, ff 0.5, bag 0.7, l2 50")
    name = f"ourcase_{ASSET}_d{DEC}" + ("" if BOOK else "_live"); b.save_model(f"{OUT}/{name}.txt"); json.dump(meta, open(f"{OUT}/{name}.json", "w"), indent=1)
    print(f"{ASSET} D={DEC}: {len(D)} rows, {len(FEATS)} features, tiers {meta['tier_abs_score']}, saved", flush=True)
