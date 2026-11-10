"""Live path vs research path for the new A model: the trader's own timeline (archive), Model.row(T, -45, 0) -> feature row -> NewA.score,
against the same booster applied to the research feature dump (scripts/dump_feats.py) for the same windows.
  python tests/test_newa_parity.py polybtc|polyeth [YYYY-MM-DD] [n_windows]"""
import os, sys, tempfile, datetime as dt, numpy as np, pandas as pd, glob
PKG = sys.argv[1] if len(sys.argv) > 1 else "polybtc"; DAY = sys.argv[2] if len(sys.argv) > 2 else "2026-11-06"; N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
ROOT = {"polybtc": "/home/user/polybtcmodel/polybtc", "polyeth": "/home/user/polyethmodel/polyeth"}[PKG]; ASSET = "BTC" if PKG == "polybtc" else "ETH"
os.environ["PAPER_DIR"] = tempfile.mkdtemp(prefix="parity_"); sys.path.insert(0, ROOT); import paper_trader as pt; from newa_live import NewA
d0 = dt.datetime.fromisoformat(DAY + "T00:00:00+00:00"); T0 = int(d0.timestamp()) - 12 * 86400
tl = pt.LiveTimeline(T0, days=14); pt.load_archive(tl); model = pt.Model(tl); na = NewA()
DEC = pt.A_DEC; D = pd.concat([pd.read_parquet(f) for f in glob.glob(f"/home/user/data/deep/feats_{ASSET}_{'' if DEC == -45 else f'd{DEC}_'}ph0_*.parquet")]).set_index("T")
rng = np.random.default_rng(5); Ts = sorted(int(d0.timestamp()) + 300 * int(k) for k in rng.choice(288, N, replace=False)); rows = []
for T in Ts:
    tl.known = (T + DEC - 7) - tl.t0_epoch; tl.last_spot_sec = T + DEC - 7; model.regime(T, DEC, 0)      # regime block precomputed 7 s before the decision, as live
    tl.known = (T + DEC) - tl.t0_epoch; tl.last_spot_sec = T + DEC; model.row(T, DEC, 0); i45 = (T + DEC) - tl.t0_epoch      # decision: bars through the second before it final + the spot kline of the decision second (its open is the row's reference price)
    s_live, d_live = na.score(model.last_X, tl.view("s_close", i45 - 15, i45)); r = D.loc[T]; s_ref = float(na.booster.predict(np.array([[float(r[f]) for f in na.feats]], dtype=np.float32))[0])
    worst = max(na.feats, key=lambda f: abs((d_live if f == "f_disp15" else float(model.last_X[f].values[0])) - float(r[f])) / (abs(float(r[f])) + 1e-3))
    rows.append((s_live, s_ref, d_live if DEC == -45 else 0.0, float(r["f_disp15"]) if DEC == -45 else 0.0, worst))
a = np.array([(x[0], x[1], x[2], x[3]) for x in rows]); ds = np.abs(a[:, 0] - a[:, 1])
print(f"{ASSET} {DAY}: {N} windows. score live vs research: max |diff| {ds.max():.5f}, mean {ds.mean():.5f} (score std {a[:,1].std():.4f}); side flips {(np.sign(a[:,0]) != np.sign(a[:,1])).sum()}; "
      f"trade-decision flips at |score| >= {na.thr:.4f}: {((np.abs(a[:,0]) >= na.thr) != (np.abs(a[:,1]) >= na.thr)).sum()}; f_disp15 max |diff| {np.abs(a[:,2]-a[:,3]).max():.5f}")
print("feature with the largest relative difference, by window (top 3):", pd.Series([x[4] for x in rows]).value_counts().head(3).to_dict())
ok = ds.max() < 0.02 and ((np.abs(a[:, 0]) >= na.thr) != (np.abs(a[:, 1]) >= na.thr)).sum() <= max(1, N // 20); print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
