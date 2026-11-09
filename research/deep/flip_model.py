"""A model built for the late-reversal question: P(the side that is currently behind wins) under the 60-s TWAP rule, from the Binance
state at t seconds after the open (t = 150, 180, 210, 240, 260).  State (all known at t): signed distance z of the price from what the
end TWAP needs, in units of the remaining volatility (exact for the part of the TWAP window already elapsed); time; volatility level and
ratios; momentum over 5-60 s, perp-spot basis and taker-flow imbalance, each signed toward a flip.  Trained on BTC windows Jan 2025 -
May 2026 (Binance label), then: calibration on Jun-Nov (BTC, and ETH as a transfer test), and against the Polymarket ask of the cheap
token at the same time (late_quotes.py).   python research/deep/flip_model.py"""
import os, sys, warnings, datetime as dt, numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250)
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); sys.path.insert(0, "/home/user/polybtcmodel/scripts"); from common import load, SRC, DEEP, auc  # noqa: E402
from dump_cross import TL, ffill  # noqa: E402
TS = [150, 180, 210, 240, 260]
def states(tl, d0, d1):
    out = []; d = d0
    while d <= d1:
        d2 = min(d + dt.timedelta(days=20), d1); a = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.UTC).timestamp()) - tl.t0; b = int(dt.datetime(d2.year, d2.month, d2.day, tzinfo=dt.UTC).timestamp()) + 86400 - tl.t0
        lo = a - 4000; hi = min(b + 400, tl.n)
        if lo < 0: d = d2 + dt.timedelta(days=1); continue
        c = ffill(tl.get("s_close", lo, hi)); lc = np.log(c); pc = ffill(tl.get("p_close", lo, hi)); cs = np.concatenate([[0.0], np.cumsum(c)]); r2 = np.concatenate([[0.0], np.cumsum(np.diff(lc, prepend=lc[0]) ** 2)])
        v = np.concatenate([[0.0], np.cumsum(np.nan_to_num(tl.get("s_volume", lo, hi)))]); tb = np.concatenate([[0.0], np.cumsum(np.nan_to_num(tl.get("s_tb_base", lo, hi)))]); basis = np.log(pc) - lc; cb = np.concatenate([[0.0], np.cumsum(basis)])
        T = np.arange((a + 299) // 300 * 300, min(b, tl.n - 400), 300); i0 = T - lo; mean = lambda u, w: (cs[w] - cs[u]) / (w - u)
        R = mean(i0 - 60, i0); E = mean(i0 + 240, i0 + 300); up = E >= R
        for t in TS:
            j = i0 + t - 1; S = c[j]; sig = np.sqrt((r2[j + 1] - r2[j + 1 - 300]) / 300 + 1e-14); s60 = np.sqrt((r2[j + 1] - r2[j + 1 - 60]) / 60 + 1e-14); s3600 = np.sqrt((r2[j + 1] - r2[j + 1 - 3600]) / 3600 + 1e-14)
            if t <= 240: need = np.log(S / R); sd = sig * np.sqrt((240 - t) + 20.5)
            else:
                k = t - 240; n = 60 - k; Ak = mean(i0 + 240, i0 + t); Ms = (60 * R - k * Ak) / n; need = np.where(Ms > 0, np.log(S / np.maximum(Ms, 1e-9)), 50.0); sd = sig * np.sqrt(n / 3 + 0.5)
            z = need / sd; lead = np.sign(z); f = {"T": tl.t0 + T, "t": t, "z": z, "flip": (up != (z > 0)).astype(np.int8), "absz": np.abs(z), "lsig": np.log(sig * 1e4), "vr_60_300": s60 / sig, "vr_300_3600": sig / s3600}
            for kk in (5, 10, 30, 60, 120): f[f"mom_{kk}"] = -lead * (lc[j] - lc[j - kk]) / (sig * np.sqrt(kk))             # > 0: moving toward a flip
            f["basis_dev"] = -lead * (basis[j] - (cb[j + 1] - cb[j + 1 - 300]) / 300) / (sig * np.sqrt(30))
            for kk in (10, 30): vv = v[j + 1] - v[j + 1 - kk]; f[f"flow_{kk}"] = -lead * (2 * (tb[j + 1] - tb[j + 1 - kk]) - vv) / (vv + 1e-12)
            f["since_cross"] = np.minimum(t, 300) * 0.0; out.append(pd.DataFrame(f))
        d = d2 + dt.timedelta(days=1)
    return pd.concat(out, ignore_index=True)
fb = f"{DEEP}/flip_states_BTC.parquet"; fe = f"{DEEP}/flip_states_ETH.parquet"
if not os.path.exists(fb): states(TL("/home/user/data"), dt.date(2025, 1, 3), dt.date(2026, 11, 7)).to_parquet(fb, index=False)
if not os.path.exists(fe): states(TL("/home/user/data_eth"), dt.date(2026, 5, 20), dt.date(2026, 11, 7)).to_parquet(fe, index=False)
B = pd.read_parquet(fb); Eh = pd.read_parquet(fe); FE = [c for c in B.columns if c not in ("T", "flip", "z", "since_cross")]
cut = int(pd.Timestamp("2026-05-25").timestamp()); tr = B[(B["T"] < cut) & (B.absz >= 0.8)]
m = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=20.0, verbose=-1, num_threads=4), lgb.Dataset(tr[FE].values, tr.flip.values), num_boost_round=400)
print(f"trained on {len(tr)} BTC states before 2026-05-25 with |z| >= 0.8; flip rate {tr.flip.mean():.3%};  importance:", dict(pd.Series(m.feature_importance('gain'), index=FE).sort_values(ascending=False).head(8).round(0)))
for nm, D in (("BTC", B[B["T"] >= int(pd.Timestamp("2026-06-01").timestamp())]), ("ETH", Eh[Eh["T"] >= int(pd.Timestamp("2026-06-01").timestamp())])):
    D = D[D.absz >= 0.8].copy(); D["p_flip"] = m.predict(D[FE].values); D["bucket"] = pd.cut(D.p_flip, [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 1.0])
    print(f"\n== {nm}, Jun-Nov 2026 (out of sample): calibration of P(flip) on the Binance TWAP label; AUC {auc(D.flip.values, D.p_flip.values):.3f} =="); print(D.groupby("bucket").agg(n=("flip", "size"), predicted=("p_flip", "mean"), realised=("flip", "mean")).round(4).to_string())
    W = load(nm, ["m2_-45_close_p"]); Q = pd.read_parquet(f"{DEEP}/latequotes_{nm}.parquet"); D = D.merge(Q, on=["T", "t"], how="inner"); D = D[D["T"].isin(W.index)]; up = W.up.reindex(D["T"]).values == 1
    D["price"] = np.where(D.z > 0, D.ask_dn, D.ask_up); D["usd"] = np.where(D.z > 0, D.usd_dn, D.usd_up); D["won"] = np.where(D.z > 0, ~up, up); D = D[np.isfinite(D.price) & (D.price <= 0.10)]
    D["twap"] = D["T"] >= pd.Timestamp(SRC[nm]["twap_from"], tz="UTC").timestamp(); D = D[D.twap]; D["fee"] = 0.07 * D.price * (1 - D.price); D["ret"] = D.won / (D.price + D.fee) - 1; D["ratio"] = D.p_flip / D.price
    D["rb"] = pd.cut(D.ratio, [0, 0.5, 1.0, 1.5, 2.0, 3.0, 1e9], labels=["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", ">3"]); D["pb"] = pd.cut(D.price, [0, 0.03, 0.05, 0.10], labels=["<=3c", "3-5c", "5-10c"])
    print(f"-- {nm}, TWAP-rule windows: buying the cheap token at its real ask ({len(D)} candidates), by price and model / price --")
    print(D.groupby(["pb", "rb"]).agg(n=("won", "size"), price_c=("price", lambda x: 100 * x.mean()), model_pct=("p_flip", lambda x: 100 * x.mean()), won_pct=("won", lambda x: 100 * x.mean()), ret_per_usd=("ret", "mean"),
                                       tstat=("ret", lambda x: x.mean() / (x.std() / np.sqrt(len(x)) + 1e-12)), usd_avail=("usd", "median")).round(2).to_string())
    D.drop(columns=["bucket", "rb", "pb"]).to_parquet(f"{DEEP}/flip_eval_{nm}.parquet", index=False)
