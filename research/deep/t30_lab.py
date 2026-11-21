"""Decision at T-30 (the user's question): a purpose-built model at T-30 trained walk-forward exactly like the T-120 one (scripts/dump_feats.py with DEC=-30,
3 grid phases, LightGBM huber on the vol-standardised TWAP move, monthly retraining from Oct 2025), plus the part of the start average already known
at T-30 (disp30 = close of second T-31 against the mean of seconds T-60..T-31).  Then:
 1) raw accuracy of the top-30 % signals at T-30 / T-45 / T-120 on the same windows, trading hours;
 2) what the market already shows at that moment (our side's price);
 3) fills on the real Polymarket tape: resting bid 0.50 from T-29 to T-15 (maker), or buying the offer up to a cap (taker, fee included).
  python research/deep/t30_lab.py"""
import sys, glob, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/btcmovement2"); from btcmodel.timeline import load_timeline
DEEP = "/home/user/data/deep"; OUT = f"{DEEP}/ourcase_BTC_d-30_ph3.parquet"; import os
if not os.path.exists(OUT):
    D = pd.concat([pd.read_parquet(f) for ph in (0, 120, 240) for f in glob.glob(f"{DEEP}/feats_BTC_d-30_ph{ph}_*.parquet")], ignore_index=True).sort_values("T").reset_index(drop=True)
    tl = load_timeline(); t0 = int(tl.t0_epoch); c = pd.Series(np.asarray(tl["s_close"], dtype=np.float64)).ffill().bfill().values; cs = np.concatenate([[0.0], np.cumsum(c)]); i = D["T"].values.astype(np.int64) - t0
    D["disp30"] = (1e4 * np.log(c[i - 31] / ((cs[i - 30] - cs[i - 60]) / 30.0))).astype(np.float32)
    LAB = ["T", "phase", "y_twap", "x_twap", "y_pt", "x_pt", "x_dec", "price", "f_disp15"]; F = [k for k in D.columns if k not in LAB and k not in ("e", "x_bps", "x_v", "wmax_v", "wmin_v", "ncross", "fup", "tsc") and D[k].std() > 0]
    X = D[F].values.astype(np.float32); x = D.x_twap.values; Tt = D["T"].values; D["ym"] = pd.to_datetime(D["T"], unit="s").dt.strftime("%Y-%m"); P = np.full(len(D), np.nan)
    LGB = dict(learning_rate=0.02, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0, max_bin=63, verbose=-1, num_threads=4, objective="huber", alpha=1.5)
    for mo in [m for m in sorted(D.ym.unique()) if m >= "2025-10"]:
        m0 = int(pd.Timestamp(mo + "-01").timestamp()); tr = Tt + 300 < m0 - 86400; te = (D.ym.values == mo) & (D.phase.values == 0)
        A = np.column_stack([np.ones(len(D)), D.lrv_300.values, D.lrv_3600.values]); coef = np.linalg.lstsq(A[tr], np.log(np.abs(x[tr]) + 0.3), rcond=None)[0]; z = np.clip(x / np.exp(A @ coef), -4, 4)
        P[te] = lgb.train(LGB, lgb.Dataset(X[tr], z[tr]), 300).predict(X[te]); print(f"  T-30 {mo}: train {int(tr.sum())}", flush=True)
    k = (D.phase.values == 0) & np.isfinite(P); pd.DataFrame({"T": Tt[k], "ym": D.ym.values[k], "x_twap": x[k], "lgb_reg300": P[k]}).to_parquet(OUT, index=False); print("features:", len(F), "(disp30 included:", "disp30" in F, ")")
S = {d: pd.read_parquet(f).set_index("T")["lgb_reg300"] for d, f in ((-30, OUT), (-45, f"{DEEP}/ourcase_BTC_ph5.parquet"), (-120, f"{DEEP}/ourcase_BTC_d-120_ph3.parquet"))}
B = pd.read_parquet(OUT).set_index("T")[["ym", "x_twap"]].join(pd.DataFrame({f"s{-d}": s for d, s in S.items()}), how="inner").dropna(); B = B[B.x_twap != 0]; t = pd.to_datetime(B.index, unit="s"); B = B[((t.dayofweek >= 5) | (t.hour >= 16))]
mos = sorted(B.ym.unique()); print("\n1) RAW ACCURACY of the top-30 % signals (thresholds from earlier months), trading hours, same windows")
for d in (30, 45, 120):
    th = pd.Series(np.nan, index=B.index)
    for m in mos[1:]: th[B.ym == m] = B[f"s{d}"][B.ym < m].abs().quantile(0.70)
    k = B[f"s{d}"].abs() >= th; h = (B[f"s{d}"] > 0) == (B.x_twap > 0); e = B.ym <= "2026-05"
    print(f"   T-{d:<3d}: Nov 2025 - May 2026 {100*h[k & e & th.notna()].mean():.1f} % ({int((k & e & th.notna()).sum())})   Jun - Nov 7 2026 {100*h[k & ~e].mean():.1f} % ({int((k & ~e).sum())})")
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); import policy_lab as L
L.SC[-30] = S[-30].reindex(L.W.index).values
for q in (0.85, 0.70, 0.50):
    th = np.full(L.n, np.inf)
    for mo in L.MONTHS: te = L.mon == mo; th[te] = np.quantile(np.abs(L.SC[-30][L.before(mo) & np.isfinite(L.SC[-30])]), q)
    L.THR[-30, q] = th
good = np.asarray((L.date.dayofweek >= 5) | (pd.to_datetime(L.Tix, unit="s").hour >= 16)); sig = L.ok & np.isfinite(L.SC[-30]) & (np.abs(L.SC[-30]) >= L.THR[-30, 0.70]) & good
print(f"\n   T-30 signals in trading hours Jun 1 - Nov 7: {int(sig.sum())}; with their trades cached: {100*np.isin(L.Tix[sig], list(L.P.keys())).mean():.0f} %")
mk = []; 
for i in np.flatnonzero(sig):
    if L.Tix[i] not in L.P: continue
    s = 0 if L.SC[-30][i] > 0 else 1; rel, ps, lift, usd, _ = L.side_view(L.Tix[i], s); a = (rel < -30) & (rel >= -60)
    if a.any(): mk.append(float(ps[a][-1]))
print(f"2) WHAT THE MARKET SHOWS AT T-30 for our side: median last price {np.median(mk):.2f}; at 0.50 or less in {100*np.mean(np.array(mk) <= 0.505):.0f} % of signals, 0.55 or more in {100*np.mean(np.array(mk) >= 0.55):.0f} %")
def show(lbl, r):
    m = good[r["i"]]; v = r["v"][m]; e = r["e"][m]; va = r["vol_after"][m] if not r["tk"][m].all() else r["vol"][m]; i = r["i"][m]
    full10 = (v * 10 / e).sum(); real200 = (v * np.minimum(200, va) / e).sum(); per = np.where(L.date[i] < "2026-08-20", 0, np.where(L.date[i] < "2026-10-01", 1, 2))
    print(f"   {lbl:58s} trades {len(v):5d} ({len(v)/160:4.1f}/day)  win {100*(v > 0).mean():5.1f} %  paid {100*e.mean():4.1f}c  {100*v.mean():+5.2f}c/share | $10: ${full10:+7,.0f} | $200 (fill-limited): ${real200:+8,.0f} | win by period " + " / ".join(f"{100*(v[per == p] > 0).mean():.1f}" for p in (0, 1, 2)))
print("\n3) FILLS on the real tape, Jun 1 - Nov 7, trading hours")
show("T-30 model, bid 0.50 from T-29 to T-15 (maker)", L.run(dec=-30, tier=0.70, stages=((-29, -15, 0.50),), tp=None, cross="skip", fresh=5, queue=50.0))
show("T-30 model, bid 0.50 from T-29 to T-15, strongest 15 %", L.run(dec=-30, tier=0.85, stages=((-29, -15, 0.50),), tp=None, cross="skip", fresh=5, queue=50.0))
for cap in (0.52, 0.55, 0.60): show(f"T-30 model, buy the offer at T-29 up to {cap:.2f} (taker, fee in)", L.run(dec=-30, tier=0.70, stages=(), take=(-29, cap), tp=None, cross="take", fresh=60, queue=50.0))
show("for comparison: live rule (T-120, bid from T-118, T-45 check)", L.run(dec=-120, tier=0.70, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=50.0))
show("for comparison: T-45 model, bid 0.50 from T-43 to T-15", L.run(dec=-45, tier=0.70, stages=((-43, -15, 0.50),), tp=None, cross="skip", fresh=5, queue=50.0))
