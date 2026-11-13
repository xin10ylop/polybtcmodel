"""Does the SPEED of the share price say something the price itself does not?  Data: speed_data.py (every window since Sept 1, every 15 s, both sides).
 1) The example: a side whose shares are dropping fast while Bitcoin still favours it.  Win rate of that side minus what it can be sold for (after the fee).
 2) Everything at once: a model gets the share price, its change over 10/20/30/60 s, taker flow, volume, and Bitcoin's position and speed, is trained on
    Sept 1 - Oct 10 and judged on Oct 11 - Nov 7.  If it cannot beat the market's own price there, no hand-made rule from these ingredients will.
 3) The model used on OUR positions (simulated fills of the live setup, trading hours, unseen weeks): sell when it says the shares are worth less than the bid.
  python research/deep/speed_test.py"""
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, log_loss
D = pd.read_parquet("/home/user/data/deep/speed_BTC.parquet"); fee = lambda p: 0.07 * p * (1 - p); day = pd.to_datetime(D["T"], unit="s").dt.normalize().values; rng = np.random.default_rng(17); pd.set_option("display.width", 250)
def side(s):      # the same state seen from the Up (s=0) or the Down (s=1) holder
    sg = 1 if s == 0 else -1; q = D.p if s == 0 else 1 - D.p; o = pd.DataFrame({"T": D["T"], "m": D.m, "day": day, "win": D.up if s == 0 else ~D.up, "q": q, "sale": D.bid if s == 0 else 1 - D.ask, "buy": D.ask if s == 0 else 1 - D.bid, "spot": sg * D.spot, "avg60": sg * D.avg60, "fin": sg * D.fin, "b10": sg * D.b10, "b30": sg * D.b30, "flow10": sg * D.flow10, "flow30": sg * D.flow30, "vol10": D.vol10, "vol30": D.vol30, "sig": D.sig})
    for k in (10, 20, 30, 60): o[f"d{k}"] = q - (D[f"p{k}"] if s == 0 else 1 - D[f"p{k}"])
    o["s"] = s; return o
X = pd.concat([side(0), side(1)], ignore_index=True); X["net"] = X.sale - fee(X.sale.clip(0.001, 0.999)); X["cost"] = X.buy + fee(X.buy.clip(0.001, 0.999)); X["z"] = X.spot / (X.sig * np.sqrt(300 - X.m)); X["win"] = X.win.astype(float)
def cell(m):
    m = m & X.net.notna()
    if m.sum() < 60: return "-"
    e = (X.win - X.net)[m]; g = e.groupby(X.day[m]).sum().values; b = g[rng.integers(0, len(g), (1000, len(g)))].sum(1) / m.sum(); return f"{100*e.mean():+.1f} [{100*np.quantile(b,.025):+.1f},{100*np.quantile(b,.975):+.1f}] ({m.sum()})"
print("1) SHARES DROPPING FAST.  Win rate of the side minus its sale price after the fee, cents per share [95 % range] (cases).  + = better to HOLD, - = better to SELL")
for k in (10, 20, 30, 60):
    rows = []
    for lo, hi in ((-1, -0.30), (-0.30, -0.20), (-0.20, -0.10), (-0.10, -0.05), (-0.05, 0.05), (0.05, 1)):
        dk = X[f"d{k}"]; b = (dk >= lo) & (dk < hi)
        rows.append({f"share price change over {k} s": f"{int(100*lo) if lo > -1 else 'less'} to {int(100*hi) if hi < 1 else 'more'}¢", "Bitcoin still $5+ on this side": cell(b & (X.spot >= 5)), "60-s average on this side, price already against": cell(b & (X.avg60 > 0) & (X.spot < 0)), "Bitcoin $5+ against": cell(b & (X.spot <= -5)), "all": cell(b)})
    print(pd.DataFrame(rows).to_string(index=False)); print()
F = ["q", "d10", "d20", "d30", "d60", "flow10", "flow30", "vol10", "vol30", "spot", "avg60", "fin", "b10", "b30", "sig", "z", "m"]; FB = ["spot", "avg60", "fin", "b10", "b30", "sig", "z", "m"]; FP = ["q", "spot", "avg60", "fin", "b10", "b30", "sig", "z", "m"]
tr = (X.day < np.datetime64("2026-10-11")).values; te = ~tr; par = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=20.0, verbose=-1, num_threads=4, seed=0)
M = {nm: lgb.train(par, lgb.Dataset(X.loc[tr, f], X.win[tr]), 400) for nm, f in (("Bitcoin only", FB), ("Bitcoin + share price", FP), ("Bitcoin + share price + speed and flow", F))}; P = {nm: M[nm].predict(X[f]) for nm, f in (("Bitcoin only", FB), ("Bitcoin + share price", FP), ("Bitcoin + share price + speed and flow", F))}
k = te & X.q.notna().values & (X.s.values == 0); print(f"2) WHO PREDICTS THE RESULT BEST on the unseen weeks (Oct 11 - Nov 7, {k.sum()} moments, lower loss = better)")
print(f"   the market's own share price            : loss {log_loss(X.win[k], X.q[k].clip(0.005, 0.995)):.4f}, AUC {roc_auc_score(X.win[k], X.q[k]):.4f}")
for nm in P: print(f"   model, {nm:38s}: loss {log_loss(X.win[k], np.clip(P[nm][k], 0.005, 0.995)):.4f}, AUC {roc_auc_score(X.win[k], P[nm][k]):.4f}")
imp = pd.Series(M["Bitcoin + share price + speed and flow"].feature_importance("gain"), index=F); print("   what the full model leans on (share of gain):", ", ".join(f"{a} {100*b/imp.sum():.0f}%" for a, b in imp.sort_values(ascending=False).head(8).items()))
pf = P["Bitcoin + share price + speed and flow"]; rows = []
for mg in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
    for lbl, edge, px, tag in (("BUY the side when the model says it is worth more than it costs", pf - X.cost.values, X.cost.values, "buy"), ("SELL a side when the model says it is worth less than the bid", X.net.values - pf, X.net.values, "sell")):
        m = te & np.isfinite(edge) & (edge > mg); g = X[m].sort_values(["T", "m"]).drop_duplicates(["T", "s"]) if tag == "buy" else X[m]
        e = (g.win - g.cost) if tag == "buy" else (g.net - g.win); gd = e.groupby(g.day).sum().values
        if len(g) < 40: continue
        b = gd[rng.integers(0, len(gd), (2000, len(gd)))].sum(1) / len(g); rows.append({"rule (unseen weeks)": lbl, "model's margin": f"{int(100*mg)}¢+", "cases": len(g), "side wins %": round(100 * g.win.mean(), 1), "price ¢ (fee in)": round(100 * (g.cost if tag == "buy" else g.net).mean(), 1), "gain ¢/share": round(100 * e.mean(), 2), "95% lo": round(100 * np.quantile(b, .025), 2), "95% hi": round(100 * np.quantile(b, .975), 2)})
print(pd.DataFrame(rows).sort_values(["rule (unseen weeks)", "cases"], ascending=[True, False]).to_string(index=False))
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from policy_lab import run, Tix, SC, date
r = run(dec=-120, tier=0.70, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=50.0); I = r["i"]; g_ = np.asarray((date.dayofweek >= 5) | (pd.to_datetime(Tix, unit="s").hour >= 16))[I] & np.asarray(date[I] >= "2026-10-11"); I = I[g_]; win = (r["v"] > 0)[g_]; sd = np.where(SC[-120][I] > 0, 0, 1)
X["pf"] = pf; Y = X.merge(pd.DataFrame({"T": Tix[I].astype(np.int64), "s": sd, "w": win}), on=["T", "s"]).sort_values(["T", "m"]); hold = np.where(win, 0.5, -0.5).sum() * 200
print(f"\n3) ON OUR POSITIONS in the unseen weeks ({len(I)} fills in trading hours, holding = ${hold:,.0f} at $100 a trade): sell at the first moment the model values the shares below the bid by the margin")
for mg in (0.02, 0.04, 0.06, 0.08, 0.10):
    c = Y[(Y.net - Y.pf > mg) & Y.net.notna()].drop_duplicates("T"); dl = ((c.net - 0.5) - np.where(c.w, 0.5, -0.5)) * 200; gd = dl.groupby(c.day).sum().values; b = gd[rng.integers(0, len(gd), (2000, len(gd)))].sum(1) if len(gd) else np.zeros(1)
    print(f"   margin {int(100*mg):2d}¢: sold {len(c):4d} positions (would have won {100*c.w.mean():.0f} %, sold for {100*c.net.mean():.0f}¢) -> change vs holding ${dl.sum():+,.0f}  [95 % range {np.quantile(b,.025):+,.0f} .. {np.quantile(b,.975):+,.0f}]")
