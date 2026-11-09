"""Kronos as a feature extractor for A ("linear probe"): the pretrained Kronos-small transformer reads the last
CTX one-minute BTC bars complete before the decision (bar start <= T-120, i.e. the bar ending at T-60) and we take its
final-position context vector (d_model) plus its own next-bar direction logit.  These embeddings are then used like any
other new input: walk-forward by month, logistic / LightGBM on [frozen heads] vs [frozen heads + PCA(embedding)].
If the frozen representation carries information about the Polymarket outcome beyond our heads, this shows it; a full
fine-tune only makes sense if it does.  Writes exports/kronos_emb.parquet."""
import os, sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, os.environ.get("KRONOS_DIR", "/home/user/kronos"))
from model import Kronos, KronosTokenizer                      # noqa: E402
from model.kronos import calc_time_stamps                       # noqa: E402
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
torch.set_num_threads(int(os.environ.get("THREADS", "3"))); CTX = int(os.environ.get("CTX", "256")); BATCH = 64
OUT = "/home/user/polybtcmodel/exports/kronos_emb.parquet"
bars = pd.read_parquet("/home/user/polybtcmodel/exports/btc_1m_2026-05-25_2026-11-07.parquet").sort_values("ts").reset_index(drop=True)
bars["amount"] = bars.volume * bars[["open", "high", "low", "close"]].mean(1)
parts = []
for mp in ("/home/user/data/master_june.parquet", "/home/user/data/master_july_L0.parquet", "/home/user/data/master.parquet", "/home/user/data/master_nov.parquet"):
    parts.append(pd.read_parquet(mp, columns=["date", "up", "m2_-45_close_p", "bm_-45_close_p", "bp_-60_z_s5", "bp_-60_z_s1"]))
W = pd.concat(parts); W = W[~W.index.duplicated(keep="last")].sort_index(); W["date"] = pd.to_datetime(W.date); W["month"] = W.date.dt.strftime("%b")
T = W.index.values.astype(np.int64)
if not os.path.exists(OUT):
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval(); model = Kronos.from_pretrained("NeoQuasar/Kronos-small").eval()
    ts = bars.ts.values; X = bars[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
    stamps = calc_time_stamps(pd.to_datetime(bars.ts, unit="s")).values.astype(np.float32)
    last = np.searchsorted(ts, T - 120, side="right") - 1          # last bar whose start <= T-120
    ok = last >= CTX; emb = np.full((len(W), model.d_model + 1), np.nan, np.float32); t0 = time.time()
    idx = np.where(ok)[0]
    with torch.no_grad():
        for j in range(0, len(idx), BATCH):
            b = idx[j:j + BATCH]; xs, ss = [], []
            for i in b:
                x = X[last[i] - CTX + 1:last[i] + 1]; mu, sd = x.mean(0), x.std(0); xs.append(np.clip((x - mu) / (sd + 1e-5), -5, 5)); ss.append(stamps[last[i] - CTX + 1:last[i] + 1])
            xb = torch.from_numpy(np.stack(xs)); sb = torch.from_numpy(np.stack(ss))
            s1, s2 = tok.encode(xb, half=True); logits, ctx = model.decode_s1(s1, s2, sb)
            # expected direction of the next bar from the s1 logits is hard to read off tokens; keep the raw context instead
            emb[b, :-1] = ctx[:, -1, :].numpy(); emb[b, -1] = ctx[:, -10:, :].mean(1).norm(dim=1).numpy()
            if (j // BATCH) % 20 == 0: print(f"{j + len(b)}/{len(idx)} windows, {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(emb, index=W.index, columns=[f"e{k}" for k in range(model.d_model)] + ["ctx_norm"]).to_parquet(OUT)
E = pd.read_parquet(OUT).loc[W.index]; print("embeddings", E.shape, "coverage", np.isfinite(E.e0).mean().round(3), flush=True)

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; mon = W.month.values; y = W.up.values.astype(int)
heads = np.column_stack([logit(W["m2_-45_close_p"].values), logit(W["bm_-45_close_p"].values), W["bp_-60_z_s5"].values, W["bp_-60_z_s1"].values])
LGB = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=20.0, verbose=-1, num_threads=4)
def walk(Xf, learner, npca=None):
    pred = np.full(len(W), np.nan)
    for mo in MONTHS:
        te = mon == mo; tr = np.isin(mon, [m_ for m_ in MONTHS if MONTHS.index(m_) < MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
        okr = np.isfinite(Xf).all(1); tr &= okr; te &= okr
        if tr.sum() < 2000 or te.sum() == 0: continue
        Xtr, Xte = Xf[tr], Xf[te]
        if npca:   # PCA on the embedding block (columns 4:), fitted on the training months only
            pca = PCA(npca).fit(Xtr[:, 4:]); Xtr = np.column_stack([Xtr[:, :4], pca.transform(Xtr[:, 4:])]); Xte = np.column_stack([Xte[:, :4], pca.transform(Xte[:, 4:])])
        if learner == "logit":
            sc = StandardScaler().fit(Xtr); lr = LogisticRegression(C=0.05, max_iter=1000).fit(sc.transform(Xtr), y[tr]); pred[te] = lr.predict_proba(sc.transform(Xte))[:, 1]
        else:
            bst = lgb.train(LGB, lgb.Dataset(Xtr, y[tr]), num_boost_round=300); pred[te] = bst.predict(Xte)
    return pred
P = {"frozen m2 head": W["m2_-45_close_p"].values, "heads [logit]": walk(heads, "logit")}
full = np.column_stack([heads, E.values])
for k in (8, 32):
    P[f"heads+kronos pca{k} [logit]"] = walk(full, "logit", k); P[f"heads+kronos pca{k} [lgb]"] = walk(full, "lgb", k)
P["kronos only pca32 [logit]"] = walk(np.column_stack([np.zeros((len(W), 4)), E.values]), "logit", 32)
rows = []
for nm, p in P.items():
    for mo in MONTHS:
        s = (mon == mo) & np.isfinite(p) & np.isfinite(P["frozen m2 head"])
        if s.sum() < 300: continue
        conf = np.abs(p[s] - 0.5) >= 0.05
        rows.append(dict(model=nm, month=mo, n=int(s.sum()), auc=roc_auc_score(y[s], p[s]), acc_conf=float(((p[s] > .5) == (y[s] == 1))[conf].mean()) if conf.sum() > 50 else np.nan, n_conf=int(conf.sum())))
R = pd.DataFrame(rows); R.to_csv("/home/user/polybtcmodel/reports/A_kronos_probe.csv", index=False)
print("\n== AUC vs the Polymarket outcome, walk-forward by month ==\n" + R.pivot(index="model", columns="month", values="auc")[MONTHS].round(3).to_string())
print("\n== accuracy on confident windows (|p-0.5| >= 0.05) ==\n" + R.pivot(index="model", columns="month", values="acc_conf")[MONTHS].round(3).to_string())
pd.DataFrame({k: v for k, v in P.items() if k != "frozen m2 head"}, index=W.index).to_parquet("/home/user/polybtcmodel/exports/kronos_probe_preds.parquet")
