"""A different model class for A: a dilated 1-D convolutional network (TCN) on the RAW second-level Binance
channels of the 600 s before T-47 (spot/perp log-returns, perp taker imbalance, spot taker imbalance, perp and
spot activity, basis).  No hand-made features.  Trained on the Polymarket outcome directly.
Splits: forward (train Jun 1 - Sep 30, early-stop on the last 10 days of Sep, test Oct 1 - Nov 7) and
backward (train Aug 20 - Nov 7, test Jun 1 - Jul 31).  Output: AUC per month for the TCN alone, and for a
logistic stack [frozen m2 head, TCN] fitted on the training period, plus the A-rule accuracy on confident windows.
Writes exports/seq_preds.parquet (T, p_tcn_fwd / p_tcn_bwd) for the trade-level evaluation in new_inputs_A.py."""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(int(os.environ.get("THREADS", "3"))); torch.manual_seed(0); np.random.seed(0)
TL = "/home/user/data/timeline"; TL0 = int(pd.Timestamp("2025-01-01").timestamp()); L = int(os.environ.get("SEQ_LEN", "600")); LAT = 47
EPOCHS = int(os.environ.get("EPOCHS", "12")); OUT = "/home/user/polybtcmodel/exports"
parts = []
for mp in ("/home/user/data/master_june.parquet", "/home/user/data/master_july_L0.parquet", "/home/user/data/master.parquet", "/home/user/data/master_nov.parquet"):
    parts.append(pd.read_parquet(mp, columns=["date", "up", "m2_-45_close_p", "bp_-60_z_s5"]))
W = pd.concat(parts); W = W[~W.index.duplicated(keep="last")].sort_index(); W["date"] = pd.to_datetime(W.date); W["month"] = W.date.dt.strftime("%b")
T = W.index.values.astype(np.int64); n = len(W); print("windows", n, flush=True)

# ---------------------------------------------------------------- raw channels -> (n, C, L) tensor
def load(name): return np.load(f"{TL}/{name}.npy", mmap_mode="r")
pc, pb, ps, pn, sc, sv, stb, scnt = (load(k) for k in ("p_close", "p_buy", "p_sell", "p_ntr", "s_close", "s_volume", "s_tb_base", "s_count"))
X = np.zeros((n, 8, L), np.float32)
end = T - LAT - TL0 + 1            # exclusive end index: last used second is T-47
for i in range(n):
    a, b = end[i] - L - 1, end[i]
    p = np.array(pc[a:b], dtype=np.float64); s = np.array(sc[a:b], dtype=np.float64)
    p = pd.Series(p).replace(0, np.nan).ffill().bfill().values; s = pd.Series(s).replace(0, np.nan).ffill().bfill().values
    X[i, 0] = np.diff(np.log(s)) * 1e4; X[i, 1] = np.diff(np.log(p)) * 1e4
    buy, sell = np.array(pb[a + 1:b], np.float64), np.array(ps[a + 1:b], np.float64); X[i, 2] = (buy - sell) / (buy + sell + 1e-6); X[i, 3] = np.log1p(buy + sell)
    vol, tb = np.array(sv[a + 1:b], np.float64), np.array(stb[a + 1:b], np.float64); X[i, 4] = np.where(vol > 0, 2 * tb / np.maximum(vol, 1e-9) - 1, 0.0); X[i, 5] = np.log1p(vol)
    X[i, 6] = np.log1p(np.array(pn[a + 1:b], np.float64)); X[i, 7] = (np.log(p[1:]) - np.log(s[1:])) * 1e4
    if i % 10000 == 0: print("built", i, flush=True)
X = np.nan_to_num(X); y = W.up.values.astype(np.float32); d = W.date.values
def norm(X, tr):
    mu = X[tr].mean(axis=(0, 2), keepdims=True); sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6; return np.clip((X - mu) / sd, -8, 8).astype(np.float32)

class TCN(nn.Module):
    def __init__(self, c_in=8, h=48, levels=(1, 2, 4, 8, 16, 32), k=5, drop=0.2):
        super().__init__(); layers = []; c = c_in
        for dil in levels:
            layers += [nn.Conv1d(c, h, k, dilation=dil, padding=dil * (k - 1) // 2), nn.GELU(), nn.Dropout(drop)]; c = h
        self.net = nn.Sequential(*layers); self.head = nn.Sequential(nn.Linear(2 * h, 32), nn.GELU(), nn.Dropout(drop), nn.Linear(32, 1))
    def forward(self, x):
        z = self.net(x); return self.head(torch.cat([z.mean(2), z[:, :, -30:].mean(2)], 1)).squeeze(1)   # whole-window and last-30-s summaries

def train_eval(tr, va, te, tag):
    Xn = norm(X, tr); Xt = torch.from_numpy(Xn); yt = torch.from_numpy(y)
    model = TCN(); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2); lossf = nn.BCEWithLogitsLoss()
    itr, iva = np.where(tr)[0], np.where(va)[0]; best, best_state, bad = -1, None, 0; t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); np.random.shuffle(itr)
        for j in range(0, len(itr), 256):
            b = itr[j:j + 256]; opt.zero_grad(); loss = lossf(model(Xt[b]), yt[b]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): pv = torch.sigmoid(torch.cat([model(Xt[iva[j:j + 2048]]) for j in range(0, len(iva), 2048)])).numpy()
        auc = roc_auc_score(y[iva], pv); print(f"{tag} epoch {ep + 1} valid AUC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best: best, bad, best_state = auc, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3: break
    model.load_state_dict(best_state); model.eval(); ite = np.where(te | va)[0]
    with torch.no_grad(): p = torch.sigmoid(torch.cat([model(Xt[ite[j:j + 2048]]) for j in range(0, len(ite), 2048)])).numpy()
    out = np.full(n, np.nan); out[ite] = p; return out, best

preds = pd.DataFrame(index=W.index)
fwd_tr = (d >= np.datetime64("2026-06-01")) & (d <= np.datetime64("2026-09-20")); fwd_va = (d > np.datetime64("2026-09-20")) & (d <= np.datetime64("2026-09-30")); fwd_te = d >= np.datetime64("2026-10-01")
bwd_tr = (d >= np.datetime64("2026-08-30")) & (d <= np.datetime64("2026-11-07")); bwd_va = (d >= np.datetime64("2026-08-20")) & (d < np.datetime64("2026-08-30")); bwd_te = d <= np.datetime64("2026-07-31")
preds["p_tcn_fwd"], _ = train_eval(fwd_tr, fwd_va, fwd_te, "forward"); preds["p_tcn_bwd"], _ = train_eval(bwd_tr, bwd_va, bwd_te, "backward")
preds.to_parquet(f"{OUT}/seq_preds.parquet")
m2 = W["m2_-45_close_p"].values; mon = W.month.values
def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
print("\n== AUC vs the Polymarket outcome ==")
for direction, p, trm in (("forward (test Oct-Nov)", preds.p_tcn_fwd.values, fwd_tr), ("backward (test Jun-Jul)", preds.p_tcn_bwd.values, bwd_tr)):
    # stack fitted on the training period with the TCN's in-sample output is optimistic; fit it on the validation slice instead
    vam = fwd_va if direction.startswith("forward") else bwd_va; okv = vam & np.isfinite(p) & np.isfinite(m2)
    lr = LogisticRegression(C=1.0).fit(np.column_stack([lg(m2[okv]), lg(p[okv])]), y[okv].astype(int))
    for mo in ("Jun", "Jul", "Aug", "Sep", "Oct", "Nov"):
        s = (mon == mo) & np.isfinite(p) & np.isfinite(m2) & ~trm & ~vam
        if s.sum() < 300: continue
        st = lr.predict_proba(np.column_stack([lg(m2[s]), lg(p[s])]))[:, 1]
        conf = np.abs(p[s] - 0.5) >= 0.05; confm = np.abs(m2[s] - 0.5) >= 0.05
        print(f"{direction:24s} {mo}: n={s.sum():5d}  frozen m2 {roc_auc_score(y[s], m2[s]):.3f}  TCN {roc_auc_score(y[s], p[s]):.3f}  stack {roc_auc_score(y[s], st):.3f}  "
              f"| acc on confident: m2 {((m2[s] > .5) == (y[s] == 1))[confm].mean() if confm.sum() else np.nan:.3f} (n {confm.sum()})  TCN {((p[s] > .5) == (y[s] == 1))[conf].mean() if conf.sum() else np.nan:.3f} (n {conf.sum()})  coef {np.round(lr.coef_[0], 2)}")
