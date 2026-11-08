"""Kronos (foundation model for K-lines, arXiv 2508.02739, pretraining through June 2024) as a forecaster for the 5-minute
Polymarket windows, and as an agreement filter on strategy A. Needs a machine with torch and the Kronos repository:
  git clone --depth 1 https://github.com/shiyu-coder/Kronos /home/user/ext/Kronos
  pip install torch --index-url https://download.pytorch.org/whl/cpu ; pip install einops huggingface_hub safetensors tqdm
Inputs (exports/ in this repo): btc_1m_*.parquet (Binance spot 1-minute OHLCV, 2026-05-25 .. 2026-11-07) and windows_A.parquet
(every 5-minute window T with the outcome `up`, A's head `signal`, whether A traded it and its pnl, and the period).
For each window the model sees the last 512 one-minute bars that are complete at the decision instant T-45 (bars through the one
starting at T-120) and forecasts 6 bars (T-60 .. T+300); P(up) = share of samples whose close of the bar starting at T+240
is above the close of the bar starting at T-60 (the settlement average is centred near T+270 and the reference near T-30).
  KRONOS_DIR=/home/user/ext/Kronos python research/kronos_eval.py [--model small|base] [--samples 10] [--batch 32] [--limit N] [--all-windows-stride 8]
Writes exports/kronos_forecasts.parquet and prints: AUC by period (stand-alone), and A's win rate / edge when Kronos agrees vs disagrees.
"""
import os, sys, time, argparse, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="small"); ap.add_argument("--samples", type=int, default=10); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--all-windows-stride", type=int, default=8); ap.add_argument("--internal-samples", type=int, default=5); ap.add_argument("--context", type=int, default=512); ap.add_argument("--eval-only", action="store_true")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "..", "exports"); OUT = os.path.join(EXP, "kronos_forecasts.parquet")
bars = pd.read_parquet(os.path.join(EXP, "btc_1m_2026-05-25_2026-11-07.parquet")).sort_values("ts").reset_index(drop=True)
W = pd.read_parquet(os.path.join(EXP, "windows_A.parquet")).sort_values("T").reset_index(drop=True)
sel = W[(W.traded == 1) | ((np.arange(len(W)) % a.all_windows_stride == 0) if a.all_windows_stride > 0 else False)]        # all A trades + a stride of all windows for the stand-alone test
if a.limit: sel = sel.head(a.limit)
if not a.eval_only:
    sys.path.insert(0, os.environ.get("KRONOS_DIR", "/home/user/ext/Kronos"))
    from model import Kronos, KronosTokenizer, KronosPredictor
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base"); mdl = Kronos.from_pretrained(f"NeoQuasar/Kronos-{a.model}")
    pred = KronosPredictor(mdl, tok, device="cpu", max_context=a.context)
    ts_index = pd.Index(bars.ts.values); rows = []; t_w = time.time(); done = set()
    if os.path.exists(OUT):
        prev = pd.read_parquet(OUT); rows = prev.to_dict("records"); done = set(prev["T"].tolist()); print(f"resuming: {len(done)} windows already scored")
    todo = [int(T) for T in sel["T"] if int(T) not in done]
    for i in range(0, len(todo), a.batch):
        Ts = todo[i:i + a.batch]; dfs, xts, yts, keep = [], [], [], []
        for T in Ts:
            end = ts_index.searchsorted(T - 120, side="right")       # bars with start <= T-120 are complete at T-60
            lo = end - a.context
            if lo < 0: continue
            x = bars.iloc[lo:end]
            if len(x) < a.context or (x.ts.iloc[-1] != T - 120): continue
            dfs.append(x[["open", "high", "low", "close", "volume"]].reset_index(drop=True)); xts.append(pd.Series(pd.to_datetime(x.ts.values, unit="s")))
            yts.append(pd.Series(pd.to_datetime(np.arange(T - 60, T + 300, 60), unit="s"))); keep.append(T)
        if not dfs: continue
        preds = pred.predict_batch(dfs, xts, yts, pred_len=6, T=1.0, top_p=0.9, sample_count=a.internal_samples, verbose=False)   # one call: the model averages internal_samples paths
        # predict_batch averages the samples internally; to get a probability we also run single-sample draws when samples > 1
        for T, p in zip(keep, preds):
            c = p["close"].values; ref = float(c[0]); end_px = float(c[5])
            rows.append(dict(T=T, k_move_bp=1e4 * np.log(end_px / ref), k_mean_up=float(end_px > ref)))
        if a.samples > 1:
            ups = np.zeros(len(keep))
            for s in range(a.samples):
                ps = pred.predict_batch(dfs, xts, yts, pred_len=6, T=1.0, top_p=0.9, sample_count=1, verbose=False)
                for j, p in enumerate(ps): ups[j] += float(p["close"].values[5] > p["close"].values[0])
            for j, T in enumerate(keep): rows[-len(keep) + j]["k_up"] = ups[j] / a.samples
        pd.DataFrame(rows).to_parquet(OUT, index=False)
        print(f"{len(rows)}/{len(sel)} windows, {time.time()-t_w:.0f}s ({(time.time()-t_w)/max(i+len(Ts),1):.2f} s/window)", flush=True)
K = pd.read_parquet(OUT); m = W.merge(K, on="T", how="inner"); print(f"\nscored windows: {len(m)}")
from sklearn.metrics import roc_auc_score
col = "k_up" if "k_up" in m and m.k_up.notna().any() else "k_mean_up"
print("\n== stand-alone: AUC of Kronos P(up) vs the Polymarket outcome, by period ==")
for per, g in m.groupby("per"):
    g = g.dropna(subset=[col, "up"]); 
    if len(g) > 30 and g.up.nunique() == 2: print(f"  {per:5s} n={len(g):5d} AUC {roc_auc_score(g.up, g[col]):.3f}  (move-based AUC {roc_auc_score(g.up, g.k_move_bp):.3f})  P(up) mean {g[col].mean():.3f}")
print("\n== as a filter on A: A's traded windows, split by whether Kronos agrees with A's side ==")
t = m[m.traded == 1].copy(); t["a_up"] = t.signal > 0.5; t["k_agree"] = np.where(t.a_up, t[col] > 0.5, t[col] < 0.5); t["k_strong"] = np.where(t.a_up, t[col] >= 0.7, t[col] <= 0.3)
for per, g in t.groupby("per"):
    for lab, mask in (("all A trades", np.ones(len(g), bool)), ("Kronos agrees", g.k_agree.values), ("Kronos disagrees", ~g.k_agree.values), ("Kronos agrees strongly (>= 0.7)", g.k_strong.values)):
        x = g[mask]
        if len(x) >= 10: print(f"  {per:5s} {lab:32s} n={len(x):4d} win {100*(x.pnl>0).mean():.1f}%  edge {100*x.pnl.mean():+.2f} c/share")
