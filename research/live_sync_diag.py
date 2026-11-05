"""Why do live forecasts differ from the archive replay? Replays the day's A (T-45) and B (T) forecasts from the
archive through the live Model, then once more with the live path's last-second artefact applied: at decision time
the last known second has no committed perp bar yet, so finalize_to() synthesizes it from spot x archive basis with
spot volume/count standing in for the perp's. Compares both with the live forecasts recorded that day.
  PAPER_DIR=... python research/live_sync_diag.py 2026-11-04
"""
import os, sys, json, datetime as dt
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import paper_trader as pt
day = sys.argv[1]; PAPER = os.environ.get("PAPER_DIR", "/home/user/polybtcmodel/paper")
t0 = int(dt.datetime.fromisoformat(day + "T00:00:00+00:00").timestamp()); t1 = t0 + pt.DAY
fcs = [json.loads(l) for l in open(f"{PAPER}/forecasts.jsonl") if l.strip()]
A = {r["T"]: r for r in fcs if r["strategy"] == "A" and t0 <= r["T"] < t1 and "close_p" in r}
B = {r["T"]: r for r in fcs if r["strategy"] == "B" and t0 <= r["T"] < t1 and "close_p" in r}
tl = pt.LiveTimeline((t0 // pt.DAY - 10) * pt.DAY); a_end = pt.load_archive(tl); assert a_end >= t1, "archive does not cover the day"
tl.finalize_to(t1); model = pt.Model(tl); basis = tl.basis
PERP = ("p_open", "p_high", "p_low", "p_close", "p_vwap", "p_buy", "p_sell", "p_ntr", "p_maxq", "p_nb", "p_lbv", "p_lsv", "p_qv")
def corrupt(sec):
    i = sec - tl.t0_epoch; a = tl.a; saved = {k: a[k][i] for k in PERP}
    px = a["s_close"][i] * basis
    for k in ("p_open", "p_high", "p_low", "p_close", "p_vwap"): a[k][i] = px
    a["p_buy"][i] = a["s_tb_base"][i]; a["p_sell"][i] = max(a["s_volume"][i] - a["s_tb_base"][i], 0); a["p_ntr"][i] = a["s_count"][i]
    a["p_nb"][i] = a["s_count"][i] / 2; a["p_qv"][i] = px * a["s_volume"][i]; a["p_maxq"][i] = 0.0; a["p_lbv"][i] = 0.0; a["p_lsv"][i] = 0.0
    return saved
def restore(sec, saved):
    i = sec - tl.t0_epoch
    for k, v in saved.items(): tl.a[k][i] = v
rows = []
for T in sorted(set(A) | set(B)):
    r = dict(T=T)
    if T in A:
        r["A_live"] = A[T]["close_p"]; r["A_clean"] = model.forecast(T, -45, 0)["close_p"]
        s = corrupt(T - 46); r["A_art"] = model.forecast(T, -45, 0)["close_p"]; restore(T - 46, s)
        rb = (tl.a["p_close"][T - 46 - tl.t0_epoch] / tl.a["s_close"][T - 46 - tl.t0_epoch]); r["A_bp"] = np.log(basis / rb) * 1e4   # synthetic - real perp, bp
    if T in B:
        r["B_live"] = B[T]["close_p"]; r["B_clean"] = model.forecast(T, -pt.LAG, pt.LAG)["close_p"]
        s = corrupt(T - 1); r["B_art"] = model.forecast(T, -pt.LAG, pt.LAG)["close_p"]; restore(T - 1, s)
    rows.append(r)
df = pd.DataFrame(rows)
for k in ("A", "B"):
    q = df.dropna(subset=[f"{k}_live", f"{k}_clean"]) if f"{k}_live" in df else pd.DataFrame()
    if not len(q): continue
    e_live = q[f"{k}_live"] - q[f"{k}_clean"]; e_art = q[f"{k}_art"] - q[f"{k}_clean"]
    print(f"{k}: n={len(q)}  live-clean: mean {e_live.mean():+.4f} |mean| {e_live.abs().mean():.4f} share>0 {(e_live>0).mean():.2f}  |  artefact-clean: mean {e_art.mean():+.4f} |mean| {e_art.abs().mean():.4f} share>0 {(e_art>0).mean():.2f}  |  corr(live err, artefact err) {e_live.corr(e_art):+.3f}  |  live vs artefact |diff| {(q[f'{k}_live']-q[f'{k}_art']).abs().mean():.4f} vs live vs clean {e_live.abs().mean():.4f}")
    if k == "A": print(f"   synthetic-minus-real perp in the last second: mean {q.A_bp.mean():+.2f} bp, corr with live err {q.A_bp.corr(e_live):+.3f}, corr with artefact err {q.A_bp.corr(e_art):+.3f}")
df.round(4).to_csv(f"{PAPER}/live_sync_diag_{day}.csv", index=False); print("saved", f"{PAPER}/live_sync_diag_{day}.csv")
