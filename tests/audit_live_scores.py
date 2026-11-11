"""Did the LIVE bot score what the model scores on Binance's official data?  Replays every rule-A decision the bot logged on one day
(PAPER_DIR/forecasts.jsonl) from the Binance archive of that day and compares the scores.  Needs the day's archive (published the next day).
  python tests/audit_live_scores.py 2026-11-11 [/home/user/paper]"""
import os, sys, json, tempfile, datetime as dt, numpy as np
DAY = sys.argv[1]; LIVE = sys.argv[2] if len(sys.argv) > 2 else "/home/user/paper"
os.environ["PAPER_DIR"] = tempfile.mkdtemp(prefix="audit_live_"); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc")); import paper_trader as pt, newa_live; from newa_live import NewA
d0 = int(dt.datetime.fromisoformat(DAY + "T00:00:00+00:00").timestamp()); d1 = d0 + 86400
recs = [json.loads(l) for l in open(os.path.join(LIVE, "forecasts.jsonl")) if l.strip()]; A = [r for r in recs if r.get("strategy") == "A" and d0 <= r["T"] < d1 and r.get("new_score") is not None and np.isfinite(r["new_score"])]
C = {r["T"]: r for r in recs if r.get("strategy") == "A_conf" and d0 <= r["T"] < d1 and r.get("score45") is not None}
if not A: print(f"no rule-A decisions with a score on {DAY} in {LIVE}/forecasts.jsonl"); sys.exit(2)
atl = pt.load_timeline(); start = max(d0 - 10 * 86400, int(atl.t0_epoch))      # the live bot starts its history 10 days back; the box keeps only about 11 days of archive
tl = pt.LiveTimeline(start, days=14); pt.load_archive(tl)
if tl.known + tl.t0_epoch < max(r["T"] for r in A): print(f"the archive ends {dt.datetime.fromtimestamp(tl.known + tl.t0_epoch, dt.UTC)}: it does not cover {DAY} yet (Binance publishes a day's files the next day; fetch them first)"); sys.exit(2)
model = pt.Model(tl); NA = {}
def na(dec):
    if dec not in NA: NA[dec] = NewA(path=os.path.join(newa_live.MODELS, f"ourcase_{newa_live.ASSET}_d{dec}_live"))
    return NA[dec]
def replay(T, dec, open_known=True):
    tl.known = (T + dec - 7) - tl.t0_epoch; tl.last_spot_sec = T + dec - 7; model.regime_cache.pop((T, dec, 0), None); model.regime(T, dec, 0)
    tl.known = (T + dec) - tl.t0_epoch; tl.last_spot_sec = (T + dec) if open_known else (T + dec - 1); model.row(T, dec, 0); i = (T + dec) - tl.t0_epoch
    return na(dec).score(model.last_X, tl.view("s_close", i - 15, i))[0]
rows = []
for r in A:
    dec = int(r.get("dec", -45)); s = replay(r["T"], dec, bool(r.get("open_known", True))); rows.append((r["T"], dec, r["new_score"], s, r.get("new_thr", np.nan), r.get("synth_clean", 0) or 0, r.get("synth_last60", 0) or 0, r.get("ffill_last60", 0) or 0, r.get("perp_lag"), r.get("action", "")))
a = np.array([(x[2], x[3], x[4]) for x in rows], float); live, rep, thr = a.T; diff = np.abs(live - rep); clean = np.array([x[5] <= 30 and x[6] == 0 and x[7] == 0 for x in rows])
dl = np.abs(live) >= thr; dr = np.abs(rep) >= thr; flips = dl != dr; sign = (np.sign(live) != np.sign(rep)) & (dl | dr)
print(f"{DAY}: {len(rows)} decisions replayed from the Binance archive (decision times {sorted(set(x[1] for x in rows))}); {clean.sum()} made on clean live data, {(~clean).sum()} during start-up / feed gaps (not traded)")
for nm, m in (("all decisions", np.ones(len(rows), bool)), ("clean data only (the ones that can trade)", clean)):
    if m.sum() < 3: continue
    print(f"  {nm:42s} n {m.sum():>4d} | score live vs archive: mean |diff| {diff[m].mean():.4f}, 95th pct {np.percentile(diff[m], 95):.4f}, max {diff[m].max():.4f} (threshold {np.nanmedian(thr[m]):.4f}, score std {rep[m].std():.4f}), correlation {np.corrcoef(live[m], rep[m])[0,1]:.4f} | trade decisions that differ {flips[m].sum()} ({flips[m].mean():.1%}) | opposite side on a signal {sign[m].sum()}")
bought = np.array([x[9].startswith("BUY") for x in rows]); 
if bought.any(): print(f"  orders actually sent: {bought.sum()}; of those the archive would also trade {dr[bought].sum()} and on the same side {((np.sign(live) == np.sign(rep)) & dr)[bought].sum()}")
if C:
    cf = [(T, c["score45"], replay(T, -45), c.get("cancelled")) for T, c in sorted(C.items())]; c = np.array([(x[1], x[2]) for x in cf], float)
    print(f"  T-45 confirmation: {len(cf)} replayed, mean |diff| {np.abs(c[:,0]-c[:,1]).mean():.4f}, sign differs {(np.sign(c[:,0]) != np.sign(c[:,1])).sum()}")
w = np.argsort(-diff)[:5]; print("  largest differences: " + "; ".join(f"{dt.datetime.fromtimestamp(rows[i][0], dt.UTC):%H:%M} live {live[i]:+.4f} archive {rep[i]:+.4f} (perp lag {rows[i][8]}, rebuilt s {rows[i][5]})" for i in w))
m = clean if clean.sum() >= 10 else np.ones(len(rows), bool); ok = diff[m].mean() < 0.01 and flips[m].mean() <= 0.05 and sign[m].sum() == 0
print("RESULT:", "PASS - the live bot scores what the model scores on Binance's own data" if ok else "CHECK - live and archive scores differ more than expected: send this output"); sys.exit(0 if ok else 1)
