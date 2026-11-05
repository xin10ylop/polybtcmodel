"""End-to-end replay of the LIVE code path on an archived day: the day's Binance bars are streamed into the trader's
LiveTimeline with the real arrival delays (spot 1 s kline ~0.8 s after the second ends, perp trades ~0.5 s), the trader's
own finalize() rule runs on a fake clock, and the A (T-45) and B (T) forecasts are taken at the instants the trader
takes them. They are compared with the backtest's forecasts for the same windows (scripts/score_m2.py, master.parquet).
  python research/replay_live_day.py 2026-11-04 [old]     ('old' = the pre-fix clock-based finalize rule, for reference)
"""
import os, sys, time, datetime as dt
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc"))
import paper_trader as pt
day = sys.argv[1]; OLD = len(sys.argv) > 2 and sys.argv[2] == "old"
D0 = int(dt.datetime.fromisoformat(day + "T00:00:00+00:00").timestamp()); D1 = D0 + pt.DAY
SPOT_DELAY, PERP_DELAY, TICK = 0.8, 0.5, 0.25
tl = pt.LiveTimeline((D0 // pt.DAY - 10) * pt.DAY); a_end = pt.load_archive(tl); assert a_end >= D1
i0, i1 = D0 - tl.t0_epoch, D1 - tl.t0_epoch
saved = {k: tl.a[k][i0:i1].copy() for k in tl.a}
for k in tl.a:                                   # wipe the day: the stream puts it back
    tl.a[k][i0:i1] = np.nan if (k in ("s_open", "s_high", "s_low", "s_close", "p_open", "p_high", "p_low", "p_close", "p_vwap", "e_c")) else 0
tl.known = i0; tl.synth[i0:i1] = False; tl.ffill[i0:i1] = False; tl.last_spot_sec = D0 - 1; tl.last_perp_sec = D0 - 1; tl.last_perp_commit = D0 - 1; tl.perp_pending = {}
tl.basis_live = None
model = pt.Model(tl)
clock = [float(D0)]
pt.time.time = lambda: clock[0]
class Stub:
    GAP = pt.Trader.GAP
    def __init__(self, tl): self.tl = tl
    finalize = pt.Trader.finalize
    def finalize_old(self):                      # the rule before 2026-11-05: clock-based, perp committed for s < sec-1
        now = pt.time.time(); sec = int(now) - 1
        for s_ in sorted(s for s in self.tl.perp_pending if s < sec - 1):
            bb = self.tl.perp_pending.pop(s_); bb["p_vwap"] = bb["pv"] / bb["vv"] if bb["vv"] > 0 else bb["p_close"]; self.tl.set_perp(s_, bb)
        self.tl.finalize_to(sec)
st = Stub(tl); fin = st.finalize_old if OLD else st.finalize
m = pd.read_parquet("/home/user/data/master.parquet", columns=["m2_-45_close_p", "m2_0_close_p", "m2_-45_h10_p"]); m = m[(m.index >= D0) & (m.index < D1)]
rows = []; next_spot = D0; next_perp = D0; t_w = time.time()
Ts = list(range(D0 + 300, D1 + 1, 300)); doneA = set(); doneB = set()
while clock[0] < D1 + 5:
    now = clock[0]
    while next_spot < D1 and now >= next_spot + 1 + SPOT_DELAY:   # kline for second next_spot has arrived
        s = next_spot; i = s - tl.t0_epoch
        if np.isfinite(saved["s_close"][i - i0]):
            tl.set_spot(s, saved["s_open"][i - i0], saved["s_high"][i - i0], saved["s_low"][i - i0], saved["s_close"][i - i0], saved["s_volume"][i - i0], saved["s_qvol"][i - i0], int(saved["s_count"][i - i0]), saved["s_tb_base"][i - i0])
            tl.last_spot_sec = s
        tl.set_eth(s, saved["e_c"][i - i0], saved["e_v"][i - i0], saved["e_bv"][i - i0]); next_spot += 1
    while next_perp < D1 and now >= next_perp + 1 + PERP_DELAY:   # all trades of second next_perp have arrived
        s = next_perp; j = s - D0
        if np.isfinite(saved["p_close"][j]) and (saved["p_buy"][j] + saved["p_sell"][j]) > 0:
            vol = saved["p_buy"][j] + saved["p_sell"][j]
            tl.perp_pending[s] = dict(p_open=saved["p_open"][j], p_high=saved["p_high"][j], p_low=saved["p_low"][j], p_close=saved["p_close"][j], p_buy=saved["p_buy"][j], p_sell=saved["p_sell"][j], p_ntr=int(saved["p_ntr"][j]), p_maxq=saved["p_maxq"][j], pv=saved["p_vwap"][j] * vol, vv=vol, p_nb=saved["p_nb"][j], p_lbv=saved["p_lbv"][j], p_lsv=saved["p_lsv"][j])
            tl.last_perp_sec = s
        next_perp += 1
    fin()
    for T in Ts:
        if T not in doneA and now >= T - 43.5 and tl.known >= (T - 45) - tl.t0_epoch:
            doneA.add(T); fc = model.forecast(T, -45, 0); i = T - 46 - tl.t0_epoch
            rows.append(dict(T=T, k="A", live=fc.get("close_p"), h10=fc.get("h10_p"), arch=m["m2_-45_close_p"].get(T, np.nan), arch_h10=m["m2_-45_h10_p"].get(T, np.nan), synth=int(tl.synth[i - 59:i + 1].sum()), ffill=int(tl.ffill[i - 59:i + 1].sum()), delay=round(now - (T - 43.5), 2)))
        if T not in doneB and T + 1.6 <= now < T + 10 and tl.known >= T - tl.t0_epoch:
            doneB.add(T); fc = model.forecast(T, -pt.LAG, pt.LAG); i = T - 1 - tl.t0_epoch
            rows.append(dict(T=T, k="B", live=fc.get("close_p"), arch=m["m2_0_close_p"].get(T, np.nan), synth=int(tl.synth[i - 59:i + 1].sum()), ffill=int(tl.ffill[i - 59:i + 1].sum()), delay=round(now - (T + 1.6), 2)))
    if int(now) % 3600 == 0 and abs(now - int(now)) < TICK / 2: print(f"  {dt.datetime.fromtimestamp(now, dt.UTC):%H:%M} ({time.time() - t_w:.0f}s)", flush=True)
    clock[0] += TICK
df = pd.DataFrame(rows); df["err"] = df.live - df.arch
print(f"\n== live-path replay of {day} ({'OLD clock rule' if OLD else 'fixed rule'}): spot delay {SPOT_DELAY}s, perp delay {PERP_DELAY}s")
for k, thr in (("A", 0.05), ("B", None)):
    q = df[(df.k == k)].dropna(subset=["live", "arch"])
    flips = (np.sign(q.live - 0.5) != np.sign(q.arch - 0.5)).mean()
    line = f"{k}: n={len(q)}  mean |live-archive| {q.err.abs().mean():.4f}  max {q.err.abs().max():.4f}  signed mean {q.err.mean():+.4f}  side flips {flips:.1%}  decisions late by {q.delay.median():.2f}s (max {q.delay.max():.2f})  synthetic/ffilled seconds used: {int(q.synth.sum())}/{int(q.ffill.sum())}"
    if thr:
        dl = (q.live - 0.5).abs() >= thr; da = (q.arch - 0.5).abs() >= thr
        line += f"\n   confident (>= 0.05): live {dl.mean():.1%} vs archive {da.mean():.1%}; trade/no-trade differs on {(dl != da).mean():.1%}; mean |p-0.5| live {(q.live-0.5).abs().mean():.4f} vs archive {(q.arch-0.5).abs().mean():.4f}"
        line += f"\n   h10 head: mean live {q.h10.mean():.3f} vs archive {q.arch_h10.mean():.3f}, mean |diff| {(q.h10 - q.arch_h10).abs().mean():.4f}"
    print(line)
out = f"/home/user/polybtcmodel/paper/replay_live_{day}{'_old' if OLD else ''}.csv"; df.round(5).to_csv(out, index=False); print("saved", out)
