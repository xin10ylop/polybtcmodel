"""The T-45 confirmation of an earlier A decision (paper_trader.Trader.confirm_a), offline on the archive timeline: for windows where the T-120
research score gives a side, the bid must be cancelled exactly when the T-45 research score has the other sign.
  A_DEC=-120 python tests/test_confirm.py [YYYY-MM-DD] [n_windows]"""
import os, sys, tempfile, glob, datetime as dt, numpy as np, pandas as pd
os.environ.setdefault("A_DEC", "-120"); DAY = sys.argv[1] if len(sys.argv) > 1 else "2026-11-06"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
os.environ["PAPER_DIR"] = tempfile.mkdtemp(prefix="confirm_"); sys.path.insert(0, "/home/user/polybtcmodel/polybtc"); import paper_trader as pt, newa_live; from newa_live import NewA
d0 = dt.datetime.fromisoformat(DAY + "T00:00:00+00:00"); tl = pt.LiveTimeline(int(d0.timestamp()) - 12 * 86400, days=14); pt.load_archive(tl)
na120 = NewA(); na45 = NewA(path=os.path.join(newa_live.MODELS, "ourcase_BTC_d-45_live"))
D120 = pd.concat([pd.read_parquet(f) for f in glob.glob("/home/user/data/deep/feats_BTC_d-120_ph0_*.parquet")]).set_index("T"); D45 = pd.concat([pd.read_parquet(f) for f in glob.glob("/home/user/data/deep/feats_BTC_ph0_*.parquet")]).set_index("T")
class FakeExec:
    def __init__(self): self.calls = []
    def cancel_now(self, st, T): self.calls.append((st, T))
tr = object.__new__(pt.Trader); tr.tl = tl; tr.model = pt.Model(tl); tr.newa45 = na45; tr.a_confirm = True; tr.a_side = {}; tr.done = set(); tr.exec = FakeExec(); tr.ffc = open(os.path.join(os.environ["PAPER_DIR"], "f.jsonl"), "w")
rng = np.random.default_rng(3); Ts = sorted(int(d0.timestamp()) + 300 * int(k) for k in rng.choice(288, N, replace=False)); bad = 0; nd = 0
for T in Ts:
    s120 = float(na120.booster.predict(np.array([[float(D120.loc[T][f]) for f in na120.feats]], dtype=np.float32))[0]); s45 = float(na45.booster.predict(np.array([[float(D45.loc[T][f]) for f in na45.feats]], dtype=np.float32))[0])
    side = "UP" if s120 > 0 else "DOWN"; tr.a_side[T] = side; want = (s45 > 0) != (side == "UP")
    tl.known = (T - 45) - tl.t0_epoch; tl.last_spot_sec = T - 45
    early = tr.confirm_a(T, T - 60.0); got = tr.confirm_a(T, T - 43.5); again = tr.confirm_a(T, T - 43.0)
    called = ("A", T) in tr.exec.calls; nd += want
    if early is not None or again is not None or got != want or called != want: bad += 1; print("MISMATCH", T, side, round(s45, 4), "want", want, "got", got, "called", called, early, again)
print(f"{N} windows on {DAY}: the T-45 model disagreed with the T-120 side in {nd}; cancel sent exactly in those: {'PASS' if bad == 0 else f'FAIL ({bad})'}"); sys.exit(0 if bad == 0 else 1)
