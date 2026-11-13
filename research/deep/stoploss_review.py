"""Review of the sell-out study ("after X seconds, if Bitcoin has moved Y dollars against us, sell").  Same simulated fills of the live setup, with three corrections:
 1. sale price: a print's timestamp is its block time, a few seconds AFTER the trade was agreed (seen in dip_gap.py), so the first sale stamped right at the
    decision moment is a price from before it.  Here the sale is the first one stamped at least 5 s after the decision (the old study used 0 s).
 2. trading hours only (weekends + weekday 16-24 UTC), which is what the bot trades now; all hours as a cross-check.
 3. the rule is CHOSEN on one period and JUDGED on the other, instead of reporting the best of 180 cells on everything.
Also asked more directly: whenever our shares could be sold for p, how often do they still win?  If that is above p in every state, no sell rule can help.
  python research/deep/stoploss_review.py"""
import sys, glob, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/btcmovement2")
from btcmodel.timeline import load_timeline
Z = np.load("/home/user/data/deep/stoploss_BTC.npz"); MARKS = Z["marks"]; W = Z["W"]; side = Z["side"]; win = Z["win"]; mv = Z["move"]; d = pd.to_datetime(Z["date"]); n = len(W); pos = {int(w): k for k, w in enumerate(W)}
LAGS = (0, 5); sell = {L: np.full((n, len(MARKS)), np.nan) for L in LAGS}
for f in sorted(glob.glob("/home/user/data/polymarket/trades/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t = t[t.window_start.isin(pos)]
    if not len(t): continue
    t["rel"] = t.ts - t.window_start; t = t[(t.rel >= 0) & (t.rel < 300)].sort_values(["window_start", "ts"], kind="stable")
    for w, g in t.groupby("window_start", sort=False):
        k = pos[int(w)]; rel = g.rel.values; o0 = g.outcome_index.values == 0; buy = g.side.values == "BUY"; ours = o0 if side[k] == 0 else ~o0; ps = np.where(ours, g.price.values, 1 - g.price.values); hit = (ours & ~buy) | (~ours & buy)
        hr, hp = rel[hit], ps[hit]
        for L in LAGS:
            q = np.searchsorted(hr, MARKS + L, side="left"); okq = (q < len(hr)); okq[okq] &= hr[q[okq]] < (MARKS + L)[okq] + 20; sell[L][k, okq] = hp[q[okq]]
tl = load_timeline(); t0 = int(tl.t0_epoch); sc_ = tl["s_close"]; sig = np.array([np.nanstd(np.diff(np.asarray(sc_[int(w) - 900 - t0:int(w) - t0], dtype=float))) for w in W])      # dollars per square-root second, last 15 minutes before the open
tt = pd.to_datetime(W, unit="s"); good = (tt.dayofweek.values >= 5) | (tt.hour.values >= 16); hold = np.where(win, 0.5, -0.5); SH = 200.0; day = d.values.astype("datetime64[D]"); early = np.asarray(d < "2026-10-01"); rng = np.random.default_rng(3)
net = {L: np.clip(sell[L], 0.001, 0.999) - 0.07 * np.clip(sell[L], 0.001, 0.999) * (1 - np.clip(sell[L], 0.001, 0.999)) for L in LAGS}
def ci(x, dd_):
    g = pd.Series(x).groupby(dd_).sum().values; b = g[rng.integers(0, len(g), (2000, len(g)))].sum(1); return np.quantile(b, .025), np.quantile(b, .975)
print(f"{n} simulated fills, {good.sum()} in trading hours (win {100*win[good].mean():.1f} %).  A sale price is known for {100*np.isfinite(sell[5][good]).mean():.0f} % of the (trade, moment) pairs; where nobody sold within 20 s the trade is simply held.")
print(f"sale price stamped 0 s vs 5 s after the decision, when Bitcoin is $30+ against us: {100*np.nanmean(sell[0][(-mv >= 30)]):.1f}¢ vs {100*np.nanmean(sell[5][(-mv >= 30)]):.1f}¢ (the old study used the first)")
N = net[5]
print("\n1. WHEN OUR SHARES CAN BE SOLD FOR p (after the fee), HOW OFTEN DO THEY STILL WIN?   trading hours.  + = better to hold, - = better to sell")
rows = []
for lo, hi in ((0.02, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60)):
    r = {"sale price after fee": f"{int(100*lo)}-{int(100*hi)}¢"}
    for nm, js in (("T+30..90", (0, 1, 2)), ("T+120..180", (3, 4, 5)), ("T+210..270", (6, 7, 8))):
        k = np.zeros_like(N, bool); k[:, js] = True; k &= np.isfinite(N) & (N >= lo) & (N < hi) & good[:, None]; ii, jj = np.nonzero(k)
        if len(ii) < 40: r[nm] = "-"; continue
        e = win[ii] - N[ii, jj]; lo_, hi_ = ci(e, day[ii]); r[nm] = f"{100*win[ii].mean():.0f}% win vs {100*N[ii, jj].mean():.0f}¢ = {100*e.mean():+.1f}¢ [{100*lo_/len(e):+.1f},{100*hi_/len(e):+.1f}] ({len(ii)})"
    rows.append(r)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 70); print(pd.DataFrame(rows).to_string(index=False))
print("\n2. BY BITCOIN'S MOVE AGAINST US: win rate if held minus sale price after fee, cents per share (number of trades).  trading hours")
rows = []
for lo, hi in ((10, 20), (20, 30), (30, 50), (50, 80), (80, 120), (120, 1e9)):
    r = {"against us by": f"${lo}-{int(hi) if hi < 1e9 else '+'}"}
    for j in (1, 3, 5, 6, 7):
        k = (-mv[:, j] >= lo) & (-mv[:, j] < hi) & np.isfinite(N[:, j]) & good; r[f"T+{MARKS[j]}"] = f"{100*(win[k] - N[k, j]).mean():+.1f} ({k.sum()})" if k.sum() >= 30 else "-"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
DS = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200]; KS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]; zmv = mv / (sig[:, None] * np.sqrt(MARKS)[None, :])       # move in units of what Bitcoin normally moves in that many seconds
def single(j, thr, M, mask): trig = (-M[:, j] >= thr) & np.isfinite(N[:, j]) & mask; return np.where(trig, N[:, j] - 0.5, hold), trig
def frm(j0, thr, M, mask):
    p = hold.copy(); done = ~mask; tr = np.zeros(n, bool)
    for j in range(j0, len(MARKS)):
        trig = ~done & (-M[:, j] >= thr) & np.isfinite(N[:, j]); p[trig] = N[trig, j] - 0.5; done |= trig; tr |= trig
    return p, tr
RULES = [(f"at T+{MARKS[j]}, ${D}+ against", single, j, D, mv) for j in range(len(MARKS)) for D in DS] + [(f"from T+{MARKS[j]} on, ${D}+ against", frm, j, D, mv) for j in range(len(MARKS)) for D in DS] + \
        [(f"at T+{MARKS[j]}, {k}x normal move against", single, j, k, zmv) for j in range(len(MARKS)) for k in KS] + [(f"from T+{MARKS[j]} on, {k}x normal move against", frm, j, k, zmv) for j in range(len(MARKS)) for k in KS]
res = []
for nm, fn, j, thr, M in RULES:
    p, trig = fn(j, thr, M, good); dl = (p - hold) * SH; res.append((nm, dl[good & early].sum(), dl[good & ~early].sum(), dl[good].sum(), int(trig.sum()), p))
R = pd.DataFrame([(a, b, c, e, f) for a, b, c, e, f, _ in res], columns=["rule", "Jun1-Sep30", "Oct1-Nov7", "all", "sold"]); P = {a: p for a, _, _, _, _, p in res}
print(f"\n3. THE RULES: {len(R)} versions tested (dollar and volatility-scaled thresholds, one check or every 30 s).  Change in profit versus holding, $100 a trade, trading hours")
print(f"   positive in BOTH periods: {((R['Jun1-Sep30'] > 0) & (R['Oct1-Nov7'] > 0)).sum()} of {len(R)};  positive on everything: {(R['all'] > 0).sum()};  average change {R['all'].mean():+,.0f};  median {R['all'].median():+,.0f}")
def show(lbl, sub):
    out = []
    for _, r in sub.iterrows():
        dl = ((P[r.rule] - hold) * SH)[good]; lo_, hi_ = ci(dl, day[good]); out.append({"rule": r.rule, "trades sold": r.sold, "Jun1-Sep30 $": round(r["Jun1-Sep30"]), "Oct1-Nov7 $": round(r["Oct1-Nov7"]), "all $": round(r["all"]), "95% lo": round(lo_), "95% hi": round(hi_)})
    print(f"   -- {lbl}"); print(pd.DataFrame(out).to_string(index=False))
show("best 5 when CHOSEN on Jun 1 - Sep 30, and what they did on Oct 1 - Nov 7", R.sort_values("Jun1-Sep30", ascending=False).head(5)); show("best 5 when chosen on Oct 1 - Nov 7, and what they did on Jun 1 - Sep 30", R.sort_values("Oct1-Nov7", ascending=False).head(5))
show("your example and its neighbours", R[R.rule.isin(["at T+180, $50+ against", "at T+180, $40+ against", "at T+180, $80+ against", "at T+240, $50+ against", "from T+180 on, $50+ against", "from T+120 on, $50+ against"])])
# what the old study reported as best, re-priced
p0 = np.where((-mv[:, 5] >= 40) & np.isfinite(net[0][:, 5]) & good, net[0][:, 5] - 0.5, hold); print(f"\n   the old best cell (T+180, $40+): with the old sale price {((p0 - hold) * SH)[good].sum():+,.0f}; with the corrected sale price {R.set_index('rule').loc['at T+180, $40+ against', 'all']:+,.0f}")
a_ = np.ones(n, bool); print("   all hours, same corrected prices: " + "; ".join(f"{nm}: {((fn(j, thr, M, a_)[0] - hold) * SH).sum():+,.0f}" for nm, fn, j, thr, M in RULES if nm in ("at T+180, $50+ against", "at T+180, $40+ against", "from T+180 on, $50+ against", "at T+240, $50+ against")))

print("\n4. THE T+180 CLUSTER.  Change versus holding, $100 a trade, trading hours: Jun1-Sep30 / Oct1-Nov7 (trades sold)")
Rx = R.set_index("rule"); rows = []
for D in (30, 40, 50, 60, 80):
    rows.append({"against us by": f"${D}+", **{f"at T+{m}": f"{Rx.loc[f'at T+{m}, ${D}+ against', 'Jun1-Sep30']:+,.0f} / {Rx.loc[f'at T+{m}, ${D}+ against', 'Oct1-Nov7']:+,.0f} ({Rx.loc[f'at T+{m}, ${D}+ against', 'sold']})" for m in (120, 150, 180, 210, 240)}})
print(pd.DataFrame(rows).to_string(index=False))
def dd(x): c = np.concatenate([[0], np.cumsum(x)]); return (np.maximum.accumulate(c) - c).max()
o = np.argsort(W[good]); th = pd.Series(hold[good] * SH).groupby(day[good]).sum(); mon = pd.to_datetime(W[good], unit="s").strftime("%b"); print(f"\n   holding: total ${hold[good].sum()*SH:,.0f}, max drawdown ${dd((hold[good]*SH)[o]):,.0f}, worst day ${th.min():,.0f}, losing days {(th < 0).sum()}")
for nm in ("at T+180, $40+ against", "at T+180, $50+ against", "from T+180 on, $50+ against"):
    p = P[nm]; tp = pd.Series(p[good] * SH).groupby(day[good]).sum(); sold = (p != hold) & good; print(f"   {nm:30s}: total ${p[good].sum()*SH:,.0f}, max drawdown ${dd((p[good]*SH)[o]):,.0f}, worst day ${tp.min():,.0f}, losing days {(tp < 0).sum()} | of the {sold.sum()} sold, {int(win[sold].sum())} would have won ({100*win[sold].mean():.0f} %), sold for {100*(p[sold]+0.5).mean():.1f}¢ after fee | by month: " + " ".join(f"{m} {((p - hold)[good][mon == m]).sum()*SH:+,.0f}" for m in ("Jun", "Jul", "Aug", "Sep", "Oct", "Nov")))
