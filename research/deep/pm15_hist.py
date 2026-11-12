"""15-minute BTC Up/Down market with the live rule, Jun 1 - Nov 7 (pre-open tape from scripts/fetch_pm15.py plus the two weeks already on disk).
Direction: BTC T-120 walk-forward score of the window starting on the quarter hour, top 30 % (threshold from earlier months, as in policy_lab); post-only bid 0.50
from T-118; cancel if the T-45 model disagrees and the bid is not filled yet, and at T-15; hold to settlement.  Queue-aware fills: filled once somebody sells
our side below 0.50 or once $Q has been sold at 0.50; the stake is limited to the dollars sold into 0.50 after that.     python research/deep/pm15_hist.py"""
import glob, numpy as np, pandas as pd
DEEP = "/home/user/data/deep"; R = "/home/user/data/pm15/btc"; MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
mk = pd.concat([pd.read_parquet(f"{R}/markets_hist.parquet"), pd.read_parquet(glob.glob(f"{R}/markets_2026*.parquet")[0])[["window_start", "condition_id", "outcome_up_price", "volume", "uma_status"]]]).drop_duplicates("window_start")
mk = mk[(mk.uma_status == "resolved") & (mk.window_start >= pd.Timestamp("2026-06-01").timestamp()) & (mk.window_start < pd.Timestamp("2026-11-08").timestamp())].set_index("window_start").sort_index(); up = mk.outcome_up_price > 0.5
s120 = pd.read_parquet(f"{DEEP}/ourcase_BTC_d-120_ph3.parquet").set_index("T")["lgb_reg300"]; s45 = pd.read_parquet(f"{DEEP}/ourcase_BTC_ph5.parquet").set_index("T")["lgb_reg300"]
sm = pd.Series(pd.to_datetime(s120.index, unit="s").strftime("%b"), index=s120.index); thr = {}
for mo in MONTHS:
    prev = MONTHS[:MONTHS.index(mo)] if mo not in ("Jun", "Jul") else ["Aug", "Sep", "Oct", "Nov"]; k = sm.isin(prev) & (s120.index >= pd.Timestamp("2026-06-01").timestamp()) & np.isfinite(s120.values); thr[mo] = s120[k].abs().quantile(0.70)
pr = [pd.read_parquet(f"{R}/pre_hist.parquet")]
for f in sorted(glob.glob(f"{R}/trades/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); pr.append(t[(t.ts - t.window_start >= -200) & (t.ts - t.window_start <= 0)])
t = pd.concat(pr).drop_duplicates(); t["rel"] = t.ts - t.window_start; t = t.sort_values(["window_start", "ts"], kind="stable"); o0 = t.outcome_index.values == 0; buy = t.side.values == "BUY"
t["pu"] = np.where(o0, t.price.values, 1 - t.price.values); t["lu"] = (o0 & buy) | (~o0 & ~buy); t["usd"] = t.price.values * t["size"].values; P = {int(w): (g.rel.values, g.pu.values, g.lu.values, g.usd.values) for w, g in t.groupby("window_start", sort=False)}
T = np.array([w for w in mk.index if w in s120.index and np.isfinite(s120[w])]); ts = pd.to_datetime(T, unit="s"); good = np.asarray((ts.dayofweek >= 5) | (ts.hour >= 16)); mon = np.asarray(ts.strftime("%b")); day = ts.values.astype("datetime64[D]"); ND = len(np.unique(day))
def sim(Q, end=-15, confirm=True):
    out = []
    for j, w in enumerate(T):
        s = s120[w]
        if abs(s) < thr[mon[j]]: continue
        sd = 0 if s > 0 else 1; won = bool(up[w]) if sd == 0 else not bool(up[w]); rel, pu, lu, usd = P.get(int(w), (np.zeros(0), np.zeros(0), np.zeros(0, bool), np.zeros(0))); po = pu if sd == 0 else 1 - pu; tb = lu if sd == 0 else ~lu
        last = np.flatnonzero((rel < -118) & (rel >= -178)); refused = len(last) > 0 and ((tb[last[-1]] and po[last[-1]] <= 0.50 + 1e-6) or (not tb[last[-1]] and po[last[-1]] < 0.495)); stop = end
        if confirm and w in s45.index and np.isfinite(s45[w]) and (s45[w] > 0) != (s > 0): stop = min(end, -44)
        k = np.flatnonzero(~tb & (rel >= -118) & (rel <= stop) & (po <= 0.50 + 1e-6)); filled = False; va = 0.0
        if len(k) and not refused:
            c = np.cumsum(usd[k]); thru = np.flatnonzero(po[k] < 0.495); a = thru[0] if len(thru) else None; b = np.flatnonzero(c >= Q); b = b[0] if len(b) else None; jj = min([x for x in (a, b) if x is not None], default=None)
            if jj is not None: filled = True; kk = np.flatnonzero(~tb & (rel >= rel[k][jj]) & (rel <= end) & (po <= 0.50 + 1e-6)); va = max(float(usd[kk].sum() - (min(Q, c[jj]) if (a is None or jj != a) else 0.0)), 0.0)
        out.append((j, won, refused, filled, va))
    return pd.DataFrame(out, columns=["j", "won", "refused", "filled", "va"])
def dd(x): c = np.cumsum(x); return float((np.maximum.accumulate(np.maximum(c, 0)) - c).max()) if len(c) else 0.0
rng = np.random.default_rng(0); pd.set_option("display.width", 250); print(f"{len(T)} resolved 15-minute markets with a score over {ND} days ({len(T)/ND:.0f}/day), {len(P)} with pre-open prints")
for hn, hm in (("TRADING HOURS (weekends + weekday 16-24 UTC)", True), ("ALL HOURS", False)):
    for Q in (10.0, 25.0):
        d = sim(Q); d = d[good[d.j.values]] if hm else d; f = d[d.filled]; j = f.j.values; w = f.won.values; se = 100 * np.sqrt(0.25 / max(len(f), 1))
        print(f"\n=== {hn}, queue ahead ${Q:.0f}: signals {len(d)/ND:.1f}/day (right {100*d.won.mean():.1f} %), refused {100*d.refused.mean():.0f} %, filled {len(f)/ND:.1f}/day; FILLED trades won {100*w.mean():.1f} % ± {se:.1f} ({len(f)} trades); not-filled signals right {100*d[~d.filled].won.mean():.1f} %")
        print("   filled win rate by period: " + "  ".join(f"{nm} {100*w[m].mean():.1f} % ({m.sum()})" for nm, m in (("Jun1-Aug19", day[j] < np.datetime64("2026-08-20")), ("Aug20-Sep30", (day[j] >= np.datetime64("2026-08-20")) & (day[j] < np.datetime64("2026-10-01"))), ("Oct1-Nov7", day[j] >= np.datetime64("2026-10-01")))) + f"   | dollars sold into us after the fill: median {f.va.median():.0f}, mean {f.va.mean():.0f}")
        rows = []
        for S in (10, 20, 50, 100, 200):
            cap = np.minimum(S, f.va.values); p = cap / 0.5 * np.where(w, 0.5, -0.5) + cap / 0.5 * 0.0035; g = pd.Series(p).groupby(day[j]).sum(); bs = g.values[rng.integers(0, len(g), (2000, len(g)))].sum(1)
            rows.append({"stake $": S, "fills completely %": round(100 * (cap >= S).mean()), "avg $ filled": round(cap.mean(), 1), "total $": round(p.sum()), "per day $": round(p.sum() / ND, 1), "95% lo": round(np.quantile(bs, .025)), "95% hi": round(np.quantile(bs, .975)), "max DD $": round(dd(p)), "worst day $": round(g.min()), **{m: round(p[mon[j] == m].sum()) for m in MONTHS}})
        print(pd.DataFrame(rows).to_string(index=False))
