"""RSI momentum (the published Kalshi 15-minute idea) checked on Polymarket with executable prices.
Signal: 14-period RSI on completed 1-minute candles crosses above 60 with the 5-minute RSI above 50 -> buy Up; mirrored below 40 / below 50 -> buy Down.
Binance spot candles (the page uses Coinbase).  Entry = what the next market buyer of that side really paid from 3 s after the candle closed, plus the taker fee,
only when that price is 0.30-0.70; one entry per market; hold to settlement, or sell when the 1-minute RSI is back through 50 (real next sale price, fee off).
  A) inside the 5-minute BTC windows, Jun 1 - Nov 7      B) inside the 15-minute BTC windows, Oct 27 - Nov 7 (all the tape there is)
  C) before the open: does RSI at T-120 predict the window, and does it add anything to our own T-120 signal?
  python research/deep/rsi_check.py"""
import sys, glob, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); sys.path.insert(0, "/home/user/btcmovement2")
from policy_lab import Tix, date, upwin, ok
from btcmodel.timeline import load_timeline
tl = load_timeline(); t0 = int(tl.t0_epoch); c = pd.Series(np.asarray(tl["s_close"], dtype=np.float64)).ffill().values
def rsi(close, n=14):
    d = np.diff(close, prepend=close[0]); up = pd.Series(np.maximum(d, 0)).ewm(alpha=1 / n, adjust=False).mean().values; dn = pd.Series(np.maximum(-d, 0)).ewm(alpha=1 / n, adjust=False).mean().values; return 100 - 100 / (1 + up / np.maximum(dn, 1e-12))
R1 = rsi(c[59::60]); R5 = rsi(c[299::300]); r1 = lambda t: R1[(np.asarray(t) - t0) // 60 - 1]; r5 = lambda t: R5[(np.asarray(t) - t0) // 300 - 1]      # value of the candle that closed at time t (1-minute) / at or before t (5-minute)
fee = lambda p: 0.07 * p * (1 - p); rng = np.random.default_rng(0); pd.set_option("display.width", 250)
def report(lbl, df):
    if len(df) < 30: return {"rule": lbl, "trades": len(df)}
    e = df.pay - df.cost; g = (100 / df.cost * e).groupby(df.day).sum(); b = g.values[rng.integers(0, len(g), (2000, len(g)))].sum(1); r = {"rule": lbl, "trades": len(df), "per day": round(len(df) / df.day.nunique(), 1), "win %": round(100 * df.win.mean(), 1), "paid ¢ (fee in)": round(100 * df.cost.mean(), 1), "¢/share": round(100 * e.mean(), 2), "total $ at $100": round(g.sum()), "95% lo": round(np.quantile(b, .025)), "95% hi": round(np.quantile(b, .975))}
    if "per" in df:
        for p in df.per.cat.categories: r[p] = round(100 * e[df.per == p].mean(), 2) if (df.per == p).any() else np.nan
    return r
# ------------------------------------------------------------------ A) 5-minute windows
IW = pd.read_parquet("/home/user/data/deep/inwindow_BTC.parquet").reindex(Tix); T = Tix.astype(np.int64); m_ok = ok & (T + 300 - t0 < len(c)); per = pd.Categorical(np.where(date < "2026-08-20", "Jun1-Aug19", np.where(date < "2026-10-01", "Aug20-Sep30", "Oct1-Nov7")), ["Jun1-Aug19", "Aug20-Sep30", "Oct1-Nov7"])
def five(mode, lo=0.30, hi=0.70, exit_rsi=False, marks=(60, 120, 180, 240), up_thr=60, dn_thr=40):
    done = np.zeros(len(T), bool); rows = []
    for m in marks:
        now = r1(T + m); prev = r1(T + m - 60); r5v = r5(T); su = (prev <= up_thr) & (now > up_thr) & (r5v > 50); sd = (prev >= dn_thr) & (now < dn_thr) & (r5v < 50)
        if mode == "fade": su, sd = sd, su
        if mode == "always": su = np.ones(len(T), bool); sd = np.zeros(len(T), bool)
        for side, s in ((0, su), (1, sd)):
            px = IW[f"askb_{m}"].values if side == 0 else 1 - IW[f"bidb_{m}"].values; k = s & m_ok & ~done & np.isfinite(px) & (px >= lo) & (px <= hi); i = np.flatnonzero(k); done[i] = True
            if not len(i): continue
            win = upwin[i] if side == 0 else ~upwin[i]; pay = win.astype(float)
            if exit_rsi:
                for m2 in [x for x in (120, 180, 240) if x > m]:
                    back = (r1(T[i] + m2) < 50) if side == 0 else (r1(T[i] + m2) > 50); sp = IW[f"bidb_{m2}"].values[i] if side == 0 else 1 - IW[f"askb_{m2}"].values[i]; q = back & np.isfinite(sp) & (pay == win)      # not sold yet
                    pay = np.where(q, sp - fee(np.clip(sp, 0.01, 0.99)), pay); win = np.where(q, -1, win)
            rows.append(pd.DataFrame({"i": i, "win": (pay > px[i] + fee(px[i])).astype(float) if exit_rsi else win.astype(float), "pay": pay, "cost": px[i] + fee(px[i]), "day": date.values.astype("datetime64[D]")[i], "per": per[i], "mark": m}))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["i", "win", "pay", "cost", "day", "per", "mark"])
print("A) INSIDE THE 5-MINUTE BTC WINDOWS (Jun 1 - Nov 7), entry at the candle close 60-240 s after the open")
print(pd.DataFrame([report("RSI momentum as published (cross 60/40, 5-min RSI agrees), hold", five("mom")), report("same, sell when 1-min RSI is back through 50", five("mom", exit_rsi=True)), report("same, price ceiling 0.66 (their best setting)", five("mom", hi=0.66)),
                    report("looser: cross 55/45", five("mom", up_thr=55, dn_thr=45)), report("stricter: cross 70/30", five("mom", up_thr=70, dn_thr=30)), report("OPPOSITE side (fade the RSI cross)", five("fade")), report("no signal: buy Up at every candle close", five("always"))]).to_string(index=False))
d = five("mom"); print("   by entry time: " + "  ".join(f"T+{m}s: {100*(d.pay-d.cost)[d.mark==m].mean():+.2f}¢ ({(d.mark==m).sum()})" for m in (60, 120, 180, 240)))
# ------------------------------------------------------------------ B) 15-minute windows (full tape Oct 27 - Nov 7 only)
mk = pd.read_parquet(glob.glob("/home/user/data/pm15/btc/markets_2026*.parquet")[0]); mk = mk[mk.uma_status == "resolved"].set_index("window_start"); rows = {"mom": [], "fade": []}
for f in sorted(glob.glob("/home/user/data/pm15/btc/trades/*.parquet")):
    t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t["rel"] = t.ts - t.window_start; t = t[(t.rel >= 290) & (t.rel <= 760)].sort_values(["window_start", "ts"], kind="stable")
    o0 = t.outcome_index.values == 0; buy = t.side.values == "BUY"; t["pu"] = np.where(o0, t.price.values, 1 - t.price.values); t["lu"] = (o0 & buy) | (~o0 & ~buy)
    for w, g in t.groupby("window_start", sort=False):
        w = int(w)
        if w not in mk.index or w + 900 - t0 >= len(c): continue
        rel, pu, lu = g.rel.values, g.pu.values, g.lu.values; upw = mk.outcome_up_price[w] > 0.5; took = {"mom": False, "fade": False}
        for m in range(300, 721, 60):                                    # 600 .. 180 s to expiry, as published
            now, prev, r5v = r1(w + m), r1(w + m - 60), r5(w + m - m % 300); su = prev <= 60 < now and r5v > 50; sd = prev >= 40 > now and r5v < 50
            for mode, (a, b) in (("mom", (su, sd)), ("fade", (sd, su))):
                if took[mode] or not (a or b): continue
                side = 0 if a else 1; k = np.flatnonzero((rel > m + 3) & (rel <= m + 33) & (lu if side == 0 else ~lu))
                if not len(k): continue
                p = pu[k[0]] if side == 0 else 1 - pu[k[0]]
                if 0.30 <= p <= 0.70: took[mode] = True; win = upw if side == 0 else not upw; rows[mode].append(dict(win=float(win), pay=float(win), cost=p + fee(p), day=np.datetime64(pd.Timestamp(w, unit="s").date())))
print("\nB) INSIDE THE 15-MINUTE BTC WINDOWS (Oct 27 - Nov 7 only, 180-600 s to expiry as published)")
print(pd.DataFrame([report("RSI momentum as published, hold to settlement", pd.DataFrame(rows["mom"])), report("OPPOSITE side", pd.DataFrame(rows["fade"]))]).to_string(index=False))
# ------------------------------------------------------------------ C) before the open
S = pd.read_parquet("/home/user/data/deep/t120_lab_base.parquet")[["T", "ym", "x_twap", "base_r300"]].rename(columns={"base_r300": "s"}); S = S[(S.x_twap != 0) & np.isfinite(S.s) & (S["T"] - 420 - t0 > 0)].reset_index(drop=True)
Tn = S["T"].values.astype(np.int64); S["r1"] = r1(Tn - 120); S["r1p"] = r1(Tn - 180); S["r5"] = r5(Tn - 300); y = S.x_twap.values > 0; tt = pd.to_datetime(S["T"], unit="s"); good = ((tt.dt.dayofweek >= 5) | (tt.dt.hour >= 16)).values; sel = S.ym.values <= "2026-05"
print("\nC) BEFORE THE OPEN (RSI known at T-120): share of windows that go the way shown, Oct 2025 - May 2026 | Jun - Nov 2026")
def hr(lbl, m, d): print(f"   {lbl:78s} {100*(d[m & sel] == y[m & sel]).mean():5.1f} % ({(m & sel).sum():6d}) | {100*(d[m & ~sel] == y[m & ~sel]).mean():5.1f} % ({(m & ~sel).sum():6d})")
up_x = (S.r1p <= 60) & (S.r1 > 60) & (S.r5 > 50); dn_x = (S.r1p >= 40) & (S.r1 < 40) & (S.r5 < 50); x = (up_x | dn_x).values; dirx = up_x.values
hr("RSI cross as published, in its direction (momentum)", x, dirx); hr("1-min RSI above 60 or below 40 (level, not cross), in its direction", ((S.r1 > 60) | (S.r1 < 40)).values, (S.r1 > 50).values); hr("1-min RSI above 70 or below 30, in its direction", ((S.r1 > 70) | (S.r1 < 30)).values, (S.r1 > 50).values)
a = np.abs(S.s.values); top = a >= S.groupby("ym").s.transform(lambda v: v.abs().quantile(0.70)).values; ds = S.s.values > 0; ag1 = (S.r1.values > 50) == ds; ag5 = (S.r5.values > 50) == ds; strong = ((S.r1.values > 60) & ds) | ((S.r1.values < 40) & ~ds); against = ((S.r1.values > 60) & ~ds) | ((S.r1.values < 40) & ds)
print("   our own top-30 % signals in trading hours:"); hr("   all of them", top & good, ds); hr("   1-min RSI on the same side of 50", top & good & ag1, ds); hr("   1-min RSI on the other side of 50", top & good & ~ag1, ds); hr("   1-min RSI strongly with us (>60 / <40)", top & good & strong, ds); hr("   1-min RSI strongly against us", top & good & against, ds)
hr("   5-min RSI on the same side of 50", top & good & ag5, ds); hr("   5-min RSI on the other side of 50", top & good & ~ag5, ds); hr("   both RSIs with us", top & good & ag1 & ag5, ds); hr("   both RSIs against us", top & good & ~ag1 & ~ag5, ds)
