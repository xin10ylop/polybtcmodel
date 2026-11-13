"""Adapting in a position with our own newer information.  The live rule cancels an UNFILLED bid when the T-45 model disagrees with the T-120 signal.
But a bid that filled before T-45 stays as a position whatever the T-45 model says.  Here: those filled trades split by what the T-45 model said afterwards,
and what getting out before the open would have earned:  taker = sell to the bid right away (first sale stamped 5-20 s after T-43, fee off);
maker = rest an offer at 0.50 / 0.51 until T-2 (filled only if a buyer pays more than our offer, or at our offer after $50 has traded there), unsold shares are held.
Trading hours, simulated fills of the live setup, $100 a trade (200 shares).     python research/deep/exit_on_disagree.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from policy_lab import run, side_view, SC, THR, Tix, date
r = run(dec=-120, tier=0.70, confirm=dict(cancel=True), tp=None, cross="skip", fresh=5, queue=50.0); I = r["i"]; good = np.asarray((date.dayofweek >= 5) | (pd.to_datetime(Tix, unit="s").hour >= 16))[I]
I = I[good]; win = (r["v"] > 0)[good]; ft = r["ft"][good]; s120 = SC[-120][I]; s45 = SC[-45][I]; side = np.where(s120 > 0, 0, 1); agree = np.isfinite(s45) & ((s45 > 0) == (s120 > 0)); strong45 = np.isfinite(s45) & (np.abs(s45) >= THR[-45, 0.70][I]); early = ft < -45
day = date.values.astype("datetime64[D]")[I]; per = np.where(date[I] < "2026-08-20", "Jun1-Aug19", np.where(date[I] < "2026-10-01", "Aug20-Sep30", "Oct1-Nov7")); rng = np.random.default_rng(13); hold = np.where(win, 0.5, -0.5); n = len(I)
print(f"{n} fills in trading hours, win {100*win.mean():.1f} %.  Filled before T-45: {early.sum()}; after: {(~early).sum()} (these only exist when the T-45 model agreed)")
def grp(lbl, m): print(f"   {lbl:62s} {m.sum():5d} trades, win {100*win[m].mean():5.1f} %, {100*hold[m].mean():+6.2f}¢/share if held | by period: " + "  ".join(f"{p} {100*win[m & (per == p)].mean():.1f}% ({(m & (per == p)).sum()})" for p in ("Jun1-Aug19", "Aug20-Sep30", "Oct1-Nov7")))
grp("filled before T-45, T-45 model AGREES", early & agree); grp("filled before T-45, T-45 model DISAGREES", early & ~agree); grp("   ... disagrees and is itself a strong signal (its top 30 %)", early & ~agree & strong45); grp("   ... disagrees weakly", early & ~agree & ~strong45); grp("filled after T-45", ~early)
tk = np.full(n, np.nan); mk = {0.50: np.full(n, np.nan), 0.51: np.full(n, np.nan), 0.49: np.full(n, np.nan)}
for k in np.flatnonzero(early & ~agree):
    rel, ps, lift, usd, _ = side_view(Tix[I[k]], side[k]); q = np.flatnonzero(~lift & (rel >= -38) & (rel < -23))
    if len(q): p = float(ps[q[0]]); tk[k] = p - 0.07 * p * (1 - p)
    for px in mk:
        b = lift & (rel >= -42) & (rel <= -2) & (ps >= px - 1e-6)
        if b.any() and ((ps[b] > px + 1e-6).any() or usd[b].sum() >= 50.0): mk[px][k] = px
def rule(lbl, exitpx, m):
    sold = m & np.isfinite(exitpx); p = np.where(sold, exitpx - 0.5, hold); dl = (p - hold) * 200; g = pd.Series(dl).groupby(day).sum().values; b = g[rng.integers(0, len(g), (2000, len(g)))].sum(1)
    return {"rule": lbl, "positions": int(m.sum()), "sold": int(sold.sum()), "their win % if held": round(100 * win[sold].mean(), 1), "sold at ¢": round(100 * np.nanmean(exitpx[sold]), 1), "change vs holding $": round(dl.sum()), "95% lo": round(np.quantile(b, .025)), "95% hi": round(np.quantile(b, .975)), **{p_: round(dl[per == p_].sum()) for p_ in ("Jun1-Aug19", "Aug20-Sep30", "Oct1-Nov7")}}
D = early & ~agree; pd.set_option("display.width", 250); print("\nGetting out before the open when the T-45 model disagrees with a position we already hold (total of holding everything: $%s)" % f"{hold.sum()*200:,.0f}")
print(pd.DataFrame([rule("sell to the bid at T-43 (taker)", tk, D), rule("offer at 0.51 until the open (maker)", mk[0.51], D), rule("offer at 0.50 until the open (maker)", mk[0.50], D), rule("offer at 0.49 until the open (maker)", mk[0.49], D),
                    rule("taker, only when T-45 disagrees strongly", tk, D & strong45), rule("offer at 0.50, only when T-45 disagrees strongly", mk[0.50], D & strong45)]).to_string(index=False))
