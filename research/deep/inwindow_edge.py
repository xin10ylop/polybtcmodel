"""Is there a tradable edge INSIDE the window when the close model disagrees with the market?  Every BTC window Jun 1 - Nov 7, each side.
At each mark: model probability of the side minus the price a market buyer actually paid just after the mark; buy when the gap >= d,
pay that price + the taker fee, hold to settlement.  $100 a trade.   python research/deep/inwindow_edge.py"""
import sys, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep")
from common import load
MARKS = [30, 60, 90, 120, 150, 180, 210, 240]; M = load("BTC", [f"m2_{e}_close_p" for e in MARKS]); P = pd.read_parquet("/home/user/data/deep/inwindow_BTC.parquet").reindex(M.index)
up = M["up"].values.astype(bool); d = pd.to_datetime(M.date.values); t = pd.to_datetime(M.index.values, unit="s"); good = (t.dayofweek.values >= 5) | (t.hour.values >= 16); alld = pd.DatetimeIndex(sorted(set(d))); ND = len(alld); rng = np.random.default_rng(6)
PER = {"Jun-Aug19": d < "2026-08-20", "Aug20-Sep30": (d >= "2026-08-20") & (d <= "2026-09-30"), "Oct-Nov7": d >= "2026-10-01"}
def legs(m, tag="a"):
    """two candidate trades per window: buy UP at the next market-buy price, or buy DOWN at 1 - the next market-sell price of UP"""
    pu = M[f"m2_{m}_close_p"].values; au = P[f"ask{tag}_{m}"].values; bd = P[f"bid{tag}_{m}"].values; out = []
    for sd, q, px, w in (("UP", pu, au, up), ("DOWN", 1 - pu, 1 - bd, ~up)):
        ok = np.isfinite(q) & np.isfinite(px) & (px > 0.02) & (px < 0.98); fee = 0.07 * px * (1 - px); v = np.where(w, 1 - px, -px) - fee; out.append((q - px, v, px, ok, w))
    return out
def stat(gap, v, px, ok, dlt, mask=None, hi=9.0):
    m = ok & (gap >= dlt) & (gap < hi)
    if mask is not None: m &= mask
    return m
print(f"{len(M)} windows, {ND} days. Columns: gap threshold d (model probability minus the price paid) -> trades/day | win rate | average price paid | cents per share after the fee | total at $100 a trade")
for tag, nm in (("a", "price paid 1-13 s after the mark"), ("b", "price paid 3-15 s after the mark (slower)")):
    print(f"\n=== {nm}, all hours ===")
    for m in MARKS:
        L = legs(m, tag); out = f"T+{m:<3d} "
        for dlt in (0.05, 0.10, 0.15, 0.20, 0.30):
            n = 0; pnl = 0.0; wn = 0; pxs = 0.0; vs = 0.0
            for gap, v, px, ok, w in L:
                k = stat(gap, v, px, ok, dlt); n += k.sum(); pnl += (v[k] * 100 / px[k]).sum(); wn += w[k].sum(); pxs += px[k].sum(); vs += v[k].sum()
            out += f"| d>={dlt:.2f}: {n/ND:>5.1f}/d {wn/max(n,1):.0%} @{pxs/max(n,1):.2f} {vs/max(n,1)*100:>+5.1f}c ${pnl:>8,.0f} "
        print(out)
print("\n=== by size of the gap (not cumulative), price paid 1-13 s after the mark: cents per share after the fee (number of trades) ===")
for m in MARKS:
    L = legs(m); out = f"T+{m:<3d} "
    for lo, hi in ((-9, -0.15), (-0.15, -0.05), (-0.05, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.25), (0.25, 9)):
        n = 0; vs = 0.0
        for gap, v, px, ok, w in L: k = ok & (gap >= lo) & (gap < hi); n += k.sum(); vs += v[k].sum()
        out += f"| {lo if lo > -9 else '':>5}..{hi if hi < 9 else '':<5}: {vs/max(n,1)*100:>+5.1f}c ({n:>5d}) "
    print(out)
def detail(nm, m, dlt, tag="a", mask=None, hi=9.0):
    tot = np.zeros(len(M)); n = 0; wn = 0
    for gap, v, px, ok, w in legs(m, tag):
        k = stat(gap, v, px, ok, dlt, mask, hi); tot[k] += v[k] * 100 / px[k]; n += k.sum(); wn += w[k].sum()
    s = pd.Series(tot).groupby(d.values).sum().reindex(alld, fill_value=0.0); bs = [s.values[rng.integers(0, ND, ND)].sum() for _ in range(2000)]; c = np.concatenate([[0], np.cumsum(s.values)])
    mo = pd.Series(tot).groupby(d.strftime("%b").values).sum()
    print(f"   {nm:58s} {n/ND:>5.1f}/day win {wn/max(n,1):.1%} total ${tot.sum():>8,.0f} [{np.percentile(bs,2.5):>8,.0f},{np.percentile(bs,97.5):>8,.0f}] maxDD ${(np.maximum.accumulate(c)-c).max():>6,.0f} | periods " + " / ".join(f"{tot[PER[k]].sum():>7,.0f}" for k in PER) + " | months " + " ".join(f"{mo.get(x, 0):>6,.0f}" for x in ("Jun", "Jul", "Aug", "Sep", "Oct", "Nov")))
print("\n=== candidates in detail ($100 a trade, hold to settlement, after the taker fee) ===")
for m in (30, 60, 90, 120):
    for dlt in (0.10, 0.15, 0.20): detail(f"T+{m}, gap >= {dlt:.2f}, all hours", m, dlt)
for m in (30, 60, 90):
    detail(f"T+{m}, gap >= 0.15, slower execution (3-15 s)", m, 0.15, "b"); detail(f"T+{m}, gap >= 0.15, trading hours only", m, 0.15, "a", good); detail(f"T+{m}, gap >= 0.15, weekday 00-16 UTC only", m, 0.15, "a", ~good)
