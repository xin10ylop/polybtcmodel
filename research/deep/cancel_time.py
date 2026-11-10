import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, DEEP
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; ASSET = "BTC"
W = load(ASSET, ["m2_-45_close_p"]); S = sims_cached(ASSET, 0.85).reindex(W.index); mon = W.month.values; n = len(W)
sc = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_ph5.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
sel = np.zeros(n, bool)
for mo in MONTHS:
    te = (mon == mo) & np.isfinite(sc); sel[te] = np.abs(sc[te]) >= np.quantile(np.abs(sc[before(mo) & np.isfinite(sc)]), 0.85)
up = sc > 0; pnl = np.where(up, S.pnl_up.values, S.pnl_dn.values); m = sel & np.isfinite(pnl)
Z = np.load(f"{DEEP}/maker2_{ASSET}.npz"); pos = pd.Series(np.arange(len(Z["W"])), index=Z["W"]); b = int(np.argmin(np.abs(Z["bids"] - 0.50))); ed = list(Z["edges"])
T = W.index.values[m]; side = np.where(up[m], 0, 1); p = pnl[m]; date = W.date.values[m]; mo_ = mon[m]; ix = pos.reindex(T).values; ok = np.isfinite(ix)
T, side, p, date, mo_, ix = T[ok], side[ok], p[ok], date[ok], mo_[ok], ix[ok].astype(int); usd = Z["usd"][ix, side, b, :]; nd = 160; rng = np.random.default_rng(3)
days, inv = np.unique(date, return_inverse=True)
def ci(v):
    d = np.bincount(inv, v, len(days)); bs = [d[rng.integers(0, len(days), len(days))].sum() / nd for _ in range(2000)]; return np.percentile(bs, [2.5, 97.5])
k0 = ed.index(-45)
print("BTC, live rule, 0.50 bid placed at T-45.  Buckets: fills arriving in each stretch")
for nm, bk in (("T-45..T-30", [k0]), ("T-30..T-15", [k0 + 1]), ("T-15..open", [k0 + 2])):
    cap = usd[:, bk].sum(1); f = cap > 0; fl = np.minimum(5, cap)
    print(f"  fills arriving {nm}: in {f.mean():.0%} of signals; those trades win {(p[f] > 0).mean():.1%}, {np.average(p[f], weights=fl[f])*100:+.2f} c/share at $5")
print("\ncancel time    signals filled   c/share ($5)   $/day at $5 [95% CI]      $/day at $50 [95% CI]     $/day at $100 [95% CI]   by month c/share ($5)")
for nm, bk in (("T-30 (now)", [k0]), ("T-15", [k0, k0 + 1]), ("open (T-0)", [k0, k0 + 1, k0 + 2])):
    cap = usd[:, bk].sum(1); out = f"{nm:12s}   {(cap > 0).mean():>6.0%}"
    fl = np.minimum(5, cap); out += f"          {(fl * p).sum() / fl.sum() * 100:+.2f}  "
    for st in (5, 50, 100):
        fl = np.minimum(st, cap); v = fl / 0.5 * p; c = ci(v); out += f"     {v.sum() / nd:>6.1f} [{c[0]:>6.1f}, {c[1]:>6.1f}]"
    fl = np.minimum(5, cap); bm = {q: round(float((fl * p)[mo_ == q].sum() / max(fl[mo_ == q].sum(), 1e-9) * 100), 1) for q in MONTHS}
    print(out, "  ", bm)
