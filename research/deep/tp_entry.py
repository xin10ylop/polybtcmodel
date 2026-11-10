"""Live rule A (BTC): the sell price (0.80 / 0.85 / 0.90 / 0.95 / hold to settlement) under each entry: every signal at 0.50 (the backtest table),
market buy at the offer + fee, market buy capped at 0.53, and the resting 0.50 bid to T-15 that also takes when it would cross.
  python research/deep/tp_entry.py"""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, DEEP
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; ASSET = "BTC"; T_IN = -43; TPS = [0.80, 0.85, 0.90, 0.95, None]
W = load(ASSET, ["m2_-45_close_p"]); mon = W.month.values; n = len(W)
sc = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_ph5.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
sel = np.zeros(n, bool)
for mo in MONTHS:
    te = (mon == mo) & np.isfinite(sc); sel[te] = np.abs(sc[te]) >= np.quantile(np.abs(sc[before(mo) & np.isfinite(sc)]), 0.85)
up = sc > 0; SIM = {tp: sims_cached(ASSET, tp).reindex(W.index) for tp in TPS}
base = np.where(up, SIM[0.85].pnl_up.values, SIM[0.85].pnl_dn.values); m = sel & np.isfinite(base); T = W.index.values[m]; date = pd.to_datetime(W.date.values[m]); upm = up[m]
PN = {tp: np.where(upm, SIM[tp].pnl_up.values[m], SIM[tp].pnl_dn.values[m]) for tp in TPS}; HIT = {tp: np.where(upm, SIM[tp].tp_up.values[m], SIM[tp].tp_dn.values[m]) for tp in TPS}
P = pickle.load(open(f"{DEEP}/exec_prints_{ASSET}.pkl", "rb"))
def offer(rel, ps, lift, t, back=60, fwd=8):
    a = lift & (rel < t) & (rel >= t - back)
    if a.any(): return ps[a][-1]
    b = lift & (rel >= t) & (rel < t + fwd); return ps[b][0] if b.any() else np.nan
ent = {k: np.full(len(T), np.nan) for k in ("mkt", "mkt53", "C")}; fee = {k: np.zeros(len(T)) for k in ent}
for i, w in enumerate(T):
    if w not in P: continue
    rel, ps, lift, usd = P[w]; a0 = offer(rel, ps, lift, T_IN)
    if np.isfinite(a0):
        ent["mkt"][i] = a0; fee["mkt"][i] = 0.07 * a0 * (1 - a0)
        if a0 <= 0.53 + 1e-9: ent["mkt53"][i] = a0; fee["mkt53"][i] = fee["mkt"][i]
        if a0 <= 0.50 + 1e-9: ent["C"][i] = a0; fee["C"][i] = fee["mkt"][i]; continue
    hit = ~lift & (ps <= 0.50 + 1e-9) & (rel >= T_IN) & (rel < -15)
    if hit.any(): ent["C"][i] = 0.50
nd = 160; x = PN[0.85]
print(f"BTC, {len(T)} signals ({len(T)/nd:.1f}/day), every one bought at exactly 0.50, sell 0.85 else settle: {x.sum()*200:,.0f} $ at $100 stakes  <- the table's $66,480")
print(f"  how the 0.50 trades end: sold at 0.85 (+35 c) {HIT[0.85].mean():.1%} | held and won (+50 c) {((~HIT[0.85]) & (x > 0)).mean():.1%} | held and lost (-50 c) {((~HIT[0.85]) & (x < 0)).mean():.1%}  ->  average {x.mean()*100:+.2f} c per share")
print(f"  price a market buyer pays on these signals: mean {np.nanmean(ent['mkt']):.3f} (+{(np.nanmean(ent['mkt'])-0.5)*100:.1f} c) and fee {fee['mkt'][np.isfinite(ent['mkt'])].mean()*100:.2f} c")
PERS = {"ALL": np.ones(len(T), bool), "Jun1-Aug19": date < "2026-08-20", "tune": (date >= "2026-08-20") & (date <= "2026-09-30"), "TEST Oct1-Nov7": date >= "2026-10-01"}
rng = np.random.default_rng(4)
def line(v, f, stake=100):
    d = pd.Series(np.where(f, v, 0.0)).groupby(date.values).sum(); c = pd.Series(f.astype(float)).groupby(date.values).sum()
    bs = [d.values[j].sum() / max(c.values[j].sum(), 1) * 100 for j in (rng.integers(0, len(d), len(d)) for _ in range(1000))]
    return f"{f.sum()/nd:>5.1f}/day  {v[f].mean()*100:+.2f} c [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]"
for nm, key in (("every signal at 0.50 (the table: not obtainable)", None), ("market buy at the offer, any price, + fee", "mkt"), ("market buy only if the offer <= 0.53, + fee", "mkt53"), ("bid 0.50 to T-15, take when it crosses (recommended)", "C")):
    print(f"\n--- {nm} ---\n sell at     trades   c/share [95% CI]            win      c/share by period: Jun1-Aug19 | tune | TEST")
    for tp in TPS:
        e = np.full(len(T), 0.50) if key is None else ent[key]; fe = np.zeros(len(T)) if key is None else fee[key]; f = np.isfinite(e); v = PN[tp] + 0.50 - np.where(f, e, 0.5) - fe
        per = " | ".join(f"{v[f & np.asarray(PERS[k])].mean()*100:+.2f}" for k in ("Jun1-Aug19", "tune", "TEST Oct1-Nov7"))
        print(f"  {('settle' if tp is None else f'{tp:.2f}'):>6}   {line(v, f)}   {(v[f] > 0).mean():.1%}     {per}")
