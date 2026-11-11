"""Execution-policy lab for rule A (BTC): one simulator on the real trade tape for staged entries.
A policy = decision time, score tier, a list of resting-bid stages (start, end, bid), an optional market buy for what is left (time, max price),
an optional check with the T-45 model at T-43 (cancel when it disagrees / take when it agrees strongly), and the sell price.
Fills: a resting bid fills when a taker sells our side at <= bid; when our side is already offered at <= bid at a stage start the offer is taken
(fee 0.07 p (1-p)); a market buy pays the offer + fee.  The sell counts only after the fill.  $100 a trade: 'full' = whole stake filled,
'real' = stake limited to the dollars that traded at our price.  Selection on Aug 20 - Sep 30, report on Oct 1 - Nov 7 and Jun 1 - Aug 19.
  python research/deep/policy_lab.py"""
import sys, os, glob, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, sims_cached, DEEP
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; ASSET = os.environ.get("ASSET", "BTC"); STAKE = 100.0; TP0 = 0.85 if ASSET == "BTC" else 0.80      # ASSET=ETH: T-45 model only (no T-120 feature history)
TRADES = "/home/user/data/polymarket/trades" if ASSET == "BTC" else "/home/user/data_eth/polymarket/trades"; CACHE = f"{DEEP}/policy_prints_{ASSET}.pkl"
W = load(ASSET, ["m2_-45_close_p"]); mon = W.month.values; n = len(W); SN = sims_cached(ASSET, None).reindex(W.index); S85 = sims_cached(ASSET, TP0).reindex(W.index)
upwin = SN.pnl_up.values > 0; ok = np.isfinite(SN.pnl_up.values); date = pd.to_datetime(W.date.values); Tix = W.index.values
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
SC = {-45: pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_ph5.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values}
if os.path.exists(f"{DEEP}/ourcase_{ASSET}_d-120_ph3.parquet"): SC[-120] = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_d-120_ph3.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values
XS = os.environ.get("XSCORE", "")      # experiment: trade this asset's markets on ANOTHER asset's T-120 score (e.g. ASSET=ETH XSCORE=BTC)
if XS: SC[-120] = pd.read_parquet(f"{DEEP}/ourcase_{XS}_d-120_ph3.parquet").set_index("T")["lgb_reg300"].reindex(W.index).values; CACHE = f"{DEEP}/policy_prints_{ASSET}_x{XS}.pkl"
THR = {}                                                        # |score| threshold per (decision, tier) and month, from the months before (walk-forward)
for dec, sc in SC.items():
    for q in (0.85, 0.70, 0.50):
        th = np.full(n, np.inf)
        for mo in MONTHS:
            te = mon == mo; th[te] = np.quantile(np.abs(sc[before(mo) & np.isfinite(sc)]), q)
        THR[dec, q] = th
need = ok & np.any([np.abs(SC[d]) >= THR[d, 0.50] for d in SC], axis=0)
if os.path.exists(CACHE): P = pickle.load(open(CACHE, "rb"))
else:
    P = {}; want = set(Tix[need].tolist())
    for f in sorted(glob.glob(f"{TRADES}/*.parquet")):
        t = pd.read_parquet(f, columns=["ts", "side", "outcome_index", "price", "size", "window_start"]); t = t[t.window_start.isin(want)]
        if not len(t): continue
        t["rel"] = t.ts - t.window_start; t = t[(t.rel >= -200) & (t.rel <= 300)].sort_values(["window_start", "ts"], kind="stable")
        for w, g in t.groupby("window_start", sort=False):
            rel = g.rel.values; o0 = g.outcome_index.values == 0; buy = g.side.values == "BUY"; pu = np.where(o0, g.price.values, 1 - g.price.values); lu = (o0 & buy) | (~o0 & ~buy); usd = g.price.values * g["size"].values
            a = rel <= 60; b = ~a
            P[w] = (rel[a].astype(np.float32), pu[a].astype(np.float32), lu[a], usd[a].astype(np.float32), float(pu[b & lu].max()) if (b & lu).any() else 0.0, float(pu[b & ~lu].min()) if (b & ~lu).any() else 1.0)
    pickle.dump(P, open(CACHE, "wb"), protocol=4)
def side_view(w, s):
    rel, pu, lu, usd, mxu, mnu = P[w]
    return (rel, pu, lu, usd, mxu) if s == 0 else (rel, 1 - pu, ~lu, usd, 1 - mnu)
def offer(rel, ps, lift, t):
    a = lift & (rel < t) & (rel >= t - 60)
    if a.any(): return ps[a][-1]
    b = lift & (rel >= t) & (rel < t + 8); return ps[b][0] if b.any() else np.nan
VA = [0.0]      # side channel: $ volume available to us after the queue ahead (set by the last maker fill)
def book_ask(rel, ps, lift, t, back=120):
    """Our side's best offer just before t, read off the most recent print of either kind: a market buy of our side at p means the offer was p;
    a market sell of our side at p means the bid was p, so the offer one tick above (these books are one tick wide)."""
    a = (rel < t) & (rel >= t - back)
    if not a.any(): return np.nan
    j = np.flatnonzero(a)[-1]; return float(ps[j]) if lift[j] else float(ps[j]) + 0.01
def simulate(w, s, stages, take, s45, cross="take", fresh=60, queue=0.0, ask_rule="lift", worst_px=False):
    """-> (entry, fee, fill time, $ volume at our price, kind) or None.  stages: [(t0, t1, bid)], take: (time, cap) or None,
    s45: None or dict(cancel=bool, take_cap=float or None, agree=bool, strong=bool) applied at T-43."""
    rel, ps, lift, usd, mxpost = side_view(w, s)
    def taker(t, cap):
        if ask_rule == "last":      # audit of ALLOW_TAKER: crossing judged from the last print before placement (matches the refusal rate seen live)
            a = book_ask(rel, ps, lift, t)
            if np.isfinite(a) and a <= cap + 1e-6:
                a = cap if worst_px else max(a, 0.01); q = lift & (ps <= cap + 1e-6) & (rel >= t - 2) & (rel < t + 8); return (float(a), 0.07 * a * (1 - a), t, max(float(usd[q].sum()), 5.0), "taker")
            return None
        a = offer(rel, ps, lift, t)
        if fresh < 60 and not (lift & (ps <= cap + 1e-6) & (rel >= t - fresh) & (rel < t + fresh)).any(): a = np.nan      # audit option: a cross needs fresh evidence (a market buy at <= cap within +-fresh s)
        if np.isfinite(a) and a <= cap + 1e-6:
            q = lift & (ps <= cap + 1e-9) & (rel >= t - 2) & (rel < t + 8); return (float(a), 0.07 * a * (1 - a), t, max(float(usd[q].sum()), 5.0), "taker")
        return None
    cut = 1e9
    if s45 is not None and s45["cancel"] and not s45["agree"]: cut = -43
    for (t0, t1, bid) in stages:
        if t0 >= cut: break
        t1 = min(t1, cut); r = taker(t0, bid) if cross != "rest" else None
        if r: return r if cross == "take" else None            # cross: "take" = pay the offer + fee, "skip" = post-only refused and no order, "rest" = the bid rests anyway (audit)
        hit = ~lift & (ps <= bid + 1e-6) & (rel >= t0) & (rel < t1)
        if hit.any():
            VA[0] = float(usd[hit].sum())
            if queue <= 0: return (bid, 0.0, float(rel[hit].min()), float(usd[hit].sum()), "maker")
            # audit option (queue position): filled only once a print trades THROUGH our price or `queue` dollars have traded at it
            j = np.flatnonzero(hit); cum = np.cumsum(usd[j]); okq = (ps[j] < bid - 1e-6) | (cum >= queue)
            if okq.any():
                a = int(np.argmax(okq)); k = j[a]      # $ left for us: what trades at our price from the print that reaches us onwards, less the part of the queue still ahead
                VA[0] = float(max(cum[-1] - queue, 0.0)) if cum[a] >= queue else float(usd[j[a:]].sum())
                return (bid, 0.0, float(rel[k]), float(usd[hit].sum()), "maker")
        if s45 is not None and s45.get("take_cap") and t0 < -43 <= t1 and s45["agree"] and s45["strong"]:      # (only checked when a stage spans T-43)
            pass
    if s45 is not None and s45.get("take_cap") and s45["agree"] and s45["strong"] and cut > -43:
        # the bid may still be resting after T-43 in the stage list; the strong-agreement take happens at T-43 instead of waiting
        pass
    if take is not None and take[0] < cut: return taker(take[0], take[1])
    return None
def run(dec=-120, tier=0.85, stages=((-118, -15, 0.50),), take=None, confirm=None, tp=0.85, cross="take", fresh=60, queue=0.0, tp_through=False, ask_rule="lift", worst_px=False):
    sc = SC[dec]; sel = ok & np.isfinite(sc) & (np.abs(sc) >= THR[dec, tier]); idx = np.flatnonzero(sel); out = []
    for i in idx:
        w = Tix[i]
        if w not in P: continue
        s = 0 if sc[i] > 0 else 1; s45 = None; stg = list(stages); tk = take
        if confirm is not None:
            a = SC[-45][i]; agree = bool(np.isfinite(a) and (a > 0) == (sc[i] > 0)); strong = bool(np.isfinite(a) and abs(a) >= THR[-45, confirm.get("q", 0.85)][i])
            if confirm.get("need_strong"): agree = agree and strong      # keep the bid only when the T-45 model agrees AND is itself in its top tier
            s45 = dict(cancel=confirm.get("cancel", False), agree=agree)
            if confirm.get("take_cap") is not None:      # strong agreement at T-43: stop resting, take the offer up to the cap
                if agree and strong: stg = [(t0, min(t1, -43), b) for (t0, t1, b) in stg if t0 < -43]; tk = (-43, confirm["take_cap"])
        r = simulate(w, s, stg, tk, s45, cross, fresh, queue, ask_rule, worst_px)
        if r is None: continue
        e, fee, ft, vol, kind = r; rel, ps, lift, usd, mxpost = side_view(w, s); q = lift & (rel > ft); mx = max(float(ps[q].max()) if q.any() else 0.0, mxpost)
        win = upwin[i] if s == 0 else not upwin[i]; hit_tp = tp is not None and (mx > tp + 1e-6 if tp_through else mx >= tp - 1e-6)      # 1e-6: prints are stored as float32
        v = ((tp - e) if hit_tp else ((1 - e) if win else -e)) - fee
        t_in = stg[0][0] if stg else (tk[0] if tk else np.nan); a0 = offer(rel, ps, lift, t_in) if np.isfinite(t_in) else np.nan
        out.append((i, v, e, min(STAKE, vol), kind == "taker", ft, a0, abs(sc[i]) / THR[dec, tier][i], vol, VA[0] if kind == "maker" else vol))
    o = np.array(out, dtype=float); return dict(i=o[:, 0].astype(int), v=o[:, 1], e=o[:, 2], cap=o[:, 3], tk=o[:, 4] > 0, ft=o[:, 5], a0=o[:, 6], rel_score=o[:, 7], vol=o[:, 8], vol_after=o[:, 9], nsig=len(idx))
PER = {"Jun1-Aug19": date < "2026-08-20", "tune": (date >= "2026-08-20") & (date <= "2026-09-30"), "TEST": date >= "2026-10-01", "ALL": np.ones(n, bool)}
TABLE = float(np.nansum(np.where(SC[-45] > 0, S85.pnl_up.values, S85.pnl_dn.values)[ok & np.isfinite(SC[-45]) & (np.abs(SC[-45]) >= THR[-45, 0.85])]) * 200)
def summ(r, per="ALL"):
    m = PER[per][r["i"]]; full = r["v"][m] * STAKE / r["e"][m]; real = r["v"][m] * r["cap"][m] / r["e"][m]
    return dict(n=int(m.sum()), win=float((r["v"][m] > 0).mean()), c=float(r["v"][m].mean() * 100), full=float(full.sum()), real=float(real.sum()), tk=float(r["tk"][m].mean()))
if __name__ == "__main__":
    print(f"backtest table (T-45, top 15 %, every signal at 0.50, sell 0.85): ${TABLE:,.0f}; windows cached {len(P)}")
    for nm, kw in (("live Nov 10: T-45, 0.50 to T-30, no fee", dict(dec=-45, stages=((-43, -30, 0.50),))), ("T-120, 0.50 to T-15, take when crossing", dict())):
        r = run(**kw); s = summ(r); print(f"  check {nm}: {s['n']} trades, win {s['win']:.1%}, full ${s['full']:,.0f}, real ${s['real']:,.0f}")
