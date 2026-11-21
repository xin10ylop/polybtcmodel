"""Every combination of decision time and later check(s), on the real Polymarket tape (Jun 1 - Nov 7, trading hours, walk-forward scores):
  decision model at T-120 (bid from T-118), T-45 (bid from T-43) or T-30 (bid from T-29); signal = |score| >= m x its own top-30 % threshold (m = 1.0 or 1.6)
  checks by a later model: T-45 at T-43 and/or T-30 at T-29; 'agree' = cancel the unfilled bid when the check points the other way,
  'strong' = cancel unless the check agrees AND is in its own top 30 %.  An unfilled bid is otherwise cancelled at T-15; fills are kept to settlement.
Fills: queue-aware maker fills (refused when our side is on offer at <= 0.50 at placement), $10 (complete) and $200 (limited to the dollars sold into us).
  python research/deep/combo_lab.py"""
import sys, itertools, numpy as np, pandas as pd; sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); import policy_lab as L
L.SC[-30] = pd.read_parquet("/home/user/data/deep/ourcase_BTC_d-30_ph3.parquet").set_index("T")["lgb_reg300"].reindex(L.W.index).values
for q in (0.85, 0.70, 0.50):
    th = np.full(L.n, np.inf)
    for mo in L.MONTHS: te = L.mon == mo; th[te] = np.quantile(np.abs(L.SC[-30][L.before(mo) & np.isfinite(L.SC[-30])]), q)
    L.THR[-30, q] = th
good = np.asarray((L.date.dayofweek >= 5) | (pd.to_datetime(L.Tix, unit="s").hour >= 16)); START = {-120: -118, -45: -43, -30: -29}; CHK_T = {-45: -43, -30: -29}
per = np.where(L.date < "2026-08-20", 0, np.where(L.date < "2026-10-01", 1, 2)); day = L.date.values.astype("datetime64[D]"); ND = 160
def combo(dec, m, checks):
    sc = L.SC[dec]; sel = L.ok & good & np.isfinite(sc) & (np.abs(sc) >= m * L.THR[dec, 0.70]); out = []
    for i in np.flatnonzero(sel):
        w = L.Tix[i]
        if w not in L.P: continue
        s = 0 if sc[i] > 0 else 1; end = -15
        for (cd, how) in checks:                                     # the first failing check ends the bid at its time
            a = L.SC[cd][i]
            if not np.isfinite(a): continue
            ok_ = ((a > 0) == (sc[i] > 0)) and (how == "agree" or abs(a) >= L.THR[cd, 0.70][i])
            if not ok_: end = min(end, CHK_T[cd]); break
        L.VA[0] = 0.0; r = L.simulate(w, s, [(START[dec], end, 0.50)], None, None, "skip", 5, 50.0)
        if r is None or r[4] != "maker": continue
        out.append((i, (L.upwin[i] if s == 0 else not L.upwin[i]), L.VA[0]))
    return out
def dd(x): c = np.cumsum(x); return float((np.maximum.accumulate(np.maximum(c, 0)) - c).max()) if len(c) else 0.0
rows = []
CK = {-120: [[], [(-45, "agree")], [(-45, "strong")], [(-30, "agree")], [(-30, "strong")], [(-45, "agree"), (-30, "agree")], [(-45, "strong"), (-30, "agree")], [(-45, "strong"), (-30, "strong")]],
      -45: [[], [(-30, "agree")], [(-30, "strong")]], -30: [[]]}
for dec in (-120, -45, -30):
    for m in (1.0, 1.6):
        for ch in CK[dec]:
            o = combo(dec, m, ch)
            if not o: continue
            i = np.array([a for a, _, _ in o]); won = np.array([b for _, b, _ in o]); va = np.array([c for _, _, c in o]); od = np.argsort(L.Tix[i]); i, won, va = i[od], won[od], va[od]
            p10 = np.where(won, 10.0, -10.0) + 20 * 0.0035; s2 = np.minimum(200.0, va); p200 = np.where(won, 0.5, -0.5) * s2 / 0.5 + s2 / 0.5 * 0.0035
            lbl = f"T{dec} decides" + ("" if m == 1.0 else " (1.6x)") + (" -> " + " -> ".join(f"T{c} {'checks' if h == 'agree' else 'strong check'}" for c, h in ch) if ch else ", no check")
            rows.append({"combination": lbl, "trades/day": round(len(i) / ND, 1), "win %": round(100 * won.mean(), 1), "$10 total": round(p10.sum()), "$200 total": round(p200.sum()), "$200 max DD": round(dd(p200)), "profit/DD": round(p200.sum() / max(dd(p200), 1), 1),
                         "win Jun1-Aug19": round(100 * won[per[i] == 0].mean(), 1), "win Aug20-Sep30": round(100 * won[per[i] == 1].mean(), 1), "win Oct1-Nov7": round(100 * won[per[i] == 2].mean(), 1)})
            print(f"  done: {lbl}", flush=True, file=sys.stderr)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 80); R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv("/home/user/data/deep/combo_lab.csv", index=False)
