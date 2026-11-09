"""Deep dive 3: from signals to a trading rule.  Per asset, all signals are out of sample for the month they are used in:
  m2   frozen close head @T-45 (logit)            z5   second model 5-min z @T-60
  new  purpose-built TWAP model (ourcase_model)   bpt  retargeted panel model (ourcase_bp), if present      bk  book-depth imbalance 1 %
Scores:  each single signal;  ens = sum of the signals each divided by its own std (std from the months before the month scored);
         stack = logistic regression on the same inputs, walk-forward by month (Jun / Jul predicted backwards from Aug-Nov).
Rule:    side = sign(score); trade the windows whose |score| is above the (1 - f) quantile of |score| in the months before
         (f = fraction of windows traded); optional votes: z5 and / or book must agree with the side.
Reported per period with the live exits (ASSET's sell limit, else settle), $100 per trade, uncapped and capacity-capped.
  ASSET=BTC NEW=ph5book:lgb_reg300 python research/deep/strategy_opt.py"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); pd.set_option("display.width", 280)
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "/home/user/polybtcmodel/research/deep"); from common import load, auc, sims_cached, capacity, book, PER, DEEP  # noqa: E402
ASSET = os.environ.get("ASSET", "BTC"); NEW = os.environ.get("NEW", "ph5book:lgb_reg300"); TP = {"BTC": 0.85, "ETH": 0.80}[ASSET]; STAKES = (100, 150) if ASSET == "BTC" else (25, 50)
W = load(ASSET, ["m2_-45_close_p", "bp_-60_z_s5"]); T = W.index.values.astype(np.int64); y = W.up.values
lz = lambda q: np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
SIG = {"m2": lz(W["m2_-45_close_p"].values), "z5": W["bp_-60_z_s5"].values, "bk": book(ASSET, T)}
tag, col = NEW.split(":"); nw = pd.read_parquet(f"{DEEP}/ourcase_{ASSET}_{tag}.parquet").set_index("T")[col]; SIG["new"] = nw.reindex(T).values - (0.5 if col.startswith(("logit", "lgb_bin")) else 0.0)
if os.path.exists(f"{DEEP}/ourcase_bp_{ASSET}.parquet"): SIG["bpt"] = pd.read_parquet(f"{DEEP}/ourcase_bp_{ASSET}.parquet").set_index("T")["bp_twap_reg"].reindex(T).values
MONTHS = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]; mon = W.month.values
def before(mo): return np.isin(mon, MONTHS[:MONTHS.index(mo)]) if mo not in ("Jun", "Jul") else np.isin(mon, ["Aug", "Sep", "Oct", "Nov"])
def ens(keys):
    s = np.zeros(len(W))
    for mo in MONTHS:
        te = mon == mo; tr = before(mo)
        for k in keys: s[te] += np.nan_to_num(SIG[k][te]) / np.nanstd(SIG[k][tr])
    return np.where(np.all([np.isfinite(SIG[k]) for k in keys if k != "bk"], axis=0), s, np.nan)
def stack(keys):
    s = np.full(len(W), np.nan); X = np.column_stack([SIG[k] for k in keys]); ok = np.isfinite(X).all(1); coefs = {}
    for mo in MONTHS:
        te = (mon == mo) & ok; tr = before(mo) & ok; sd = X[tr].std(0); lr = LogisticRegression(C=1.0, max_iter=300).fit(X[tr] / sd, y[tr]); s[te] = lr.decision_function(X[te] / sd); coefs[mo] = dict(zip(keys, lr.coef_[0].round(3)))
    return s, coefs
SCORES = {k: v for k, v in SIG.items()}
SCORES["ens(m2,z5)"] = ens(["m2", "z5"]); SCORES["ens(new,z5)"] = ens(["new", "z5"]); SCORES["ens(new,z5,m2)"] = ens(["new", "z5", "m2"]); SCORES["ens(new,z5,bk)"] = ens(["new", "z5", "bk"])
if "bpt" in SIG: SCORES["ens(new,bpt)"] = ens(["new", "bpt"]); SCORES["ens(new,bpt,z5)"] = ens(["new", "bpt", "z5"])
SCORES["stack(m2,z5)"], _ = stack(["m2", "z5"]); SCORES["stack(new,z5,m2,bk)"], CO = stack(["new", "z5", "m2", "bk"])
print(f"{ASSET}: stack(new,z5,m2,bk) coefficients on standardized inputs by month:", CO)
rows = []
for nm, s in SCORES.items():
    r = {"score": nm}
    for mo in MONTHS: r[mo] = auc(y[mon == mo], s[mon == mo])
    r["Jun-Nov"] = auc(y, s); rows.append(r)
print("\n== AUC vs the Polymarket outcome =="); print(pd.DataFrame(rows).set_index("score").round(3).to_string())

S = sims_cached(ASSET, TP).reindex(W.index); cap = capacity(ASSET).reindex(W.index).fillna(0.0).values; date = W.date.values; per = W.period.values
def evaluate(name, score, frac, votes=()):
    side = np.sign(score); sel = np.zeros(len(W), bool)
    for mo in MONTHS:
        te = mon == mo; tr = before(mo) & np.isfinite(score); thr = np.quantile(np.abs(score[tr]), 1 - frac) if frac < 1 else 0.0; sel[te] = np.isfinite(score[te]) & (np.abs(score[te]) >= thr)
    for v in votes: sel &= np.isfinite(SIG[v]) & (np.sign(SIG[v]) == side)
    pnl = np.where(side > 0, S.pnl_up.values, S.pnl_dn.values); sel &= np.isfinite(pnl) & (side != 0); out = []
    for p_ in PER:
        m = sel & (per == p_); x = pnl[m]; nd = len(pd.date_range(*PER[p_]))
        if m.sum() == 0: continue
        r = dict(rule=name, period=p_, n=int(m.sum()), per_day=m.sum() / nd, win=(x > 0).mean(), edge_c=x.mean() * 100, dir_acc=((y[m] == 1) == (side[m] > 0)).mean())
        for st in STAKES:
            usd = x * 2 * np.minimum(st, cap[m]); d = pd.Series(usd).groupby(date[m]).sum().reindex(pd.date_range(*PER[p_]), fill_value=0.0); eq = d.cumsum()
            r[f"${st}/day"] = usd.sum() / nd; r[f"maxdd${st}"] = (eq.cummax() - eq).max(); r[f"worst${st}"] = d.min()
        r["uncapped$100/day"] = x.sum() * 200 / nd; out.append(r)
    return out, sel, side
pm = W["m2_-45_close_p"].values; z5 = SIG["z5"]
def frozen(book_vote):
    sel = np.isfinite(pm) & (np.abs(pm - 0.5) >= 0.05) & np.isfinite(z5) & (np.sign(z5) == np.sign(pm - 0.5))
    if book_vote: sel &= np.isfinite(SIG["bk"]) & (np.sign(SIG["bk"]) == np.sign(pm - 0.5))
    sc = np.where(sel, pm - 0.5, np.nan); return evaluate("FROZEN A" + (" + book vote" if book_vote else ""), sc, 1.0)[0]
ROWS = frozen(False) + frozen(True); KEEP = {}
for nm in [k for k in SCORES if k not in ("bk",)]:
    for frac in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0):
        for votes in ((), ("z5",), ("bk",), ("z5", "bk")):
            if votes and nm not in ("m2", "new", "ens(new,z5)", "ens(new,z5,m2)") : continue
            if "z5" in votes and "z5" in nm: continue
            o, sel, side = evaluate(f"{nm} | top {int(frac * 100)}%" + (f" | votes {'+'.join(votes)}" if votes else ""), SCORES[nm], frac, votes); ROWS += o; KEEP[o[0]["rule"] if o else ""] = (sel, side)
E = pd.DataFrame(ROWS); E.to_csv(f"/home/user/polybtcmodel/reports/deep_strategy_{ASSET}.csv", index=False)
st = STAKES[0]
def show(rules, title):
    print(f"\n== {title} ==")
    for metric in ("per_day", "win", "edge_c", f"${st}/day", f"maxdd${st}"):
        print(f"-- {metric} --"); print(E[E.rule.isin(rules)].pivot(index="rule", columns="period", values=metric).reindex(rules)[list(PER)].round(3 if metric == "win" else 1).to_string())
# selection on tune only: best $/day (capped, first stake) among rules with tune win >= frozen's tune win; and best $/day overall
t = E[E.period == "tune"].set_index("rule"); fz = t.loc["FROZEN A"]
best_usd = t[f"${st}/day"].sort_values(ascending=False).head(12).index.tolist(); best_win = t[(t.n >= 300) & (t[f"${st}/day"] >= fz[f"${st}/day"])].sort_values("win", ascending=False).head(8).index.tolist()
show(["FROZEN A", "FROZEN A + book vote"] + best_usd, "top 12 rules by TUNE $/day (capacity-capped), with the frozen rules")
show(["FROZEN A", "FROZEN A + book vote"] + best_win, "top 8 rules by TUNE win rate among those earning at least the frozen rule's $/day on tune")
pd.to_pickle({k: v for k, v in KEEP.items() if k in set(best_usd + best_win)}, f"{DEEP}/deep_rules_{ASSET}.pkl")
