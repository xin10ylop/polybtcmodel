"""Rule A signal filter and T-43 check: A_MIN_X (trade only from that multiple of the tier threshold) and A_CONFIRM_STRONG (keep an unfilled bid
only when the T-45 model agrees and is in its own top 30 %).  Defaults must reproduce the previous rule.   python tests/test_strong_rule.py"""
import os, sys, tempfile, importlib
os.environ.update(PAPER_DIR=tempfile.mkdtemp(prefix="strong_test_")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc")); import paper_trader as pt
R = []
def check(nm, c): R.append(bool(c)); print(("PASS " if c else "FAIL ") + nm)
thr, t45 = 0.0700, 0.0957
check("defaults: trade from the tier threshold", pt.a_trade_ok(0.0701, thr) and pt.a_trade_ok(-0.0701, thr) and not pt.a_trade_ok(0.0699, thr) and not pt.a_trade_ok(float("nan"), thr) and not pt.a_trade_ok(None, thr))
check("defaults: T-43 keeps the bid when the T-45 model agrees, however weakly", pt.a_keep_bid("UP", 0.001, t45) and pt.a_keep_bid("DOWN", -0.001, t45))
check("defaults: T-43 cancels when the T-45 model disagrees; no T-45 score keeps the bid", not pt.a_keep_bid("UP", -0.2, t45) and not pt.a_keep_bid("DOWN", 0.01, t45) and pt.a_keep_bid("UP", float("nan"), t45))
pt.A_MIN_X = 1.7; pt.A_CONFIRM_STRONG = True
check("A_MIN_X=1.7: trades from 0.1190, not below", pt.a_trade_ok(0.1191, thr) and pt.a_trade_ok(-0.1191, thr) and not pt.a_trade_ok(0.1189, thr) and not pt.a_trade_ok(-0.10, thr))
check("A_CONFIRM_STRONG: keeps the bid only on strong agreement", pt.a_keep_bid("UP", 0.0958, t45) and pt.a_keep_bid("DOWN", -0.15, t45) and not pt.a_keep_bid("UP", 0.0956, t45) and not pt.a_keep_bid("UP", -0.15, t45))
check("A_CONFIRM_STRONG: no T-45 score still keeps the bid (as before)", pt.a_keep_bid("DOWN", float("nan"), t45))
check("the doubling is still measured from the tier threshold", (pt.A_STRONG_X == 0 and pt.a_stake_mult(0.2, thr) == 1.0) or pt.a_stake_mult(0.2, thr) == pt.A_STRONG_MULT)
print(f"{sum(R)}/{len(R)} checks passed"); sys.exit(0 if all(R) else 1)
