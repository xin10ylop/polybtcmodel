"""Rule A stake multiplier: x A_STRONG_MULT when |score| >= A_STRONG_X times the trade threshold, otherwise 1; off when A_STRONG_X = 0.   python tests/test_stake_mult.py"""
import os, sys, tempfile
os.environ.update(PAPER_DIR=tempfile.mkdtemp(prefix="mult_test_"), A_STRONG_X="2.2", A_STRONG_MULT="2"); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "polybtc")); import paper_trader as pt
R = []
def check(nm, c): R.append(bool(c)); print(("PASS " if c else "FAIL ") + nm)
thr = 0.0700
check("just below 2.2x the threshold: normal stake", pt.a_stake_mult(0.1539, thr) == 1.0); check("at 2.2x the threshold: double", pt.a_stake_mult(0.1541, thr) == 2.0); check("strong DOWN signal: double", pt.a_stake_mult(-0.25, thr) == 2.0)
check("ordinary signal: normal stake", pt.a_stake_mult(0.08, thr) == 1.0); check("no score: normal stake", pt.a_stake_mult(float("nan"), thr) == 1.0 and pt.a_stake_mult(None, thr) == 1.0)
pt.A_STRONG_X = 0.0; check("switched off (A_STRONG_X=0): always the normal stake", pt.a_stake_mult(0.5, thr) == 1.0)
print(f"{sum(R)}/{len(R)} checks passed"); sys.exit(0 if all(R) else 1)
