"""Live decision rules for the two strategies (pure functions; no I/O).

Timing facts (measured on 27,000 markets, Aug-Nov 2026; see research/ref_timing3.py):
  * Polymarket's 'price to beat' is the Chainlink BTC/USD value sampled ~30 s (20-45 s, drifts by day)
    BEFORE the nominal window start T; the end price is sampled ~30 s before T+300 likewise.
  * Hence the effective window on Binance time is [T-30, T+270]. The share price starts moving at
    T-45..T-15, i.e. 'before the open' already contains informed flow.

Inputs expected from the btcmovement2 model (models/wf5_20260430, frozen) run on a timeline shifted by
30 s so that its 5-minute window is the effective one (see scripts/score_m2.py: ShiftedTL):
  p_close(e') = P(close >= P0) with P0 = Binance spot open of second T-30, elapsed e' = seconds since T-30.
"""
from dataclasses import dataclass


@dataclass
class Order:
    side: str            # "UP" | "DOWN"
    price: float         # limit price for the token of `side`
    kind: str            # "maker" (rest) | "fixed" (user's pre-open fill)
    cancel_after_s: int  # cancel if unfilled
    tp: float            # resting sell limit on the same token
    sl: float            # stop level: sell at market when a print <= sl
    tstop_s: int         # seconds after nominal open T at which to sell at market if still open


def open_mispricing(p_model_close: float, last_print_up: float, delta: float = 0.12, cancel_after_s: int = 10,
                    tp: float = 0.80, sl: float = 0.10, tstop_s: int = 240):
    """Strategy B. At the nominal open T (model computed from bars < T, e' = 30 of the effective window):
    compare the model's P(up) with the last UP-equivalent print (DOWN prints converted as 1 - price).
    If they disagree by >= delta, rest a bid at the last print on the model's side for `cancel_after_s`
    seconds. Returns an Order or None."""
    dis = p_model_close - last_print_up
    if abs(dis) < delta:
        return None
    if dis > 0:
        return Order("UP", round(last_print_up, 3), "maker", cancel_after_s, tp, sl, tstop_s)
    return Order("DOWN", round(1.0 - last_print_up, 3), "maker", cancel_after_s, tp, sl, tstop_s)


def preopen_directional(p_model_close_at_minus45: float, thr: float = 0.05, entry_price: float = 0.50,
                        tp: float = 0.80, sl: float = 0.20, tstop_s: int = 240):
    """Strategy A. Decision at T-45 (pseudo window starting at T-45, e = 0, data < T-45): buy the side the
    5-minute close head favours if |p - 0.5| >= thr, at ~0.50, and make sure the order is FILLED BEFORE
    T-45..T-40 (cancel it at T-40 if unfilled: later fills are made against informed flow).
    Returns an Order or None."""
    p = p_model_close_at_minus45
    if abs(p - 0.5) < thr:
        return None
    side = "UP" if p > 0.5 else "DOWN"
    return Order(side, entry_price, "fixed", 5, tp, sl, tstop_s)


def stake(bankroll: float, kelly_fraction: float = 0.057, cap: float = 0.05):
    """Fraction of bankroll per trade: 1/4 Kelly from the test period (B) capped at 5%."""
    return bankroll * min(kelly_fraction, cap)
