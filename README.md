# polybtcmodel — Polymarket "BTC Up or Down 5m" trading system built on three BTC forecasters

This repository turns the three Bitcoin forecasting repositories (`btc-predict`, `btc-movement`,
`btcmovement2`) into an executable, back-tested trading strategy for Polymarket's 5-minute
BTC Up/Down markets. Everything here was built **without modifying the original repositories**:
their frozen models are called through their own public interfaces, and a retrained
`btc-predict` variant lives only in this repo's scripts.

Data: 95 days of Polymarket history (27,334 markets, 44 M trade prints, 2026-08-01 → 2026-11-04)
and Binance 1-second spot/perp data (2026-02-01 → 2026-11-03). Polymarket data was used **only**
to simulate trades and to evaluate; the BTC models never saw it.

**Headline (untouched test period 2026-10-01 → 2026-11-03, 34 days, after fees and slippage):**

| strategy | trades/day | avg P&L per share | return on stake | win rate | Sharpe (daily) | t-stat | max drawdown ($100/trade) |
|---|---|---|---|---|---|---|---|
| **B: open mispricing** (δ = 0.12, TP 0.80 / SL 0.10 / time-stop T+240) | 22 | **+5.6 ¢** | **+11.2 %** | 65 % | 14.5 | 4.6 | $519 |
| B2: same, δ = 0.08 (more trades) | 55 | +2.7 ¢ | +5.4 % | 64 % | 11.0 | 3.5 | $1,050 |
| A: pre-open directional (T−45, TP 0.80 / SL 0.20 / T+240) | 59 | +2.3 ¢ | +4.5 % | 57 % | 10.7 | 3.2 | $891 |
| A + B combined ($100 per trade) | 80 | — | **$17.8 k over 34 days** | — | 18.2 | — | $500 |

Both strategies were selected on the tuning period (2026-08-20 → 2026-09-30) and then run once on the
test period. They are also positive on the earlier, structurally different August 1–19 period.
Full tables: `reports/summary.json`, `reports/B_exit_grid.csv`, trade-level records `reports/trades_*.csv`.

---

## 1. What was learned about the three models

| repo | what it predicts | frozen artifact used | strength | weakness for this task |
|---|---|---|---|---|
| `btcmovement2` | P(S ≥ P0) at 10 s / 15 s / 30 s / 60 s / close, at every second of a 5-min window, plus touch & first-passage probabilities | `models/wf5_20260430` (trained through 2026-04-30; regime-aware) | best 5-minute close head (AUC 0.57 at the effective open, 0.61 ten seconds later); explicit reference-relative framing matches the market exactly | at the window open its 5-minute direction is only AUC ≈ 0.53–0.54 (an efficient market) |
| `btc-movement` | same questions with a structural (vol × drift × Student-t) stack | `artifacts/models` (trained Feb 2024 → Feb 2026) | very similar skill; useful as a second opinion; stacking adds ~0.005 AUC | essentially the same signal (perp→spot lead-lag) |
| `btc-predict` | minute-boundary forecasts of the next 10 s … 5 min (spot & perp) with cross-venue satellites | retrained here walk-forward (core features, cut-off before each 2-month block) because the shipped production models were trained through the test period | only model with a native 5-minute horizon starting at T−60; adds a little to the stack | minute-boundary only; satellites (Coinbase, alts) not rebuilt here |

All three reproduce their published skill on the same Binance data (e.g. btc-predict spot +15 s hit
rate 60 %, btcmovement2 10 s direction AUC 0.66). All three share the same core mechanism, the
perp-leads-spot lead-lag; it is **worth ~62 % accuracy over 10 seconds and ~53 % over 5 minutes**.

## 2. What was learned about the market (the facts that decide the strategy)

1. **The reference price is sampled before the open.** Polymarket resolves on Chainlink BTC/USD.
   Matching 27,000 `priceToBeat` values against Binance shows the reference equals Binance **≈ 30 s
   before the nominal window start** (20–45 s, drifting day by day; no lag in early August, lag
   from Aug 7 on). The end price is lagged the same way. Resolution agrees 93 % with
   `sign(Binance[T+270] − Binance[T−30])` and 99.7 % when that move exceeds 5 bp.
   (`research/ref_timing3.py`)
2. **"Before the open" already contains informed flow.** The share price leaves 0.50 at a median of
   T−25 s (10 % of windows by T−45). Fills at 0.50 after ~T−40 are made against people who already
   know the direction. Fills before T−45 are clean (taker buys in that interval win 50–52 %).
3. **The market prices the displacement from the reference, not the drift.** Knowing the exact
   Chainlink reference adds nothing over the market price (`research/exact_ref.py`), but the model's
   drift (perp lead-lag, flow) does: when model and market disagree by > 0.05 near the open, the
   realized frequency moves 6–9 ¢ in the model's direction (`research/market_vs_model.py`).
4. **Tight TP/SL cannot work.** With TP 0.60 / SL 0.45 the break-even P(TP first) is 43 %; the
   unconditional value is 38 % and even with the *outcome known* it is ~42 %, because the first
   1–3 bp after the reference is a coin flip. Wide stops with a late time-stop keep the directional
   edge and pay almost no fee (fee = 0.07·p(1−p) per share, taker only; ≈ 0 near 0 or 1).
5. **No reversal/"close pressure" edge at 5 minutes** (model-free, `research/close_pressure.py`).
6. Pre-open prices sit in 0.48–0.52 (92 % of pre-open volume), i.e. the pre-open market carries no
   information (corr with outcome 0.02).

## 3. The strategies

### B — open mispricing (primary)
* At the nominal open **T**, run the btcmovement2 close head on the **effective window** (timeline
  shifted by 30 s so P0 = Binance open of second T−30, elapsed 30 s; data < T only).
* Take the last Polymarket print (DOWN prints converted to UP-equivalent `1 − p`).
* If `|p_model − p_market| ≥ 0.12`, rest a **maker** bid on the model's side at the last print for
  **10 s**; if unfilled, cancel (fills 60–65 % of signals).
* Exits: resting sell (maker) at **0.80**; stop at **0.10** (sell at market when a print ≤ 0.10);
  **time-stop at T+240 s** (sell at market). Never hold to settlement.
* Why it works: the market reacts to the displacement but under-weights the model's short-horizon
  drift; a resting bid at the current price is not adversely selected because the mismatch is a
  level mismatch that persists for seconds.
* Robust on the test period in every slice (vol terciles, weekday/weekend, all ET hour buckets, both
  sides, slippage 0–3 ¢ → +5.9 … +4.9 ¢) and on the August regime (+4.5 ¢).

### A — pre-open directional (the user's base case, conditional on clean fills)
* Decision at **T−45** from the btcmovement2 close head run as a pseudo window starting at T−45
  (data < T−45). Trade only if `|p − 0.5| ≥ 0.05` (≈ 60 windows/day).
* Buy the chosen side at **0.50 and make sure the fill happens before T−40**; cancel otherwise.
* Exits: TP 0.80 (maker), SL 0.20, time-stop T+240.
* Test: +2.3 ¢ per share (+4.5 %), 57 % win rate, t = 3.2. **Risk:** if the resting bid is only
  filled by informed flow (a 0.50 bid left open until T), the simulation with true maker fills shows
  the edge disappears (fills are adverse), see `research/h1_maker.py`. This strategy is only
  profitable with the execution the user describes (always filled at ~0.50 before T−40).

### Rejected on evidence
* Tight TP/SL (0.60/0.45) with any signal: negative or ~0 after fees.
* Pre-reference decisions earlier than T−60: AUC ≈ 0.515, not enough.
* Taker entries at the open or later (+5 s … +120 s): costs exceed the edge.
* Maker bids below the market (0.49): more adverse selection, not less.
* Combining every model head in a stacker: +0.005 AUC at best; `btcmovement2` close head alone is
  the decisive input, `btc-predict` 5-min z-score helps a little at T−60.

### Sizing and risk
* Per-trade return (B, test): mean +12 %, SD 72 % → full Kelly 0.23 of bankroll; use **¼ Kelly ≈ 5 %**
  of bankroll per trade, cap 5 %. Trades are sequential (one window at a time).
* Daily P&L at $100/trade: mean +$260, worst day −$519 (B); A and B are uncorrelated (ρ = −0.13),
  so the combination halves the drawdown relative to total P&L.
* Stop trading if the rolling 3-day P&L is negative **and** the model's 24 h skill monitor
  (`btcmove.live.RegimeMonitor`) alerts, or if the reference-lag estimate (`research/ref_timing3.py`
  logic on the last day's windows) leaves the 20–45 s band.

## 4. How the experiments were run (reproducibility)

```
pip install -r requirements.txt            # plus the three model repos on sys.path (see scripts)
python scripts/fetch_binance.py 2026-02-01 2026-11-03 4        # one download, all three repo formats
python scripts/fetch_polymarket.py 2026-08-01 2026-11-04 16    # markets + trade prints (gamma + data-api v2)
BTC_DATA_DIR=/home/user/data python -m btcmodel.timeline       # (in btcmovement2) timeline
python scripts/build_bm_data.py 2026-02 ... 2026-11             # btc-movement monthly npz, seasonal, regime
python scripts/truth.py 2026-07-25 2026-11-03                   # Binance truth per window
python scripts/score_m2.py test && python scripts/score_m2.py 2026-07-25 2026-11-03   # frozen btcmovement2
python scripts/score_bm.py test && python scripts/score_bm.py 2026-07-25 2026-11-03   # frozen btc-movement
python scripts/train_bp.py                                      # btc-predict walk-forward retrain
python scripts/build_master.py                                  # one row per window + all forecasts
python polybtc/experiments.py H1|H2|H3|H4 ; python research/h1_maker.py ; python research/h2b.py ...
python polybtc/final_report_data.py                             # locked configs -> reports/
```
`score_m2.py test` / `score_bm.py test` prove bit-for-bit equality with each repo's own live
predictor (`Predictor.predict`, `artifacts/replay_demo.jsonl`).

Protocol: forecasts are out-of-sample (models frozen before May 2026 or walk-forward); strategy
parameters were chosen on 2026-08-20 → 2026-09-30 only; 2026-10-01 → 2026-11-03 was evaluated
once; 2026-08-01 → 08-19 (different lag regime) is reported as a sanity check.

Simulation assumptions (`polybtc/polysim.py`): TP is a resting limit that fills on the first
print at/above it (the user verified this); SL and time-stop are taker: fill at the triggering
print minus 1 ¢ slippage, fee 0.07·p(1−p). Maker entries fill only when a print at/below the bid
occurs before the cancel time.

## 5. Layout

```
polybtc/          polysim.py (trade simulator) · strategy_eval.py (metrics, grids) · experiments.py (H1–H4)
                  final_strategy.py (locked configs, robustness) · final_report_data.py · live_rules.py
scripts/          data fetch (Binance shared cache, Polymarket), model scorers (btcmovement2, btc-movement,
                  btc-predict retrain), truth table, master table, replay demo
research/         every hypothesis test, kept as run (reference timing, microstructure, adverse selection,
                  model-vs-market, exact reference, maker/taker entries, regimes)
reports/          summary.json, B_exit_grid.csv, trades_*.csv
```

## 6. Going live — what still has to be verified
* The reference lag must be re-measured every day (it drifted from 0 to 30 s in August). If it
  vanishes, strategy B's timing changes (run the model at the true reference time).
* The Polymarket last print must be read from a live feed with < 1 s latency; the model needs the
  Binance 1-second bars finalised (≈ 1.5–2 s after each second, see btc-predict's skew notes), so
  in practice the decision is at T+2 s, which the backtest approximates with d = 0 (data < T) and a
  10 s resting window.
* Maker fills were simulated as "a print at/below my bid" — queue position is ignored.
* Fees: `feeSchedule` of each market (rate 0.07, exponent 1 at the time of writing); re-check.
