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
| **B: open mispricing** (δ = 0.12, sell limit 0.85, else hold to settlement) | 22 | **+6.6 ¢** | **+13.3 %** | 66 % | 16.1 | 4.6 | $1,048 |
| B0: the same with the previous exits (TP 0.80 / SL 0.10 / T+240), kept as a shadow | 22 | +5.6 ¢ | +11.2 % | 65 % | 14.5 | 4.6 | $519 |
| D: high-confidence settlement capture (T+90, model ≥ 0.90 for a side trading in [0.80, 0.90], bid at the last print, hold to settlement) | 10 | +2.3 ¢ | +4.6 % | **89 %** | 4.5 | 1.4 | $645 |
| B2: same, δ = 0.08 (more trades) | 55 | +2.7 ¢ | +5.4 % | 64 % | 11.0 | 3.5 | $1,050 |
| A: pre-open directional (T−45, btc-predict agreement, sell limit 0.85, else hold to settlement) | 40 | +4.7 ¢ | +9.3 % | 64 % | 13.6 | 4.2 | $1,120 |
| A0: the same with the previous exits (TP 0.80 / SL 0.05 / T+240), kept as a shadow | 40 | +3.4 ¢ | +6.7 % | 64 % | 12.3 | 3.4 | $606 |
| C: "B five seconds early" (T−5, δ = 0.08, bid at the last print until the open, TP 0.80 / SL 0.05 / T+240) | 40 | +2.6 ¢ | +5.2 % | 64 % | 7.2 | 2.8 | $1,950 |
| A + B combined ($100 per trade) | 62 | — | **$17.7 k over 34 days** | — | 18.6 | — | $596 |
| **A + B + D (the running book)** | 72 | — | **$23.7 k over 34 days** | — | 19.8 | — | $873 |
| A + C + B (C first, B on the rest; C not run) | 93 | — | $27.4 k over 34 days | — | 20.9 | — | $846 |

Both strategies were selected on the tuning period (2026-08-20 → 2026-09-30) and then run once on the
test period. They are also positive on the earlier, structurally different August 1–19 period.
Full tables: `reports/summary.json`, `reports/B_exit_grid.csv`, trade-level records `reports/trades_*.csv`.
Data note: the gamma market listing is offset by one day, so the first calendar day of each fetched range (Aug 1 in the main set, Jul 1 in July) holds a single window; June 1 was re-listed with a shifted query and is complete.


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

Exits updated 2026-11-04 (`research/ideas10.py` idea 8): sell limit 0.85, otherwise hold to settlement, no stop. Better
than the previous TP 0.80 / SL 0.10 / T+240 in all three TWAP-era periods (+5.5 / +5.2 / +6.7 ¢ vs +4.5 / +4.5 / +5.6);
in July (point-price rule, 195 trades) the previous exits did better (+2.7 vs +1.3 ¢). The previous exits run as shadow B0.
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

### C — "B five seconds early" (added 2026-11-04; **switched off 2026-11-05 at the user's request**, code kept behind `ENABLED` in `paper_trader.py`)

At T−5 most of the 60-second reference window is already known and the book has usually left 0.50 on pre-open
flow. The close head re-evaluated at T−5 is compared with the last print; when they disagree by ≥ 0.08 a bid rests at
the last print on the model's side until the open (fills 56–81% of the time, median 1 s), exits TP 0.80 / SL 0.05 /
T+240. Found by a grid over decision time (T−25 … T−5), threshold, minimum tilt, entry mode and exits, selected on
tune and read on test and Aug 1–19; the T−5 cell sits on a plateau (every neighbouring threshold/stop positive in all
three periods). Fading the pre-open move (buying the cheap side when the book is tilted against the model) is weak and
rare; the signal is on the *follow* side. C overlaps with B on only a third of its windows; the combined book (C first,
B on the rest) beats B alone in all three TWAP-era periods (paired day bootstrap p = 0.015 / 0.001 / 0.026). It is the
junior stream: smaller edge per trade (+2.6 ¢ test), twice the trades, larger drawdowns ($1,950 test vs $519 for B),
and nothing in July under the old point-price rule (B stayed positive there). Paper-traded as strategy C.

### D — high-confidence settlement capture (added 2026-11-04, `research/ideas10.py` idea 6)

At T+90 the close head is much sharper (AUC ≈ 0.8). When it gives one side ≥ 0.90 while that side's last print is still
≤ 0.90, a bid rests at the last print for 10 s and the position is held to settlement (no take-profit, no stop). It wins
83 / 87 / 89 / 90 % of the time on Aug 1–19 / tune / test / July for +1.5 / +0.9 / +2.3 / +3.6 ¢ per share on a stake
near 0.85: the high-win-rate, low-profit stream the user asked for, about 10 trades a day. Its daily P&L is nearly
uncorrelated with the others. Paper-traded as strategy D.

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

**Execution layer (`polybtc/live_executor.py`, 2026-11-05).** With `LIVE=1` the paper trader hands every A / B / D decision
to an executor built on Polymarket's official Python SDK (`polymarket-client`): A rests a BUY at 0.50 at T−45 and cancels it
at T−40 if unfilled; B and D rest a BUY at the last print and cancel after 10 s; a fill gets a post-only SELL at 0.85 (A, B); the live timeline finalizes a second only once its spot kline and its perp bar have both arrived (previously the clock finalized the last second before its bars came in, so every decision saw a forward-filled spot close and a perp synthesized from it, a spurious lead-lag signal worth 0.04 on the close head in replay); real gaps are synthesized with a rolling live basis and every decision record counts the synthetic / forward-filled seconds it used (`synth_last60`, `ffill_last60`, must be 0); the audit's new `bias` section flags one-sided days, an off-range mean head and any decision on incomplete bars, and the parity section prints the signed live-minus-archive shift; every order is post-only (maker, no fee); one that would cross the book is not placed (a BUY is skipped, a SELL falls back to holding to settlement) unless `ALLOW_TAKER=1`, which re-sends it crossing and books the taker fee 0.07·p(1−p) on the fill; the tally prints the fees paid, which should read $0.00;
anything unsold at the close settles and is redeemed through the Relayer (gasless, needs a Relayer API key from
polymarket.com → Settings → API Keys). Sizing `STAKE_USD` per trade (min 5 shares), `MAX_OPEN` positions, `DAILY_LOSS_LIMIT`
(no new trades for the UTC day below it), kill file `PAPER_DIR/STOP` (cancels open buys, blocks new ones). `DRY_RUN=1`
(default) logs every order without touching the account: run a day of dry-run on the box before `DRY_RUN=0`.
Env for real orders: `POLYMARKET_PRIVATE_KEY` (signer), `POLYMARKET_WALLET_ADDRESS` (account wallet from the profile menu),
`POLYMARKET_RELAYER_API_KEY`, `POLYMARKET_RELAYER_API_KEY_ADDRESS`. Collateral is pUSD (deposits via the site auto-wrap).
Records: `live_orders.jsonl`, `live_trades.jsonl`, `live.log` next to the paper logs.


**Live perp feed (found 2026-11-05, open).** The models' perp features are venue-scaled (Binance USDT-M). The trader feeds
OKX trades; the parity audit shows the live forecasts differ from the archive replay by 0.03 on average with direction flips
on a third of A windows, and the live-path rescoring shows the perp features carry about a third of B's edge and two thirds of
A's trade volume. Until the feed is calibrated (scale factors measured against the Binance archive) or replaced and the daily
parity section is clean, the paper P&L measures the live system as it is, not the backtest.

* The reference lag must be re-measured every day (it drifted from 0 to 30 s in August). If it
  vanishes, strategy B's timing changes (run the model at the true reference time).
* The Polymarket last print must be read from a live feed with < 1 s latency; the model needs the
  Binance 1-second bars finalised (≈ 1.5–2 s after each second, see btc-predict's skew notes), so
  in practice the decision is at T+2 s, which the backtest approximates with d = 0 (data < T) and a
  10 s resting window.
* Maker fills were simulated as "a print at/below my bid" — queue position is ignored.
* Fees: `feeSchedule` of each market (rate 0.07, exponent 1 at the time of writing); re-check.

## 7. Resolution rule update (found 2026-11-04) and the improvement pass

**The markets now settle on a 60-second Chainlink TWAP** (since early September; August markets used a
point price). This is the mechanism behind the "~30 s lag": the start reference is the average of
[T−60, T], centred at T−30, and the end reference the average of [T+240, T+300]. A TWAP-based Binance
label agrees with Polymarket's resolution 96.4% of the time (point label: 93.4%). The effective-window
model used throughout is the right approximation, and the T+240 time-stop sits exactly where the end
TWAP starts.

Improvement pass for strategy A (selection on tune, confirmation on test; `research/improve_A.py`,
`research/improve_A2.py`, `research/direct_twap_model.py`, `reports/A_improve_grid.csv`):

| idea | verdict |
|---|---|
| stack m2 + bm + bp (+ partial-TWAP displacement, +short heads) | +0.005 AUC at best; richer stacks overfit tune; rejected |
| partial-TWAP displacement at T−45 (15 s of the reference already known) | no information (coef ≈ 0); rejected |
| direct LightGBM on the TWAP-sign target, trained on the 3 months available | worse than the frozen head (AUC 0.51–0.52 vs 0.52–0.53): too little history; rejected |
| trailing stops | never in the top configurations; rejected |
| decision at T−40 | better on tune, not on test, fill-timing risk; rejected |
| hour-of-day / session filters | inconsistent between tune and test; rejected |
| confidence-scaled stake | raises mean, not Sharpe; optional |
| **require btc-predict's 5-min forecast (T−60) to agree with the side** | consistent: tune +2.1¢ vs +1.5¢, test **+2.9¢ vs +2.25¢**, Sharpe 11.6 vs 10.7, win 57.5%, ~39 trades/day; **adopted** |
| TP 0.75 / SL 0.10 instead of 0.80 / 0.20 | 65% win rate, slightly lower mean; alternative for fewer losing trades |
| **exhaustive 141,120-cell grid incl. hold time every 5 s** (`research/exit_grid_time.py`, `reports/A_exit_grid_time.csv.gz`) | hold-time optimum T+210–245 (profit rises until the end TWAP starts at T+240, then decays); TP 0.79–0.80 / SL 0.01–0.05 best at every hold ≥ 200 s; short holds with low TP win 65–70% but earn < 1¢ | confirms TP 0.80 / SL 0.05 / T+240 |
| **exhaustive 1¢ grid, 9,408 cells** (`research/exit_grid_full.py`, `reports/A_exit_grid_full.csv`, `research/sl_bootstrap.py`) | TP 0.79–0.81 is a robust ridge in both periods; the 0.20 stop hurts: with TP 0.80 / time-stop T+240, SL ≤ 0.05 beats SL 0.20 in all three periods (Aug 1–19 +62¢/day p=0.01, tune +22¢/day p=0.10, test +18¢/day p=0.12 per $1 share), win 63–64% vs 57%, average loser −43¢ vs −33¢ | **adopted: TP 0.80 / SL 0.05 (catastrophe stop) / time-stop T+240** |

Strategy A (frozen, live): decision at T−45 from the btcmovement2 close head on a pseudo window starting at
T−45, |p − 0.5| ≥ 0.05, btc-predict core 5-min z-score at T−60 agreeing in sign, buy at 0.50 filled before
T−40, sell limit 0.85 (resting); if it has not filled by the close the position is held to settlement (1.00 / 0.00, no fee).
No stop and no time-stop: adopted 2026-11-04 from the settlement grid (`research/settle_exit_A.py`): TP 0.85-then-settle
beats the previous exits (TP 0.80 / SL 0.05 / T+240) in all four periods (+0.6 / +0.6 / +1.3 / +1.0 ¢ on Aug 1–19 / tune /
test / July; p = 0.004 on test) because ~80% of trades that reach the take-profit go on to resolve our way, and a ≤ 0.05
print still resolves our way often enough that the stop lost money once settlement was allowed. Previous exits: T+240 time-stop, 0.05 stop
(sell at market). Test period with these exits: +3.3¢ per share, 64% win rate, Sharpe 12.5.

## Go-live audit (2026-11-06)

| layer | test | result |
|---|---|---|
| model vs backtest, live code path | `research/replay_live_day.py`: 2026-11-04 streamed through the trader's timeline with real feed delays | A mean error 0.0019, decisions differ on 1 % of windows; B identical |
| model vs backtest, real feeds on the box | `paper_audit.py --day 2026-11-05 --since 13:20 --sample 1` (Binance REST perp, Frankfurt) | A 0.0061 (max 0.029), signed −0.0003, 1 decision of 127 differs; B 0.0055, 1 differs; btc-predict 0.012 |
| perp feed completeness | `scripts/perp_feed_check.py` on the box | 33 trades/s, 5.1 BTC/s vs archive 25 / 1.2 for the hour: complete |
| timeline integrity | `synth_last60` / `ffill_last60` on every decision record | 0 synthetic seconds on 128 windows |
| timing | audit latency section | decisions 0.37 s after schedule (p95 0.55), feeds < 1.2 s old |
| rules | audit rules section | 127 A and 256 B/C/D decisions, 0 mismatches |
| exits | audit exits section, replayed on Polymarket's trade history | 19 trades, 0 mismatches, P&L identical to the cent |
| orders | simulated exchange (partial fills, cancel/fill race, status errors, STOP, crossing post-only) | sell limit per fill chunk, race fill recorded, loop never stalls, fees $0.00 |
| SDK | `from polymarket import SecureClient, RelayerApiKey` on the box | ok |
| position cap | backtest decisions blocked by MAX_OPEN | 3 blocks 0.2 % of decisions, 4 blocks none → run with `MAX_OPEN=4` |
| daily loss limit | `research/daily_tp_A.py` | a malfunction guard only; −$60 at $10 stakes fires on ~1 day in 20 and costs little |
| withdrawals | `research/withdrawals.py` | start $300, every Sunday withdraw everything above $400 |

Known limits: a resting 0.50 bid for A and a bid at the last print for B fill only if someone sells into them (the backtest assumes the A fill and
simulates B's by later prints; the audit replays B fills with 4 disagreements in 13); the first live week measures the real fill rates. The 14 A trades
of the night of 2026-11-05/06 (5 wins) are statistically ordinary (a 14-trade stretch this bad occurs in 3.5 % of the backtest's stretches) and get the
same parity check on 2026-11-07.

## 8. Paper trading (`polybtc/paper_trader.py`)

Feeds: Binance spot and ETH 1 s klines (websocket mirror + REST backfill), OKX BTC-USDT perpetual trades
(REST polling; the Binance perpetual is geo-blocked here, OKX stands in for the lead-lag feature), the
Polymarket CLOB market websocket (prints and books). Model history from the local archive; the btc-predict
live feature path is parity-tested against training (`python polybtc/bp_live.py`). Logs: `paper/forecasts.jsonl`
(every window), `paper/paper_trades.jsonl`, `paper/paper.log`; `python polybtc/paper_trader.py --summary`
prints P&L with resolutions. Strategy A is simulated with the user's execution assumption (filled at 0.50
before T−40); strategy B's buy limit counts as filled when a print at or below the bid occurs within 10 s (queue position ignored). Both are reported with equal weight.

Running it on your own box: `scripts/setup_box.sh` (clones the four repos under `/home/user`, installs, pulls
10 days of Binance archive history, starts the trader in an auto-restart loop; works as root or as a sudo user).
Its first lines print the reachability of the Binance spot mirror, OKX, Binance's futures REST and Polymarket's CLOB.
Daily, re-run the script: it tops up the archive, rebuilds the timeline and restarts the trader.

**Perp feed (`PERP_FEED`), found 2026-11-05.** The models' perp features need Binance's own USDT-M perp trades:
* `binance` — the aggTrade websocket (`fstream.binance.com`): archive-identical. Not reachable from US hosts (HTTP 451)
  and silently empty from Germany (the handshake and subscription succeed, no data; Binance's spot streams work there).
* `binance_rest` — the same trades by REST polling (`/fapi/v1/aggTrades`, `fromId` pages, ~0.6 s cadence, ~2,000 of the
  2,400 weight/min): same venue, sizes and flow, ~0.3–0.9 s staleness on the newest second. **Used on the Frankfurt box.**
* `okx` — OKX swap trades by REST polling: reachable from the US, but OKX's price lags Binance by ~1 s and its flow is
  only loosely related (see §7's feed diagnostic): a third of A's decisions flip direction. Fallback only.
`scripts/ws_probe.py` tests the stream variants and the polling freshness from a new host in two minutes.

**Auditing the paper trader** (`polybtc/paper_audit.py`, run daily for the previous UTC day after the data refresh):
`PAPER_DIR=/home/user/paper python3 polybtc/paper_audit.py --day YYYY-MM-DD`. It checks four things and writes
`audit_<day>.json`: (1) coverage — every 5-minute window the trader was up for has an A and a B decision, on time
(A at T−44 s, B at T+1.6 s; gaps and late decisions are listed); (2) rules — a trade exists exactly when the frozen
rule says so, from the recorded inputs (`forecasts.jsonl` carries p, market, btc-predict z, agreement and the action
taken; `orders.jsonl` the B bids and their fill/cancel); (3) model parity — the live forecasts are recomputed offline
from the archive data for the same seconds (btcmovement2 close head and btc-predict z); differences come only from
the live feeds, and the report counts direction flips and trade/no-trade decisions that would differ; (4) exits —
every trade's exit and every B maker fill is replayed on Polymarket's own trade history and compared with what the
websocket produced; for A the presence of a print at ≤ 0.50 on our side in [T−45, T−40] is reported (fill plausibility).

Order mechanics (A): buy limit at 0.50 placed at T−45 (must fill before ~T−40, else cancel); after the
fill, a resting sell limit at 0.80 (TP); the stop (0.05) and the time-stop (T+240) are market sells triggered
by the bot (Polymarket has no native stop orders).

Model-improvement experiments (accepted only if tune and test both improve with the frozen configuration):

| experiment | result | verdict |
|---|---|---|
| btc-predict + Coinbase premium (`scripts/train_bp_cb.py`, `scripts/eval_bp_cb.py`) | better on Binance metrics (5-min IC 0.043 → 0.049) but no gain vs the Polymarket outcome (AUC 0.514/0.526 vs 0.514/0.528); agreement filter worse on tune; paired bootstrap CI includes 0 | rejected |
| direct LightGBM on the TWAP-sign target, 3 months (`research/direct_twap_model.py`) | AUC 0.51–0.52 vs 0.52–0.53 for the frozen head | rejected |
| regime filters (`research/edge_drivers_A.py`, `research/regime_filters_A.py`) | quiet windows: test +4.4¢ (5/31 negative days) but tune unchanged; counter-trend: small gains on both | not adopted; regime flags logged live for prospective validation |
| model-driven trade management on top of A: exit when the post-reference close head says P(our side) < q, let winners run when P > 0.9 (`research/dynamic_exit.py`, `reports/A_dynamic_exit.csv`) | average loser −43¢ → −35/−38¢ but recoveries cut too: tune +0.3¢ (p≈0.15), test −0.3¢ (p≈0.9); let-run no effect | rejected |
| pre-open Polymarket taker flow toward our side as a filter (`research/preopen_flow.py`) | higher mean per trade (tune +2.5¢ vs +2.05¢, test +3.2¢ vs +2.9¢) but removes ~40% of still-profitable trades: total daily P&L lower in all periods | not adopted (sizing tilt at most) |
| pre-open stack for A (`research/preopen_stack.py`): all 78 pre-open forecasts (btcmovement2 at −120…−45 incl. horizon heads, btc-predict at −120/−60 all horizons, btc-movement at −45, the last pre-open print) combined by a logistic and a gradient-boosting learner fitted on tune, scored by confidence on test and Aug 1–19 | in-sample it looks spectacular (GBM top 5%: 80% hit, 14¢/trade) and out of sample it is gone: every confidence bucket on test hits 52–55% and none beats the frozen rule (best logistic decile 55% hit / 63% win / 2.8¢ ≈ frozen 56% / 64% / 3.4¢); the Aug 1–19 top-5% bucket (+4.2¢, p 0.11) failed on test | rejected: pre-open directional accuracy tops out near 55–56% whichever way the forecasts are combined; the frozen rule already sits there |
| edge on the live data path (`scripts/score_m2_synth.py`, `research/live_path_edge.py`): the whole Aug–Nov period rescored with the perp synthesized from spot, i.e. the live trader's fallback when no perp feed is available | the perp features matter: A fires a third as often (404 vs 1,303 tune trades; per-trade edge kept), B keeps two thirds of its edge (+3.3 / +4.7 ¢ vs +4.8 / +6.6), D loses most of its small edge (+0.0 / +1.0 ¢ vs +0.9 / +2.3); the parity audit of 2026-11-04 (live OKX feed vs archive: A mean error 0.033, direction flips on 10 of 30 windows) shows the live feed is far from Binance | **open issue**: the live perp feed (OKX via REST polling) must be calibrated to Binance scale or replaced (Hyperliquid websocket is the candidate) and re-verified with the parity audit before real money |
| where the live perp feed breaks (`scripts/score_m2_variant.py`, `research/feed_diagnostic.py`; OKX vs Binance for one hour: `research/record_perp_feeds.py`) | OKX volume 2.3× smaller, large-trade volume 7.5× smaller, buy/sell imbalance correlation 0.22 at 1 s, and **OKX's price lags Binance by ~1 s** (same-second return correlation 0.11, OKX one second later 0.80). Rescoring 4 days with the Binance perp degraded: scale alone changes almost nothing (A mean error 0.002, 2% flips); a 1 s delay alone gives 0.012 / 13% flips; no perp at all 0.026 / 29% flips and A's signal rate collapses from 23% to 4% of windows; the live OKX feed is at 0.025–0.033 / 35% | the lead-lag features need a perp feed that is not behind Binance: the fix is Binance's own perp stream (`fstream.binance.com` aggTrades, reachable from a non-US host) or a venue measured to be as fast; scaling OKX would not help |
| Kalshi `KXBTC15M` ("BTC price up in next 15 mins?", 60 s BRTI average before the close vs before the open), one day 2026-11-03 (`research/kalshi_fetch.py`, `research/kalshi_day.py`, `reports/kalshi_2026-11-03.csv`) | a second real venue, far bigger than Polymarket's 5-min market (median 38k trades and 3.4 M contracts per market), settlement agrees with the Binance TWAP label 99%; **no pre-open phase**: the market opens at the reference with the price at 0.50 and leaves it within 3 s, so nothing like A exists there; after the open the price tracks the Binance displacement about as well as a naive fair value at every elapsed time (Brier 0.242 vs 0.238 at 30 s, 0.213 vs 0.207 at 120 s, market better later), disagreements ≥ 0.10 are rare (4–18 of 96 markets) and a model-free B-type trade at 60 s makes +1.5 ¢ on 24 trades | no exploitable lag found model-free on one day; a real test needs a 15-minute model (ours are 5-minute) and more days; low priority next to Polymarket where the edge is measured |
| ten trade ideas + five hedges (`research/ideas10.py`, `research/hedges5.py`, `reports/ideas10_{main,july}.csv`, `reports/hedges5_main.csv`): late B (T+30/60/120), post-open overshoot fade, A confirmed at the open, both-sides pre-open maker, A consensus filters, high-confidence settlement capture, B + btc-predict agreement, B/C with settlement exits, dip add-on, B regime filters; hedges: after the TP print, pair completion (with/without TP), B pair completion, stop replacement | two keepers: **B with "sell 0.85 else settle, no stop"** (+5.5 / +5.2 / +6.7 ¢ vs +4.5 / +4.5 / +5.6 frozen; test dd $427 vs $519) and **high-confidence settlement capture** (model ≥ 0.85–0.90 at T+60…T+240 while that side trades ≤ 0.80–0.90, bid at the last print, hold to settlement: 80–91% win rate, +1–5 ¢ per share, $1–2.5k per period). Both-sides pairs are risk-free but the unavoidable single fills lose more than the pairs make; every hedge fills on the windows that were going to pay and sits on the losers (hedge after TP, pair completion at 92% "wins", stop replacement all negative); late B, fades, confirmations, consensus and regime filters do not beat the frozen rules on all periods | B exits candidate; capture strategy candidate (July pending) |
| **June 2026, a second never-seen month** (`research/oos_month.py`, `scripts/train_bp_june.py`, `reports/oos_june.json`, `reports/trades_{A,B,D}_june.csv`; frozen btcmovement2 April model with the point-price alignment, btc-predict fold trained on Feb–May only, same build as July) | the most volatile month in the data (median 5-min vol 8.6 bp vs 3.4–6.9 in Aug–Oct; June 4–5 at 18–22 bp gave the model no confident forecast at all): A +4.3 ¢ / 64 % win on 530 trades ($4.5k at $100 per trade, dd $820, t = 2.4) vs the previous exits +2.2 ¢; B +2.9 ¢ / 45 % win on 158 trades ($1.1k, dd $1.8k, t = 0.8; previous exits flat); D 89.5 % win, +2.6 ¢ on 181 trades ($0.5k); A+B+D $6.2k over 30 days ($206/day), dd $649, worst day −$454, 8 negative days | fifth period positive for A and D with the frozen rules; B's edge is real but thin in a high-volatility month (its stop-less exits are what kept it positive) |
| **end-to-end replay of the live code path** (`research/replay_live_day.py`, `paper/replay_live_2026-11-04{,_old}.csv`): 2026-11-04 streamed from the Binance archive into the trader's own LiveTimeline with real arrival delays (spot kline 0.8 s after the second ends, perp trades 0.5 s), the trader's finalize() on a fake clock, A at T−45 and B at T taken at the trader's instants, compared with the backtest's forecasts for the same windows (`scripts/score_m2.py`) | fixed rule: A mean \|live − backtest\| 0.0019 (max 0.022), side flips 2.8 % (all heads within 0.01 of 0.5), confident windows 7.3 % vs 8.4 %, trade/no-trade decision differs on 1.0 % of windows, B identical to 4 decimals, 0 synthetic seconds; the previous clock-based rule on the same stream: A mean error 0.027, 44 % side flips, confident 2.1 % vs 8.4 %, decision differs on 9.8 %, 524 synthetic seconds | the live pipeline now reproduces the backtest; the audit's parity section is the same test on real feeds |
| volatility gating (`research/vol_gate.py`, `reports/vol_gate.csv`): A only in calm windows, B / D only in volatile ones, by the realised vol of the previous 5 minutes (known at decision time), thresholds = tune-period quantiles, read on Aug 1–19 / tune / test / June / July | A's edge per trade is not lower in volatile windows (tune 4.1 ¢ volatile vs 2.2 ¢ calm, July 6.3 vs 1.3, test 3.8 vs 4.5); it only trades less often there, so every calm-only gate loses: −$23 to −$132 per day on tune/test/June/July. B's volatile tercile is its best in Aug / tune / June but its calm tercile is its best on test (7.2 ¢) and Aug (5.4 ¢): a volatile-only gate costs −$13 to −$443 per day on Aug/tune/test and gains +$27 to +$43 only on June/July, where B barely earned anyway. D has no pattern and no money in it | rejected: the model already self-gates (its confidence falls with volatility), and the trades it still takes in volatile windows are as good as the others |
| withdrawal policy (`research/withdrawals.py`, `reports/withdrawals.csv`): A+B+D at $10 per trade, account starting at $300 or $500, cash above a buffer swept out daily / weekly (Sundays) / never, on all five periods | the strategy's earnings are identical under every rule (fixed stakes: withdrawn + final − start = earned, e.g. test $2,368, tune $1,875, June $617, July $632); the rule only sets how much sits in the account; with a $300 start the balance never fell below $119 under any rule and never below the $30 three positions need, so trading never stops; weekly sweep above $400 keeps the lowest balance at $262–$300 across periods while paying out 85–95 % of the earnings | adopted as the operating rule: start $300, every Sunday withdraw everything above $400 |
| **last-minute capture, candidate strategy E** (`research/last_minute.py`, `reports/last_minute_{main,june,july}.csv`, `reports/trades_E_last_minute.csv`): decide at T+180/210/240/260 from the close head at that instant, bid at the last print on the model's side for 10 s (maker), hold to settlement; 80 cells of (q, maxpx) | T+260 is a different animal from T+90 (D): the whole (q, maxpx) surface is positive on tune and test; the cell q 0.80 / maxpx 0.90 gives +2.3 / +3.5 ¢ per share on tune / test (t 1.5 / 2.0, win 60 %, entry 0.57, 17 trades/day, $741 / $596 per day at $100, worst day −$559 / −$1,124, dd $1.6k / $2.5k) and nothing on the three point-price periods (Aug 1–19 +0.2 ¢, June −0.7 ¢, July −0.1 ¢). Mechanism: 40 s before the close the end TWAP is largely determined and the model's 10-second lead-lag head (its strongest skill, AUC 0.6) predicts the remainder; the market's last prints lag it. 86 % of the simulated fills are by prints within 0.10 of the bid (win 58–62 %), 14 % by sweeps (win 40–49 %, −14 to −24 ¢). Daily P&L uncorrelated with A+B+D (−0.03 / −0.26); A+B+D+E $1,188 / $1,293 per day on tune / test with dd $1.1k / $1.3k | **dropped at the user's request** (2026-11-06): `CFG_E` kept in the code but not in `ENABLED`; it rested on two TWAP-era periods and on fill realism in the last 40 s |
| stop-loss and stricter entries for D (`research/d_stop.py`, `reports/D_stop_main.csv`): sell at the market when our side prints ≤ 0.75 / 0.70 / 0.60 / 0.50 (else settle), q 0.90–0.95, maxpx 0.85–0.90, on Aug 1–19 / tune / test | a stop at 0.75 fires on 41–58 % of D's trades and most of those still resolve our way: the win rate falls from 83–89 % to 43–60 % and the edge to −1.5 / +0.4 / +0.6 ¢ (Aug / tune / test) from +1.6 / +0.9 / +2.3; 0.70 and 0.60 are the same story; only a 0.50 stop is roughly neutral (+1.1 / +0.6 / +2.8 ¢) and it still lowers the win rate. q 0.92 gives +2.7 / +0.5 / +5.1 ¢ on half the trades (no gain in $/day); q 0.95 leaves 24–58 trades per period | rejected: D's losers are the price dipping and coming back; a stop turns them into realised losses and cannot raise the win rate, it lowers it |
| ten ways to raise D's win rate (`research/d_winrate.py`, `reports/D_winrate.csv`): later decision (T+150 / T+180), q 0.95, market agreement (side in [0.80, 0.90]), short heads agreeing, price ≥ 10 bp past the reference, calm windows, low touch-back probability, 10-s head ≥ 0.90, two combos | three filters change nothing (once the close head is ≥ 0.90 the short heads and touch probabilities already pass); later decisions help Aug and hurt test; q 0.95 and the move filter go negative on a period; calm-only helps tune, hurts Aug. The one that raises the win rate in all five periods is requiring the market to agree: side trading in [0.80, 0.90] → win 88 / 88 / 90 / 91 / 91 % (Aug / tune / test / June / July) vs 83 / 87 / 89 / 90 / 90 %, edge +1.7 / +0.7 / +2.0 / +3.7 / +3.4 ¢, 5–25 % fewer trades | **adopted**: D now requires the side to trade between 0.80 and 0.90 (`minpx` 0.80) |
| **November 4–7 continuation** (`reports/oos_november.json`, `reports/trades_{A,B,D}_november.csv`; archive extended through Nov 7, `scripts/score_bp_days.py` scores btc-predict with the production models through the live path) | frozen rules, $100 per trade: A +$150 over Nov 4–7 (Nov 5 +$480, Nov 6 −$370, Nov 7 +$70, 61 % win on 77 trades after the box went down on Nov 6 19:40), B −$69 (Nov 7 +$113), D +$27; book +$10 over the four days (Nov 7 +$84). Trade-by-trade against the box's paper trades from Nov 5 13:20: every window both took agrees in side and result (A 7/7, B 8/8, D 5/5), but on Nov 6 the box took 5 A windows the backtest did not and vice versa; Nov 5's parity had 1 decision in 127 differing | bad days in the backtest too; the Nov 6 A decision divergence needs the box's own Nov 6 parity audit |
| daily take-profit / stop-loss for A (`research/daily_tp_A.py`, `reports/A_daily_tp.csv`): stop opening A trades for the UTC day once the realised day P&L reaches +X or −Y, on Aug 1–19 / tune / test / June / July with the current exits, $100 per trade | a daily take-profit loses money in every period at every level (+$300: −$85 to −$200 per day vs no limit, +$500: −$60 to −$115, +$750: −$30 to −$80; Sharpe not better either): A's winning days keep winning, the rest of the day after +X is positive on average, so stopping forfeits it. A daily stop also costs, less: −$300 is −$5 to −$53 per day on four periods (+$19 on test), −$500 is −$3 to −$47 and fires on 3–33% of days; −$150/−$200 costs $60–$160 per day | no take-profit; the daily stop is kept only as a safety net against a malfunction, set wide (−$600 per $100 stake, −$60 at $10) so it rarely fires |
| A exits that hold to settlement (`research/settle_exit_A.py`, `reports/A_settle_exit_{main,july}.csv`; settlement pays 1/0 with no fee) | holding every trade to settlement beats the frozen exits in all four periods (+1.7 / +0.2 / +2.2 / +0.0 ¢; p 0.03 / 0.40 / 0.002 / 0.49) with a 52–56% win rate and ~2× the drawdown; **TP 0.80 then settle (no stop, no time-stop)** is best on tune (+3.3 vs +2.5 ¢, p 0.03), equal on test, +0.6 ¢ in July (p 0.12), −0.2 ¢ in Aug 1–19 (p 0.65), with the highest win rate (65–67%) and a lower tune drawdown ($820 vs $1,273); the 0.05 stop hurts once settlement is allowed (a ≤ 0.05 print still resolves our way often enough) | candidate change to A's exits, user's call (their brief excluded holding to settlement) |
| pre-open mispricing grid (`research/preopen_fade.py`, `reports/A_preopen_fade_grid.csv`; `research/preopen_follow_check.py`): decide at T−25 … T−5 on model-vs-last-print disagreement, fade vs follow regimes, bid/taker entries, exits | fade weak and rare (best cell +2 ¢ / 53% on test, ≈ 0 in Aug); follow at T−5 with δ 0.08 positive in Aug 1–19 / tune / test (+4.3 / +3.3 / +2.6 ¢) and additive to B | **adopted as strategy C** (see §3) |
| tight sell limit for A (`research/tight_tp_A.py`, `reports/A_tight_tp.csv`): 0.50 fill at T−45, sell limit 0.53 / 0.54 / 0.55 placed immediately, stop 0.05, T+240; also entered at T−30 | the limit fills in 89–94% of trades but the 6–11% that never reach it lose ~45¢ each: +0.0 to +0.5¢ per share (tune −$950 … +$570, test +$70 … +$1,220 at $100/trade) vs +2.2 / +3.4¢ ($5.7k / $8.9k) with the 0.80 limit; a 0.50 bid resting from T−30 fills on the wrong windows and loses $3–5k | rejected |
| later entry for A (`research/entry_timing_A.py`, `reports/A_entry_timing.csv`): decide at T−35 / −25 / −15 / −10 with the forecast available then, buy as a taker, with a bid at the last print (10 s) or with a bid at 0.50 | the later forecasts win more often (up to 71% at T−10 vs 62–64% at T−45) but the market reprices faster than the model sharpens: average entries 0.52–0.60, and no variant beats T−45 / 0.50 on test (frozen $8.9k; best later variant $2.8k, most negative); tune and test disagree on every later variant | rejected: T−45 at 0.50 is the only entry where the market has not yet priced what the model knows |
| late-flip exit for A (`research/early_flip_exit.py`, `reports/A_early_flip_exit_{main,july}.csv`): after the 0.50 fill, re-evaluate the close head at T−40 … T+30 and sell at the market when P(our side) < q | the later forecasts do carry information (a flip at T0 cuts the hit rate of the original side from 59% to 45%, at T+30 from 65% to 38%) but the market has already repriced by then: every rule (q 0.35–0.50, 7 checkpoint sets) is ≤ the frozen exits on tune (bootstrap p 0.2–0.98), mixed on test, and all are worse in July | rejected |
| btcmovement2 refit on 2025 → Jul 2026 (`scripts/refit_m2.sh`, `scripts/train_m2_wf.py`, `scripts/score_m2_new.py`; windows thinned 1/4 (2025) and 1/3 (2026) to fit 15 GB) | worse than the frozen April model at T−45: AUC 0.511 vs 0.518 (tune), 0.523 vs 0.529 (test); test P&L +2.25¢ vs +2.9¢ at the live threshold | rejected; frozen `wf5_20260430` kept |
