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
| D: high-confidence settlement capture (T+90, model ≥ 0.90 for a side trading ≤ 0.90, bid at the last print, hold to settlement) | 10 | +2.3 ¢ | +4.6 % | **89 %** | 4.5 | 1.4 | $645 |
| B2: same, δ = 0.08 (more trades) | 55 | +2.7 ¢ | +5.4 % | 64 % | 11.0 | 3.5 | $1,050 |
| A: pre-open directional (T−45, btc-predict agreement, sell limit 0.85, else hold to settlement) | 40 | +4.7 ¢ | +9.3 % | 64 % | 13.6 | 4.2 | $1,120 |
| A0: the same with the previous exits (TP 0.80 / SL 0.05 / T+240), kept as a shadow | 40 | +3.4 ¢ | +6.7 % | 64 % | 12.3 | 3.4 | $606 |
| C: "B five seconds early" (T−5, δ = 0.08, bid at the last print until the open, TP 0.80 / SL 0.05 / T+240) | 40 | +2.6 ¢ | +5.2 % | 64 % | 7.2 | 2.8 | $1,950 |
| A + B combined ($100 per trade) | 62 | — | **$17.7 k over 34 days** | — | 18.6 | — | $596 |
| A + C + B (C first, B on the windows C did not fill) | 93 | — | **$27.4 k over 34 days** | — | 20.9 | — | $846 |
| A + C + B + D | 103 | — | **$28.3 k over 34 days** | — | 21.0 | — | $815 |

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

### C — "B five seconds early" (added 2026-11-04, `research/preopen_fade.py`, `research/preopen_follow_check.py`)

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

## 8. Paper trading (`polybtc/paper_trader.py`)

Feeds: Binance spot and ETH 1 s klines (websocket mirror + REST backfill), OKX BTC-USDT perpetual trades
(REST polling; the Binance perpetual is geo-blocked here, OKX stands in for the lead-lag feature), the
Polymarket CLOB market websocket (prints and books). Model history from the local archive; the btc-predict
live feature path is parity-tested against training (`python polybtc/bp_live.py`). Logs: `paper/forecasts.jsonl`
(every window), `paper/paper_trades.jsonl`, `paper/paper.log`; `python polybtc/paper_trader.py --summary`
prints P&L with resolutions. Strategy A is simulated with the user's execution assumption (filled at 0.50
before T−40); strategy B's buy limit counts as filled when a print at or below the bid occurs within 10 s (queue position ignored). Both are reported with equal weight.

Running it on your own box: `scripts/setup_box.sh` (clones the four repos under `/home/user`, installs, pulls
10 days of Binance archive history, starts the trader in an auto-restart loop). Check first that
`data-api.binance.vision` and `www.okx.com` answer from the box's IP. Daily, re-run `scripts/fetch_binance.py`
for the previous day and rebuild the timeline so the regime features keep a full history.

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
| ten trade ideas + five hedges (`research/ideas10.py`, `research/hedges5.py`, `reports/ideas10_{main,july}.csv`, `reports/hedges5_main.csv`): late B (T+30/60/120), post-open overshoot fade, A confirmed at the open, both-sides pre-open maker, A consensus filters, high-confidence settlement capture, B + btc-predict agreement, B/C with settlement exits, dip add-on, B regime filters; hedges: after the TP print, pair completion (with/without TP), B pair completion, stop replacement | two keepers: **B with "sell 0.85 else settle, no stop"** (+5.5 / +5.2 / +6.7 ¢ vs +4.5 / +4.5 / +5.6 frozen; test dd $427 vs $519) and **high-confidence settlement capture** (model ≥ 0.85–0.90 at T+60…T+240 while that side trades ≤ 0.80–0.90, bid at the last print, hold to settlement: 80–91% win rate, +1–5 ¢ per share, $1–2.5k per period). Both-sides pairs are risk-free but the unavoidable single fills lose more than the pairs make; every hedge fills on the windows that were going to pay and sits on the losers (hedge after TP, pair completion at 92% "wins", stop replacement all negative); late B, fades, confirmations, consensus and regime filters do not beat the frozen rules on all periods | B exits candidate; capture strategy candidate (July pending) |
| A exits that hold to settlement (`research/settle_exit_A.py`, `reports/A_settle_exit_{main,july}.csv`; settlement pays 1/0 with no fee) | holding every trade to settlement beats the frozen exits in all four periods (+1.7 / +0.2 / +2.2 / +0.0 ¢; p 0.03 / 0.40 / 0.002 / 0.49) with a 52–56% win rate and ~2× the drawdown; **TP 0.80 then settle (no stop, no time-stop)** is best on tune (+3.3 vs +2.5 ¢, p 0.03), equal on test, +0.6 ¢ in July (p 0.12), −0.2 ¢ in Aug 1–19 (p 0.65), with the highest win rate (65–67%) and a lower tune drawdown ($820 vs $1,273); the 0.05 stop hurts once settlement is allowed (a ≤ 0.05 print still resolves our way often enough) | candidate change to A's exits, user's call (their brief excluded holding to settlement) |
| pre-open mispricing grid (`research/preopen_fade.py`, `reports/A_preopen_fade_grid.csv`; `research/preopen_follow_check.py`): decide at T−25 … T−5 on model-vs-last-print disagreement, fade vs follow regimes, bid/taker entries, exits | fade weak and rare (best cell +2 ¢ / 53% on test, ≈ 0 in Aug); follow at T−5 with δ 0.08 positive in Aug 1–19 / tune / test (+4.3 / +3.3 / +2.6 ¢) and additive to B | **adopted as strategy C** (see §3) |
| tight sell limit for A (`research/tight_tp_A.py`, `reports/A_tight_tp.csv`): 0.50 fill at T−45, sell limit 0.53 / 0.54 / 0.55 placed immediately, stop 0.05, T+240; also entered at T−30 | the limit fills in 89–94% of trades but the 6–11% that never reach it lose ~45¢ each: +0.0 to +0.5¢ per share (tune −$950 … +$570, test +$70 … +$1,220 at $100/trade) vs +2.2 / +3.4¢ ($5.7k / $8.9k) with the 0.80 limit; a 0.50 bid resting from T−30 fills on the wrong windows and loses $3–5k | rejected |
| later entry for A (`research/entry_timing_A.py`, `reports/A_entry_timing.csv`): decide at T−35 / −25 / −15 / −10 with the forecast available then, buy as a taker, with a bid at the last print (10 s) or with a bid at 0.50 | the later forecasts win more often (up to 71% at T−10 vs 62–64% at T−45) but the market reprices faster than the model sharpens: average entries 0.52–0.60, and no variant beats T−45 / 0.50 on test (frozen $8.9k; best later variant $2.8k, most negative); tune and test disagree on every later variant | rejected: T−45 at 0.50 is the only entry where the market has not yet priced what the model knows |
| late-flip exit for A (`research/early_flip_exit.py`, `reports/A_early_flip_exit_{main,july}.csv`): after the 0.50 fill, re-evaluate the close head at T−40 … T+30 and sell at the market when P(our side) < q | the later forecasts do carry information (a flip at T0 cuts the hit rate of the original side from 59% to 45%, at T+30 from 65% to 38%) but the market has already repriced by then: every rule (q 0.35–0.50, 7 checkpoint sets) is ≤ the frozen exits on tune (bootstrap p 0.2–0.98), mixed on test, and all are worse in July | rejected |
| btcmovement2 refit on 2025 → Jul 2026 (`scripts/refit_m2.sh`, `scripts/train_m2_wf.py`, `scripts/score_m2_new.py`; windows thinned 1/4 (2025) and 1/3 (2026) to fit 15 GB) | worse than the frozen April model at T−45: AUC 0.511 vs 0.518 (tune), 0.523 vs 0.529 (test); test P&L +2.25¢ vs +2.9¢ at the live threshold | rejected; frozen `wf5_20260430` kept |
