# intraday-bot

A **new, standalone** US-equities bot — S&P 500 on 15-minute bars, two sleeves:

| Sleeve | Entry (15m, close-of-bar) | Exit |
|---|---|---|
| **momentum** | SMA20 crosses above SMA50, price > SMA50, RSI14 < 75, SPY regime risk-on | 1% trailing stop (broker-side), or SMA20/50 cross-down |
| **meanrev** | RSI(2) < 10, stock above its 200-day SMA (prior close), SPY regime risk-on | RSI(2) > 60, 26-bar time stop, or 2% hard stop |

Regime = SPY above its 200-day SMA **as of the previous close** (no look-ahead).
Portfolio: max 8 positions, 12.5% of equity each, long only, no margin.

This project is fully separate from `chawani_bot` (which stays as-is) and from
anything Cassiora. It reuses one *idea-level* lesson set from chawani:
fail-closed data handling, pre-registered kill criteria, split validation.

**FINRA retired the Pattern Day Trader rule on 2026-06-04**, replacing it with
an intraday margin framework (real-time margin checks instead of a 3-trades/
5-days counter). Alpaca's API now always returns `daytrade_count: null` — the
backtests below still model the OLD 3-per-5-days cap (that's what applied
when the historical trades happened), but `bot.py` no longer checks or
enforces it live; margin sufficiency is the broker's real-time problem now,
not a gate this code checks.

## Files

- `universe.json` — S&P 500 list (copied from chawani's 2026-06-20 snapshot)
- `data.py` — downloads/caches 60d of 15m bars + 2y daily (Yahoo free limits)
- `strategies.py` — indicators + entry/exit signal generation (pure, no I/O)
- `backtest.py` — portfolio simulator: next-open fills, intrabar stops,
  friction per side, PDT day-trade accounting, split-half stability
- `bot.py` — live/paper Alpaca engine running the VALIDATED profile only:
  daily-bar momentum, 8% trailing stop, SMA cross-down exits, once-per-day
  cycle inside a 10:00-15:00 ET window (skips opening-auction and closing
  churn; slight drift from the backtest's at-the-open fills, accepted for
  better spreads). PAPER + DRY_RUN by default; `--once` for a single cycle;
  mean reversion is research-only and not in the bot.
- `backtest_daily.py` / `download_daily10y.py` — 10-year daily-bar validation
- `dashboard.py` — local web dashboard (see below)

## How to run

```bash
pip install -r requirements.txt
python data.py            # ~5-10 min first time; resumable; cached to data/
python backtest.py        # 5 bps/side friction
python backtest.py --cost-bps 10   # pessimistic friction stress
```

Outputs land in `results/`: summary JSON, per-run trade CSVs, equity-curve PNG.

## Kill criteria (pre-registered — decided BEFORE seeing results)

Carried over from `chawani_bot/DAY_TRADING_BACKTEST_PLAN.md`, which set the bar
before this project existed. The bot does **not** go live unless, after
frictions:

1. Profit factor >= 1.3
2. Beats SPY buy-and-hold over the same window
3. Max drawdown no worse than SPY's over the same window
4. Both split halves are profitable (no single-regime fluke)
5. Result survives the 10 bps/side friction stress

## Results — Alpaca out-of-sample validation (2026-07-11)

**441 sessions, 2024-07-11..2026-04-14 — none of it seen when parameters were
chosen.** 499 names, 5 bps/side unless noted. Full detail in `results/`.

| run | return | PF | maxDD | trades | worst-5d day trades |
|---|---|---|---|---|---|
| momentum, 2% trail | +4.8% | 1.02 | -14.0% | 2,253 | 18 |
| momentum, 3% trail | **+25.3%** | 1.11 | **-13.9%** | 1,768 | 12 |
| momentum, 3% trail @ 10 bps | +0.4% | 1.00 | -16.6% | 1,770 | 12 |
| momentum, 4% trail | +21.7% | 1.11 | -15.6% | 1,635 | 8 |
| meanrev | **-90.0%** | 0.71 | -90.1% | 18,651 | 245 |
| combined (3% trail) | -2.5% | 0.99 | -21.3% | 2,407 | 86 |
| **SPY buy & hold** | **+26.2%** | | -20.8% | 0 | 0 |

**VERDICT: momentum survives but FAILS the kill criteria; meanrev is
FALSIFIED; do not deploy either.**

- Momentum 3-4% trail is a real plateau, not a lucky knife-edge: both halves
  profitable (+10.3%/+13.6%), Sharpe 0.91 vs SPY 0.90, a third less drawdown
  than SPY. But: PF 1.11 < 1.3 (criterion 1 FAIL), 25.3% < SPY 26.2%
  (criterion 2 FAIL), and at 10 bps/side the entire edge vanishes to +0.4%
  (criterion 5 FAIL). Its profit is a bet that fills cost <= 5 bps — no
  margin of safety. Net: a lower-drawdown SPY clone you could replace with
  simply buying SPY.
- Mean reversion: -90% over 21 months across 18,651 trades. The +16.8%
  zero-cost result on the 60-day Yahoo window was pure noise — exactly the
  trap split-validation exists to catch. Sleeve disabled in bot.py
  (BOT_ENABLE_MEANREV=false).
- PDT: even momentum-only breaches the 3-per-5-days cap (worst window: 12);
  the live bot's PDT guard would distort behavior vs the backtest.

## Results — 10-year DAILY-bar backtest (2026-07-11)

2,514 sessions, 2016-07-11..2026-07-10, 499 names, 5 bps/side unless noted.
`python backtest_daily.py --trail-pct 8`. **Survivorship bias warning: today's
S&P list applied backwards — strategy numbers are optimistic.**

| run | 10y return | CAGR | PF | maxDD | Sharpe | avg hold |
|---|---|---|---|---|---|---|
| momentum, 3% trail | +42% | 3.6% | 1.08 | -20% | 0.39 | 7d |
| momentum, 5% trail | +49% | 4.1% | 1.15 | -21% | 0.41 | 18d |
| momentum, 8% trail | +76% | 5.8% | **1.31** | -22% | 0.52 | 40d |
| momentum, 8% @ 10 bps | +65% | 5.1% | 1.27 | -22% | 0.46 | 40d |
| meanrev (5% stop, 10d) | +96% | 7.0% | 1.09 | -31% | 0.56 | 5d |
| combined (8% trail) | +83% | 6.3% | 1.24 | -23% | 0.54 | 29d |
| **SPY buy & hold** | **+315%** | **15.3%** | | -34% | 0.89 | |

**VERDICT: the strategies are real but dominated by buy-and-hold.** Momentum
8% finally clears PF >= 1.3, survives 10 bps, and cuts drawdown by a third —
but it earns ~1/3 of SPY's CAGR because it sits in cash between signals and
trails stops out of winners in a decade that mostly went up. Consistent
finding across all three studies (15m/60d, 15m/21mo, daily/10y): long-only
signal timing on S&P names underperforms full-time market exposure.

## Results — rotation strategies, 10y daily (2026-07-12, pre-registered)

Window 2017-08-01..2026-07-10 (common signal warmup), criteria written in
`backtest_rotation.py` BEFORE the first run. Cash modelled at 0% (conservative).

| strategy | return | CAGR | maxDD | Sharpe | verdict |
|---|---|---|---|---|---|
| dual momentum (SPY/cash, 12m) | +172% | 11.9% | -33.7% | 0.75 | **FAIL** (D1, D2) |
| xs momentum (top-20, 12-1) | +2,322% | 43.0% | -43.9% | 1.25 | **FAIL** (X2 drawdown) |
| xs momentum + SPY regime | +690% | 26.1% | -32.8% | 1.00 | **PASS (provisional)** |
| SPY buy & hold | +251% | 15.1% | -33.7% | 0.85 | benchmark |

All three survive the 10 bps stress unchanged (monthly turnover is cheap).

**Bias control:** the equal-weight average of the same (survivorship-biased)
universe with NO ranking made +315% — so the ranking adds ~+375pp beyond the
biased list, with lower maxDD (-32.8% vs -38.3%). The signal is not pure
bias, but bias still inflates it: 26% CAGR is NOT believable as-is. Before
any deployment, xs_momentum_regime must be reproduced on point-in-time
S&P constituents (historical membership lists), not today's list backwards.
Dual momentum's honest FAIL: the 2020-style fast crash is too quick for a
12-month monthly filter; identical maxDD to SPY with 3pp less CAGR.

### Point-in-time reproduction (2026-07-12) — VERDICT: DROPPED

Reran with fja05680/sp500 historical membership (`--pit`): each month ranks
ONLY actual index members on that date. 110 former members' price histories
recovered from Yahoo; 108 truly delisted names unavailable (mean monthly
membership coverage 91.4%, min 83.1% — residual cuts both ways: missing
acquired winners AND bankrupt losers).

| strategy | biased | point-in-time | verdict |
|---|---|---|---|
| xs momentum | +2,322% (43.0% CAGR) | +397% (19.7%) | FAIL (maxDD -38.8% > SPY) |
| xs momentum + regime | +690% (26.1% CAGR) | **+88% (7.3%)** | **FAIL X1 — DROPPED** |
| SPY | +251% (15.1%) | +251% (15.1%) | benchmark |

Per the rule pre-registered in `backtest_rotation.py` before the run:
xs_momentum_regime failed X1 point-in-time => the edge is attributed to
survivorship bias and the strategy is dropped, no parameter changes allowed.
~87% of the apparent regime-variant edge was the bias: the ranking loved
stocks that later JOINED the index (it "bought" them pre-inclusion, which no
live trader restricted to index members could have done). Notably the plain
(no-regime) PIT variant still beat SPY on return (+397%) with real signal —
but through deeper drawdowns, failing the pre-registered risk bar. The
regime filter, which looked brilliant on biased data, mostly just missed
the 2020/2023 rebounds on real data (2022 -17.6%, 2023 -3.4%).

## Operational layer (2026-07-12)

bot.py is self-healing: reconciliation every 5-min wake-up (protection
coverage audit, orphan-order cleanup, state sync), marketable-limit entries
(last+0.5% cap, whole shares), idempotent client_order_ids (crash-safe),
15% drawdown-from-HWM circuit breaker, alerts (alerts.py -> log +
ALERTS.json banner on the dashboard + optional BOT_ALERT_WEBHOOK), and a
heartbeat the dashboard turns red when the bot goes silent >15 min.
Future risk rules (sector cap, earnings blackout) deliberately NOT added —
untested rules don't ship; backtest them first.

## Dashboard

`python dashboard.py` then open http://127.0.0.1:8050 — live paper-account
equity, positions, fills, margin cushion, kill-switch status, the research
scoreboard (all summary_*.json runs), and a **trade journal**: every
completed round trip with buy/sell timestamps and prices, qty, P&L in $ and
%, holding period, strategy, and how the exit happened (trailing stop vs
bot cross-down) — reconstructed from Alpaca's fill records, so it also
captures stops that fire broker-side while the bot is offline.
Auto-refreshes every 60s. Paper only. (The old "day trades used" counter
was removed — Alpaca no longer reports it, see the PDT note above.)

## Paper-test plan (2 months)

1. **Reset or separate the paper account first.** The current paper account
   already carries chawani's positions — results will be uninterpretable if
   both bots share it. Alpaca dashboard -> reset paper account, or create a
   second paper key set for this project's .env.
2. Start the bot: set BOT_DRY_RUN=false (keep BOT_PAPER=true!) and run
   `python bot.py`. It trades once per day inside the 10:00-15:00 ET window;
   leave it running, or schedule `python bot.py --once` daily at 10:05 ET.
3. Success bar, pre-registered: bot behaves as backtested (fills near next-bar
   opens, stops fire correctly, no unprotected positions, no margin calls);
   performance within the backtest's plausible range vs SPY over the same
   window. A 2-month P&L is NOISE — the test validates operations, not alpha.

## Earlier: Yahoo 60-day run (60 sessions, 2026-04-15..2026-07-10, 494 names)

Kept for the record — superseded by the Alpaca validation above.

| run | 0 bps | 5 bps/side | 10 bps/side | trades | worst-5d day trades |
|---|---|---|---|---|---|
| momentum (1% trail) | +2.4% | **-11.4%** | -21.7% | 1,120 | 85 |
| meanrev | +16.8% | **-18.4%** | -43.5% | 2,855 | 236 |
| combined | +4.1% | **-16.2%** | -30.0% | 1,455 | 124 |
| momentum, 2% trail | | +0.9% | | 437 | 24 |
| momentum, 3% trail | | +5.9% | | 292 | 10 |
| **SPY buy & hold** | +8.7% | +8.7% | +8.7% | 0 | 0 |

Against the pre-registered kill criteria (best variant, 3% trail):
PF 1.16 < 1.3 FAIL - underperforms SPY FAIL - maxDD tie - halves positive
but PF < 1.3 both halves - 10 bps stress untested (moot). Meanrev's zero-cost
"edge" was one good month: H1 +22.0% (PF 1.43), H2 -3.6% (PF 0.93).

Why it fails, in one line each:
1. **Friction eats the edge**: avg trade wins ~0.1-0.2% gross; a 10 bps round
   trip is the whole edge. Meanrev swings -35pts going 0->5 bps.
2. **PDT makes it illegal anyway**: 50-2,400 same-day round trips vs the 3-per-
   5-days cap on a $12.5k account.
3. **The stop-width gradient points AWAY from intraday**: every widening of the
   stop (fewer, longer trades) improved results; the best variant holds ~8.4
   days — i.e. it converges back to swing trading on daily bars, which
   chawani_bot already does.

## Known limitations (read before trusting any numbers)

- **60 days of 15m data is a smoke test, not validation.** That's Yahoo's free
  intraday limit — one market regime, one season. Before real money: pull
  1-2+ years of minute bars from Alpaca's free IEX feed (creds already exist)
  and re-run, per the Phase 0 plan.
- **Survivorship bias:** today's S&P 500 list applied backwards. Negligible
  over 60 days, real over years.
- **PDT rule (historical):** applied when this backtest ran (account was
  $12,500 < $25k, so max 3 same-day round trips per 5 days). `backtest.py`
  still reports day-trade counts for that reason. FINRA retired the rule
  2026-06-04; `bot.py` no longer enforces it (see the note near the top).
- IEX-sourced fills and 5 bps friction are estimates; live spreads vary.
