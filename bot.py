"""
Daily-momentum bot — live/paper trading engine (Alpaca).

Runs the ONE strategy that survived all three backtests (60d/15m, 21mo/15m
out-of-sample, 10y/daily): daily-bar momentum with an 8% trailing stop.
Mean reversion was falsified out-of-sample and is not in this bot at all
(it remains in backtest.py/backtest_daily.py for research only).

The live loop mirrors backtest_daily.py's semantics exactly:
  * Signals computed once per day on COMPLETED daily bars (today's partial
    bar is excluded — no look-ahead).
  * Entries/exits execute once per day inside a 10:00-15:00 ET window —
    open+30min skips the opening-auction churn, the 15:00 cutoff skips the
    closing rush. (Slight drift from the backtest's at-the-open fills;
    accepted deliberately for better spreads.)
  * Entry: SMA20 crosses above SMA50, close > SMA50, RSI14 < 75, SPY above
    its 200-day SMA. Candidates ranked by close/SMA50 - 1; strongest first.
  * Exit: broker-side 8% GTC trailing stop (survives bot crashes), plus
    SMA20/50 cross-down exit executed by the bot.

Safety rails:
  * PAPER unless BOT_PAPER=false; DRY_RUN unless BOT_DRY_RUN=false.
  * Fails CLOSED on stale/missing data; KILL_SWITCH file halts entries.
  * Daily loss halt (-2% on the day) and a 15% drawdown-from-high-water
    circuit breaker that halts entries for human review.
  * Max 8 positions at 12.5% of equity each.
  * Cancels only the target symbol's orders before closing — never
    account-wide (the paper account may be shared).

Professional/operational layer:
  * RECONCILIATION every 5-minute wake-up (not just in the trade window):
    every position must carry protective sell-order coverage >= its size —
    missing/short protection is re-attached and alerted; orphan sell orders
    (position already gone) are canceled; state.json is synced to broker
    truth. A crashed stop-attach self-heals within one wake-up.
  * MARKETABLE-LIMIT entries (last trade + 0.5% cap), never naked market
    orders. Whole shares; a stock pricier than one position budget is
    skipped. If the limit doesn't fill immediately, it works as a DAY order
    and reconciliation attaches the stop whenever it fills.
  * IDEMPOTENT orders: client_order_id = dm-e-<date>-<symbol>, so a
    crash-and-restart cannot double-submit the same day's entry.
  * ALERTS (alerts.py) to log + ALERTS.json (dashboard banner) + optional
    BOT_ALERT_WEBHOOK on: unprotected position, stop-attach failure, close
    failure, drawdown halt, cycle crash.
  * HEARTBEAT: heartbeat.json every wake-up; the dashboard shows it and
    flags a silent bot — a dead bot must never look like a quiet one.

Run:
    python bot.py            # continuous: one cycle per trading day
    python bot.py --once     # single cycle (for Task Scheduler / testing)

DO NOT go live: the strategy trails SPY buy-and-hold by ~9%/yr over 10
years. The 2-month paper test validates OPERATIONS, not alpha (README).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from alpaca.data.requests import StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    TrailingStopOrderRequest,
)

from alerts import alert
from alpaca_data import _fetch_slice, client as data_client, load_env
from strategies import StrategyParams, wilder_rsi

log = logging.getLogger("daily_momentum_bot")
ROOT = Path(__file__).parent
KILL_SWITCH = ROOT / "KILL_SWITCH"
STATE_FILE = ROOT / "state.json"
P = StrategyParams()

NY = "America/New_York"
LOOKBACK_DAYS = 420            # covers 200-day SMA + warmup
BATCH = 100                    # symbols per daily-bars request


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Config:
    api_key: str
    secret_key: str
    paper: bool = True
    dry_run: bool = True
    universe: tuple[str, ...] = ()
    regime_symbol: str = "SPY"
    max_positions: int = 8
    pos_frac: float = 0.125
    trail_pct: float = 8.0         # validated width: 10y PF 1.31, holds ~40d
    max_daily_loss_pct: float = 0.02
    # Trade window: 10:00-15:00 ET. The first 30 min after the open and the
    # last hour before the close have the widest spreads and wildest prints;
    # orders only go out inside the window. Broker-side trailing stops remain
    # active 24/5 regardless — protection is never window-limited.
    open_delay_min: int = 30       # window opens this long after 09:30 ET
    window_end_hour: int = 15      # no new cycle at/after 15:00 ET
    # Circuit breaker: equity 15% below its high-water mark = stop the
    # experiment and get a human. Roughly the strategy's historical worst
    # (10y backtest maxDD -22%), so triggering means "worse than tested".
    dd_halt_pct: float = 0.15
    limit_slip_pct: float = 0.005  # marketable-limit cap: last trade +0.5%
    # Hard dollar cap on TOTAL invested capital (Atul, 13 Jul 2026):
    # $10,000 working capital divided across the position slots
    # ($1,250 each at 8 slots); never more than $10,000 in the market at
    # once, the rest stays cash. Overridable via BOT_MAX_EXPOSURE_USD.
    max_total_exposure_usd: float = 10_000.0

    @classmethod
    def from_env(cls) -> "Config":
        load_env()                 # picks up .env next to this script
        key = os.environ.get("ALPACA_API_KEY", "")
        sec = os.environ.get("ALPACA_SECRET_KEY") or \
            os.environ.get("ALPACA_API_SECRET", "")
        if not key or not sec:
            raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY must be set.")
        uni_file = os.environ.get("BOT_UNIVERSE_FILE", str(ROOT / "universe.json"))
        universe = tuple(t.replace("-", ".") for t in
                         json.loads(Path(uni_file).read_text())["sp500"])
        if _env_bool("BOT_ENABLE_MEANREV", False):
            log.warning("BOT_ENABLE_MEANREV is set but mean reversion was "
                        "falsified out-of-sample and is not part of this bot.")
        return cls(api_key=key, secret_key=sec,
                   paper=_env_bool("BOT_PAPER", True),
                   dry_run=_env_bool("BOT_DRY_RUN", True),
                   universe=universe,
                   max_total_exposure_usd=float(
                       os.environ.get("BOT_MAX_EXPOSURE_USD", "10000")))


class DataError(Exception):
    pass


class State:
    """Per-symbol entry metadata + last processed session, across restarts."""

    def __init__(self):
        self.d: dict = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

    def save(self):
        STATE_FILE.write_text(json.dumps(self.d, indent=1, default=str))

    def opened(self, symbol: str):
        self.d.setdefault("positions", {})[symbol] = {
            "entered": datetime.now(timezone.utc).isoformat()}
        self.save()

    def closed(self, symbol: str):
        self.d.get("positions", {}).pop(symbol, None)
        self.save()

    @property
    def last_run(self) -> str:
        return self.d.get("last_run_date", "")

    @last_run.setter
    def last_run(self, v: str):
        self.d["last_run_date"] = v
        self.save()


class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.trading = TradingClient(cfg.api_key, cfg.secret_key, paper=cfg.paper)
        self.data = data_client()
        self.state = State()
        self._running = True

    def stop(self, *_):
        self._running = False
        log.info("Shutdown requested")

    # ---------------- data ---------------- #

    def daily_bars(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Batched, completed daily bars for many symbols. Fails closed:
        symbols with missing/short/stale data are simply absent."""
        start = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
        today_ny = pd.Timestamp.now(tz=NY).date()
        out: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), BATCH):
            chunk = symbols[i:i + BATCH]
            raw = _fetch_slice(self.data, chunk, TimeFrame.Day, start,
                               datetime.now(timezone.utc) - timedelta(minutes=16))
            if raw is None or raw.empty:
                continue
            for s in chunk:
                try:
                    df = raw.xs(s, level="symbol").sort_index()
                except KeyError:
                    continue
                # completed bars only — drop today's in-progress daily bar
                idx_ny = df.index.tz_convert(NY)
                df = df[[d.date() < today_ny for d in idx_ny]]
                if len(df) < P.daily_trend_sma + 5:
                    continue
                if df["close"].isna().any() or (df["close"] <= 0).any():
                    continue
                last_ny = df.index[-1].tz_convert(NY).date()
                if (today_ny - last_ny).days > 5:
                    continue                    # stale — refuse
                out[s] = df
        return out

    # ---------------- signals (mirror backtest_daily) ---------------- #

    @staticmethod
    def _smas(df: pd.DataFrame):
        c = df["close"]
        return c, c.rolling(P.fast_sma).mean(), c.rolling(P.slow_sma).mean()

    def momentum_entry(self, df: pd.DataFrame) -> float | None:
        """Returns the rank (close/SMA50 - 1) if entry fires, else None."""
        c, fast, slow = self._smas(df)
        if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]) or pd.isna(slow.iloc[-2]):
            return None
        cross_up = fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]
        if cross_up and c.iloc[-1] > slow.iloc[-1] \
                and wilder_rsi(c, P.rsi_slow).iloc[-1] < P.rsi_slow_max:
            return float(c.iloc[-1] / slow.iloc[-1] - 1)
        return None

    def momentum_exit(self, df: pd.DataFrame) -> bool:
        _, fast, slow = self._smas(df)
        if pd.isna(fast.iloc[-2]) or pd.isna(slow.iloc[-2]):
            return False
        return bool(fast.iloc[-1] < slow.iloc[-1] and fast.iloc[-2] >= slow.iloc[-2])

    def regime_ok(self, spy: pd.DataFrame) -> bool:
        sma = spy["close"].rolling(P.regime_sma).mean().iloc[-1]
        return bool(spy["close"].iloc[-1] > sma)

    # ---------------- guards ---------------- #

    def entries_blocked(self) -> str | None:
        if KILL_SWITCH.exists():
            return "KILL_SWITCH present"
        acct = self.trading.get_account()
        eq, last_eq = float(acct.equity or 0), float(acct.last_equity or 0)
        if last_eq > 0 and (last_eq - eq) / last_eq >= self.cfg.max_daily_loss_pct:
            return f"daily loss limit ({(last_eq - eq) / last_eq:.2%})"

        # Drawdown circuit breaker from the running high-water mark.
        hwm = max(float(self.state.d.get("hwm", 0)), eq)
        if hwm != self.state.d.get("hwm"):
            self.state.d["hwm"] = hwm
            self.state.save()
        if hwm > 0 and eq < hwm * (1 - self.cfg.dd_halt_pct):
            if not self.state.d.get("dd_halt_alerted"):
                alert("critical", "Drawdown circuit breaker tripped",
                      f"equity ${eq:,.0f} is {(1 - eq / hwm):.1%} below "
                      f"high-water ${hwm:,.0f} — entries halted, review needed")
                self.state.d["dd_halt_alerted"] = True
                self.state.save()
            return f"drawdown circuit breaker ({(1 - eq / hwm):.1%} from HWM)"
        if self.state.d.get("dd_halt_alerted"):
            self.state.d["dd_halt_alerted"] = False
            self.state.save()

        # FINRA retired the PDT rule (June 4, 2026); Alpaca now returns
        # daytrade_count as None and enforces intraday margin at order time
        # instead. This bot sizes positions from EQUITY (not buying_power),
        # so it never relies on the 4x margin the new framework unlocks —
        # margin sufficiency is the broker's problem, not a gate we check here.
        return None

    # ---------------- execution ---------------- #

    def open_positions(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self.trading.get_all_positions()}

    def _cancel_symbol_orders(self, symbol: str):
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        for o in self.trading.get_orders(req):
            self.trading.cancel_order_by_id(o.id)

    def _attach_stop(self, symbol: str, qty: float, coid: str | None = None):
        self.trading.submit_order(TrailingStopOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC, trail_percent=self.cfg.trail_pct,
            **({"client_order_id": coid} if coid else {})))

    def reconcile(self):
        """Self-healing audit, every wake-up: (a) every position carries
        protective sell coverage >= its size — missing/short protection is
        replaced and alerted; (b) sell orders without a position are orphans
        and get canceled; (c) state.json is synced to broker truth. In
        DRY_RUN it only reports what it would fix."""
        try:
            positions = self.open_positions()
            open_orders = self.trading.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.OPEN, limit=500))
        except Exception as exc:
            log.warning("reconcile: broker query failed (%s) — next wake-up", exc)
            return

        sells: dict[str, list] = defaultdict(list)
        for o in open_orders:
            if str(o.side.value) == "sell":
                sells[o.symbol].append(o)

        for sym, qty in positions.items():
            prot = sells.get(sym, [])
            covered = sum(float(o.qty or 0) for o in prot)
            if covered < qty - 1e-6:
                msg = f"{sym}: qty {qty} but protective coverage {covered}"
                if self.cfg.dry_run:
                    log.warning("[DRY RUN] reconcile would fix: %s", msg)
                    continue
                alert("critical", "Unprotected position — attaching stop", msg)
                try:
                    for o in prot:          # replace partial with full coverage
                        self.trading.cancel_order_by_id(o.id)
                    self._attach_stop(sym, qty)
                except Exception as exc:
                    alert("critical", f"{sym}: STOP ATTACH FAILED", str(exc))
            elif len(prot) > 1:
                # overlapping protection could double-sell into a short
                if self.cfg.dry_run:
                    log.warning("[DRY RUN] reconcile would trim %d extra "
                                "protective orders on %s", len(prot) - 1, sym)
                    continue
                alert("warning", f"{sym}: {len(prot)} protective orders",
                      "keeping the first, canceling extras")
                for o in prot[1:]:
                    try:
                        self.trading.cancel_order_by_id(o.id)
                    except Exception:
                        pass

        for sym, orders in sells.items():
            if sym not in positions:
                if self.cfg.dry_run:
                    log.warning("[DRY RUN] reconcile would cancel orphan sell "
                                "order(s) on %s", sym)
                    continue
                alert("warning", f"{sym}: orphan sell order(s)",
                      "position gone — canceling")
                for o in orders:
                    try:
                        self.trading.cancel_order_by_id(o.id)
                    except Exception:
                        pass

        st = self.state.d.setdefault("positions", {})
        changed = False
        for sym in list(st):
            if sym not in positions:
                st.pop(sym)
                changed = True
        for sym in positions:
            if sym not in st:
                st[sym] = {"entered": datetime.now(timezone.utc).isoformat()}
                changed = True
        if changed:
            self.state.save()

    def _heartbeat(self, note: str):
        try:
            (ROOT / "heartbeat.json").write_text(json.dumps({
                "at": datetime.now(timezone.utc).isoformat(),
                "note": note,
                "dry_run": self.cfg.dry_run,
                "last_cycle_date": self.state.last_run,
            }))
        except Exception:
            pass

    def enter(self, symbol: str, notional: float):
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        if self.trading.get_orders(req):
            return
        # Marketable limit: last trade + 0.5% cap. No live quote => no trade.
        try:
            trade = self.data.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbol))[symbol]
            last = float(trade.price)
        except Exception as exc:
            log.warning("%s: no live quote (%s) — skipping (fail closed)",
                        symbol, exc)
            return
        limit = round(last * (1 + self.cfg.limit_slip_pct), 2)
        qty = int(notional // limit)
        if qty < 1:
            log.info("%s: $%.2f/share exceeds the per-position budget — skipped",
                     symbol, limit)
            return
        if self.cfg.dry_run:
            log.info("[DRY RUN] BUY %d %s @ limit %.2f + %.0f%% trailing stop",
                     qty, symbol, limit, self.cfg.trail_pct)
            return

        coid = f"dm-e-{datetime.now(timezone.utc):%Y%m%d}-{symbol}"
        try:
            order = self.trading.submit_order(LimitOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY, limit_price=limit,
                client_order_id=coid))
        except Exception as exc:
            if "client_order_id" in str(exc).lower():
                log.info("%s: today's entry already submitted (idempotent id) "
                         "— skipping duplicate", symbol)
                return
            raise
        for _ in range(30):
            o = self.trading.get_order_by_id(order.id)
            if str(o.status.value) == "filled" and o.filled_qty:
                filled = float(o.filled_qty)
                try:
                    self._attach_stop(symbol, filled,
                                      coid=f"dm-t-{datetime.now(timezone.utc):%Y%m%d}-{symbol}")
                except Exception as exc:
                    alert("critical", f"{symbol}: stop attach failed after fill",
                          f"{exc} — reconciliation will retry within 5 min")
                self.state.opened(symbol)
                log.info("%s: filled %d @ %.2f, %.0f%% trail attached",
                         symbol, int(filled), float(o.filled_avg_price or 0),
                         self.cfg.trail_pct)
                return
            time.sleep(2)
        log.info("%s: limit order still working — reconciliation attaches the "
                 "stop whenever it fills", symbol)

    def close(self, symbol: str, reason: str):
        if self.cfg.dry_run:
            log.info("[DRY RUN] SELL %s (%s)", symbol, reason)
            return
        # No PDT deferral needed: FINRA retired the rule June 4, 2026.
        try:
            self._cancel_symbol_orders(symbol)   # only THIS symbol's stop
            self.trading.close_position(symbol)
        except Exception as exc:
            alert("critical", f"{symbol}: close FAILED ({reason})",
                  f"{exc} — position may be unprotected until next reconcile")
            return
        self.state.closed(symbol)
        log.info("%s: closed (%s)", symbol, reason)

    # ---------------- the once-per-day cycle ---------------- #

    def run_cycle(self) -> bool:
        """Returns True when today's cycle ran (or is done), False to retry."""
        self.reconcile()      # every wake-up, window or not: audit protection
        clock = self.trading.get_clock()
        now_ny = pd.Timestamp.now(tz=NY)
        today = str(now_ny.date())
        if self.state.last_run == today:
            return True
        if not clock.is_open:
            log.info("Market closed (next open %s)", clock.next_open)
            return False
        open_ny = pd.Timestamp(clock.next_close).tz_convert(NY).normalize() + \
            pd.Timedelta(hours=9, minutes=30)
        window_start = open_ny + pd.Timedelta(minutes=self.cfg.open_delay_min)
        window_end = open_ny.normalize() + pd.Timedelta(hours=self.cfg.window_end_hour)
        if now_ny < window_start:
            return False                       # wait for the window to open
        if now_ny >= window_end:
            log.info("Past today's %s-%02d:00 ET trade window — next try "
                     "tomorrow (stops stay active at the broker)",
                     window_start.strftime("%H:%M"), self.cfg.window_end_hour)
            self.state.last_run = today
            return True

        log.info("=== Daily cycle %s ===", today)
        positions = self.open_positions()
        needed = sorted(set(list(self.cfg.universe) + [self.cfg.regime_symbol])
                        | set(positions))
        bars = self.daily_bars(needed)

        spy = bars.get(self.cfg.regime_symbol)
        if spy is None:
            log.error("SPY data unavailable — failing closed, no action")
            self.state.last_run = today
            return True

        # 1) exits first: SMA cross-down on completed bars
        for sym in list(positions):
            df = bars.get(sym)
            if df is None:
                log.warning("%s: no data for exit check — holding (stop is live)", sym)
                continue
            if self.momentum_exit(df):
                self.close(sym, "cross_down")
                positions.pop(sym, None)

        # 2) entries
        blocked = self.entries_blocked()
        if blocked:
            log.warning("Entries blocked: %s", blocked)
            self.state.last_run = today
            return True
        if not self.regime_ok(spy):
            log.info("Regime risk-off (SPY < 200-day SMA) — no new entries")
            self.state.last_run = today
            return True

        candidates = []
        for sym in self.cfg.universe:
            if sym in positions or sym not in bars:
                continue
            rank = self.momentum_entry(bars[sym])
            if rank is not None:
                candidates.append((rank, sym))
        candidates.sort(reverse=True)

        acct = self.trading.get_account()
        # Dollar exposure cap: total invested capital never exceeds
        # max_total_exposure_usd; that budget is divided equally across the
        # position slots. Everything above the cap stays in cash.
        invested = sum(abs(float(p.market_value or 0))
                       for p in self.trading.get_all_positions())
        budget = self.cfg.max_total_exposure_usd - invested
        per_slot = self.cfg.max_total_exposure_usd / self.cfg.max_positions
        slots = self.cfg.max_positions - len(positions)
        log.info("%d candidates, %d slots, invested $%.0f / cap $%.0f, "
                 "$%.0f per position",
                 len(candidates), max(slots, 0), invested,
                 self.cfg.max_total_exposure_usd, per_slot)
        for rank, sym in candidates[:max(slots, 0)]:
            notional = min(per_slot, float(acct.buying_power or 0), budget)
            if notional < 100:
                log.info("Exposure cap reached ($%.0f left) — no more entries",
                         max(budget, 0))
                break
            self.enter(sym, notional)
            budget -= notional          # pessimistic: assume it fills in full

        self.state.last_run = today
        log.info("=== Cycle complete ===")
        return True

    def run_forever(self):
        log.info("Daily-momentum bot | %s | dry_run=%s | %d names | "
                 "max %d pos @ %.1f%% | trail %.0f%%",
                 "PAPER" if self.cfg.paper else "LIVE", self.cfg.dry_run,
                 len(self.cfg.universe), self.cfg.max_positions,
                 self.cfg.pos_frac * 100, self.cfg.trail_pct)
        while self._running:
            try:
                self._heartbeat("ok")
                self.run_cycle()
            except Exception as exc:
                log.exception("Cycle failed — retrying in 5 min")
                alert("critical", "Bot cycle crashed",
                      f"{type(exc).__name__}: {exc} — retrying in 5 min")
            for _ in range(300):
                if not self._running:
                    break
                time.sleep(1)
        self._heartbeat("stopped")
        log.info("Stopped cleanly")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(ROOT / "bot.log")])
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="run a single cycle attempt and exit")
    args = ap.parse_args()

    bot = Bot(Config.from_env())
    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    if args.once:
        bot._heartbeat("once")
        bot.run_cycle()
    else:
        bot.run_forever()


if __name__ == "__main__":
    main()
