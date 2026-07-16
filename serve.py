"""
Cloud entry point (Railway/any host): run the trading bot loop AND the
dashboard in one process.

- The bot runs in a daemon thread (its once-per-day cycle, reconciliation,
  heartbeat) exactly as `python bot.py` would.
- Flask serves the dashboard on 0.0.0.0:$PORT so the host can route to it.

Locally you'd still run `python bot.py` and `python dashboard.py` separately;
this file exists so a single Railway service covers both. Set DASHBOARD_USER
and DASHBOARD_PASS in the host's variables to password-protect the public
dashboard (strongly recommended — it shows the account and can place sells).

Note on cloud filesystems: state.json/heartbeat.json are ephemeral and reset
on redeploy. That's fine — the bot's reconcile() rebuilds position state from
the broker (the source of truth) on the next cycle; only the drawdown
high-water mark resets, which is acceptable for a paper test.
"""
from __future__ import annotations

import logging
import os
import sys
import threading

from bot import Bot, Config
from dashboard import app


def _run_bot():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])
    try:
        Bot(Config.from_env()).run_forever()
    except Exception:
        logging.getLogger("serve").exception("bot thread died")


if __name__ == "__main__":
    threading.Thread(target=_run_bot, daemon=True, name="bot").start()
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
