"""
Alert channel for the bot — no silent failures.

Every alert goes three places:
  1. the log (CRITICAL/WARNING level),
  2. ALERTS.json next to this file (last 50, newest first) — the dashboard
     renders these as a red banner list,
  3. optionally an HTTP webhook if BOT_ALERT_WEBHOOK is set in the
     environment/.env (works with Slack/Discord/Telegram-bridge URLs;
     posts {"text": "..."} JSON).

Webhook failures never break the caller — an alert about the alert system
failing still lands in the log and ALERTS.json.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
ALERTS_FILE = ROOT / "ALERTS.json"
MAX_KEPT = 50

log = logging.getLogger("alerts")


def alert(severity: str, title: str, detail: str = "") -> None:
    """severity: 'critical' | 'warning' | 'info'"""
    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "severity": severity,
        "title": title,
        "detail": detail,
    }
    (log.critical if severity == "critical" else log.warning)(
        "ALERT [%s] %s %s", severity, title, detail)

    try:
        existing = json.loads(ALERTS_FILE.read_text()) if ALERTS_FILE.exists() else []
    except Exception:
        existing = []
    existing.insert(0, entry)
    try:
        ALERTS_FILE.write_text(json.dumps(existing[:MAX_KEPT], indent=1))
    except Exception:
        log.exception("could not write ALERTS.json")

    hook = os.environ.get("BOT_ALERT_WEBHOOK", "").strip()
    if hook:
        try:
            body = json.dumps(
                {"text": f"[{severity.upper()}] {title} — {detail}"}).encode()
            req = urllib.request.Request(
                hook, data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            log.warning("alert webhook failed (%s) — alert still logged", exc)
