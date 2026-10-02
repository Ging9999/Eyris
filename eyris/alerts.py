"""Round notifications to your phone (ntfy) and/or Discord. Optional and never fatal.

Configure with environment variables (either or both):
  NTFY_TOPIC           a hard-to-guess topic name; install the ntfy app and subscribe to it.
                       ntfy.sh topics are readable by anyone who knows the name, so pick a random one.
  NTFY_SERVER          default https://ntfy.sh
  DISCORD_WEBHOOK_URL  a channel webhook (Server settings > Integrations > Webhooks)

Messages carry only the round id, the outcome, the reason, the holdings source and
the receipt status: never weights, tokens or credentials.
"""
import json
import os
import urllib.request

TIMEOUT = 10


def _post(url, data, headers):
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status


def notify(title, message, urgent=False):
    """Send to every configured channel. Returns the channels that accepted it."""
    sent = []
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        try:
            _post(f"{server}/{topic}", message.encode("utf-8"),
                  {"Title": title.encode("ascii", "replace").decode(), "Priority": "high" if urgent else "default",
                   "Tags": "warning" if urgent else "chart_with_upwards_trend"})
            sent.append("ntfy")
        except Exception:
            pass
    hook = os.environ.get("DISCORD_WEBHOOK_URL")
    if hook:
        try:
            body = json.dumps({"content": f"{'⚠️ ' if urgent else ''}**{title}**\n{message}"[:1900]}).encode()
            _post(hook, body, {"Content-Type": "application/json", "User-Agent": "eyris-alerts"})
            sent.append("discord")
        except Exception:
            pass
    return sent


def round_summary(status, round_id, log=None, receipt=None):
    """(title, message, urgent) for one `run` outcome."""
    log = log or {}
    receipt_status = receipt.get("status") if isinstance(receipt, dict) else None
    lines = [f"status: {status}"]
    if log:
        lines += [f"reason: {log.get('reason')}", f"holdings: {log.get('holdings_source')}",
                  f"last bar: {log.get('last_bar')}"]
        if log.get("vix_multiplier", 1.0) != 1.0:
            lines.append(f"vix overlay x{log['vix_multiplier']}")
        if log.get("news_trims"):
            lines.append(f"news trims: {', '.join(log['news_trims'])}")
    if receipt_status:
        lines.append(f"receipt: {receipt_status}")
    urgent = (status not in ("UPLOADED", "HOLD_NO_UPLOAD", "SLOT_ALREADY_USED")
              or log.get("breaker") or log.get("holdings_source") in ("none", "local_state")
              or (receipt_status is not None and receipt_status not in ("PENDING_SELECTION", "VALID", "EXECUTED",
                                                                         "SELECTED", "ACCEPTED")))
    return f"Eyris {round_id}", "\n".join(lines), bool(urgent)
