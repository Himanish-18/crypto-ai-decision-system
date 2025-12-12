import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("observability_alerts")


class AlertRouter:
    """
    Routes Critical 'Hard-Veto' alerts to notification channels.
    """

    def __init__(self, channels: Optional[Dict[str, Any]] = None):
        self.channels = channels or {}

    def send_alert(self, level: str, message: str, context: Dict[str, Any] = None):
        """
        Send alert based on severity.
        """
        context_str = str(context) if context else ""
        full_message = f"[{level}] {message} | {context_str}"

        logger.error(f"🚨 ALERT: {full_message}")

        if level == "CRITICAL":
            # Hard-Veto logic: Could trigger a kill switch
            self._trigger_kill_switch()
            self._notify_external(full_message)

    def _trigger_kill_switch(self):
        logger.critical("☠️ KILL SWITCH TRIGGERED: Halting all execution.")
        # Logic to write a 'STOP' file or kill process

    def _notify_external(self, message: str):
        # Placeholder for Slack/Email integration
        # requests.post(webhook_url, json={"text": message})
        pass
