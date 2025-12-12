import logging
import os
from typing import Optional

import requests


class TelegramAlertHandler(logging.Handler):
    """
    Custom Logging Handler to send critical messages to Telegram.
    """

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        super().__init__()
        self.token = token or os.getenv("TELEGRAM_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def emit(self, record):
        if not self.token or not self.chat_id:
            return

        # Only send CRITICAL or ERROR messages (or custom level)
        if record.levelno >= logging.CRITICAL:
            log_entry = self.format(record)
            self.send_message(f"🚨 **CRITICAL ALERT** 🚨\n\n{log_entry}")

    def send_message(self, text: str):
        try:
            payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
            requests.post(self.api_url, json=payload, timeout=5)
        except Exception as e:
            # Fallback to standard error to avoid infinite loop
            print(f"Failed to send Telegram alert: {e}")
