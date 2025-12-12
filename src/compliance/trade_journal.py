import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("compliance_journal")


@dataclass
class TradeEntry:
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    timestamp: float
    strategy_id: str
    execution_venue: str

    def to_fix_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary mimicking FIX tags.
        """
        return {
            "35": "D",  # MsgType: New Order Single
            "11": self.order_id,  # ClOrdID
            "55": self.symbol,  # Symbol
            "54": "1" if self.side == "BUY" else "2",  # Side
            "38": self.qty,  # OrderQty
            "44": self.price,  # Price
            "60": self.timestamp,  # TransactTime
            "1": self.strategy_id,  # Account/Strategy
            "100": self.execution_venue,  # ExDestination (Custom)
        }


class TradeJournal:
    """
    Immutable Journal for Trade Logging.
    """

    def __init__(self, log_dir: str = "logs/compliance"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.journal_file = self.log_dir / "trade_journal.csv"
        self._ensure_file()

    def _ensure_file(self):
        if not self.journal_file.exists():
            with open(self.journal_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "FIX_Message"])

    def log_trade(self, entry: TradeEntry):
        """
        Log a trade entry.
        """
        fix_msg = json.dumps(entry.to_fix_dict())
        with open(self.journal_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), fix_msg])
        logger.info(f"📝 Compliance: Trade logged {entry.order_id}")
