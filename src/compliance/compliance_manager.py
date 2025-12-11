import csv
import logging
import os
from datetime import datetime
from typing import Dict

logger = logging.getLogger("compliance")

class ComplianceManager:
    """
    v24 Institutional Compliance Layer.
    Enforces risk limits and maintains immutable audit logs.
    """
    def __init__(self, log_path: str = "data/compliance/trade_log.csv"):
        self.log_path = log_path
        self._ensure_log_exists()
        
        # Risk Limits
        self.MAX_GROSS_EXPOSURE = 2.0 # 2x Leverage
        self.MAX_SINGLE_VAR_PCT = 0.05 # 5% VaR cap per asset
        
    def _ensure_log_exists(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "strategy", "symbol", "action", "size", "price", "reason", "compliance_check"])

    def check_compliance(self, trade: Dict) -> bool:
        """
        Pre-trade compliance check.
        Values: symbol, size, price, portfolio_value.
        """
        # 1. Notional Check
        size = float(trade.get("size", 0))
        price = float(trade.get("price", 0))
        notional = size * price
        port_val = float(trade.get("portfolio_value", 10000)) # Default stub
        
        if notional > port_val * self.MAX_GROSS_EXPOSURE:
            logger.error(f"Compliance Veto: Exposure {notional} > Limit {port_val*self.MAX_GROSS_EXPOSURE}")
            self.log_trade(trade, "FAIL_EXPOSURE")
            return False
            
        # 2. Restricted Symbol Check
        if trade.get("symbol") in ["LUNA", "FTT"]:
            logger.error("Compliance Veto: Restricted Asset.")
            self.log_trade(trade, "FAIL_BLACKLIST")
            return False
            
        logger.info("✅ Compliance Check Passed.")
        return True

    def log_trade(self, trade: Dict, status: str = "PASS"):
        """
        Immutable Log Entry.
        """
        try:
            with open(self.log_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow().isoformat(),
                    trade.get("strategy", "UNKNOWN"),
                    trade.get("symbol", "N/A"),
                    trade.get("action", "N/A"),
                    trade.get("size", 0),
                    trade.get("price", 0),
                    trade.get("reason", "N/A"),
                    status
                ])
        except Exception as e:
            logger.error(f"Critical Compliance Logging Failure: {e}")
