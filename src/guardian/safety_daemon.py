import logging
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger("guardian")

class SafetyDaemon:
    def __init__(self, data_dir: Path, initial_capital: float = 10000.0):
        self.data_dir = data_dir
        self.guardian_dir = data_dir / "guardian"
        self.guardian_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.guardian_dir / "state.json"
        
        self.initial_capital = initial_capital
        self.state = self._load_state()
        
        # Thresholds
        self.max_daily_drawdown_pct = 0.02 # Updated to 2%
        self.max_total_drawdown_pct = 0.15 
        self.max_volatility_percentile = 0.99
        self.max_data_delay_seconds = 300 
        self.max_slippage_pct = 0.005 
        self.max_exposure_pct = 0.05 # 5% Total Open Exposure
        self.max_losing_streak = 3 # Kill switch
        
    def _load_state(self):
        # Default State
        default_state = {
            "start_of_day_equity": self.initial_capital,
            "last_update_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "is_locked": False,
            "lock_reason": None,
            "losing_streak": 0
        }

        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    loaded_state = json.load(f)
                    # Merge defaults to ensure all keys exist
                    for k, v in default_state.items():
                        if k not in loaded_state:
                            loaded_state[k] = v
                    return loaded_state
            except Exception as e:
                logger.error(f"Failed to load guardian state: {e}")
        
        return default_state

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=4)

    def update_equity(self, current_equity: float):
        """Update equity state, handling day rollover."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        
        if today != self.state["last_update_date"]:
            # New Day: Reset daily tracking
            self.state["start_of_day_equity"] = current_equity
            self.state["last_update_date"] = today
            logger.info(f"Guardian: New Day. Reset Start Equity to {current_equity}")
            
        self._save_state()

    def check_system_health(self, model, scaler, features_df: pd.DataFrame, timeframe_minutes: int = 60) -> bool:
        """Check if critical components are loaded and data is fresh."""
        # 1. Component Check
        if model is None or scaler is None:
            logger.critical("Guardian: Model or Scaler missing! HALTING.")
            return False
            
        if features_df is None or features_df.empty:
            logger.critical("Guardian: Features DataFrame empty! HALTING.")
            return False
            
        # 2. Data Freshness Check
        last_timestamp = features_df["timestamp"].max()
        # Ensure timestamp is tz-aware UTC for comparison
        if last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
            
        now = pd.Timestamp.utcnow()
        delay = (now - last_timestamp).total_seconds()
        
        # Allow delay up to timeframe + buffer
        max_allowed_delay = (timeframe_minutes * 60) + self.max_data_delay_seconds
        
        if delay > max_allowed_delay:
            logger.error(f"Guardian: Data Stale! Delay: {delay:.0f}s > {max_allowed_delay}s. PAUSING.")
            logger.info(f"Debug: Last TS: {last_timestamp}, Now: {now}")
            return False
            
        return True

    def check_market_conditions(self, market_state: dict) -> bool:
        """Check if market conditions are safe for trading."""
        # 1. Volatility Check (ATR > 99th percentile)
        atr_pct = market_state.get("atr_pct", 0)
        if atr_pct > self.max_volatility_percentile:
            logger.warning(f"Guardian: Extreme Volatility (ATR Pct: {atr_pct:.4f}). NO NEW TRADES.")
            return False
            
        return True

    def check_financial_health(self, current_equity: float) -> bool:
        """Check drawdown and capital preservation limits."""
        # 0. Check Lock
        if self.state["is_locked"]:
            logger.critical(f"Guardian: Account LOCKED due to: {self.state['lock_reason']}")
            return False
            
        self.update_equity(current_equity)
        
        # 1. Total Drawdown (Capital Preservation)
        total_dd = (self.initial_capital - current_equity) / self.initial_capital
        if total_dd > self.max_total_drawdown_pct:
            self.state["is_locked"] = True
            self.state["lock_reason"] = f"Total Drawdown {total_dd*100:.2f}% > {self.max_total_drawdown_pct*100}%"
            self._save_state()
            logger.critical(f"Guardian: {self.state['lock_reason']}. LOCKING ACCOUNT.")
            return False
            
        # 2. Daily Drawdown
        start_equity = self.state["start_of_day_equity"]
        daily_dd = (start_equity - current_equity) / start_equity
        if daily_dd > self.max_daily_drawdown_pct:
            logger.error(f"Guardian: Daily Drawdown {daily_dd*100:.2f}% > {self.max_daily_drawdown_pct*100}%. TRADING DISABLED FOR TODAY.")
            return False
            
        return True

    def check_trade_streak(self, last_trade_pnl: float):
        """Update and check losing streak."""
        if last_trade_pnl < 0:
            self.state["losing_streak"] += 1
        else:
            self.state["losing_streak"] = 0
            
        if self.state["losing_streak"] >= self.max_losing_streak:
            self.state["is_locked"] = True
            self.state["lock_reason"] = f"Kill Switch: {self.state['losing_streak']} consecutive losses."
            logger.critical(f"Guardian: {self.state['lock_reason']} LOCKING ACCOUNT.")
            
        self._save_state()

    def check_exposure(self, current_exposure: float, total_capital: float) -> bool:
        """Check if total open exposure exceeds limit."""
        exposure_pct = current_exposure / total_capital
        if exposure_pct > self.max_exposure_pct:
            logger.warning(f"Guardian: Max Exposure Reached ({exposure_pct*100:.2f}% > {self.max_exposure_pct*100}%). NO NEW TRADES.")
            return False
        return True

    def check_execution_safety(self, current_price: float, order_price: float) -> bool:
        """Check for excessive slippage or bad execution."""
        # Simple slippage check: |Order - Current| / Current
        # In live market orders, we might check spread instead.
        # Here we assume order_price is what we expect to pay.
        
        diff = abs(current_price - order_price) / current_price
        if diff > self.max_slippage_pct:
            logger.warning(f"Guardian: High Slippage Risk! Diff: {diff*100:.2f}%. HOLDING.")
            return False
            
        return True
