import logging
import ccxt
import time
import sys
import os
from typing import Dict, Any, Optional

# Add project root to sys path to ensure config import works if run from src
sys.path.append(os.getcwd())
from config.secrets_loader import load_secrets

logger = logging.getLogger("binance_executor")

class BinanceExecutor:
    def __init__(self):
        # 1. Load Secrets securely
        secrets = load_secrets()
        self.api_key = secrets["BINANCE_API_KEY"]
        self.secret_key = secrets["BINANCE_API_SECRET"]
        self.testnet = secrets["BINANCE_TESTNET"]
        
        # 2. Safety State
        self.initial_balance = 0.0
        self.max_daily_loss = 0.02 # 2%
        self.kill_switch_drop = 0.03 # 3%
        self.is_live_safety_active = False # True if on Mainnet

        if not self.api_key or not self.secret_key:
            logger.info("ℹ️ Running in Simulation Mode (No API Keys provided).")
            self.exchange = None
        else:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future', 
                }
            })
            
            if self.testnet:
                self.exchange.set_sandbox_mode(True)
                logger.info("✅ Connected to Binance TESTNET 🟢")
            else:
                logger.warning("🚨 Connected to Binance MAINNET 🔴 (REAL MONEY AT RISK)")
                self.is_live_safety_active = True
                self._initialize_safety_locks()

    def _initialize_safety_locks(self):
        """On Mainnet start, capture balance and enforce limits."""
        try:
            balance = self.get_balance("USDT")
            if balance is None:
                raise ValueError("Could not fetch balance for safety check.")
                
            self.initial_balance = balance
            logger.info(f"🔒 Safety Lock Initialized. Start Balance: ${self.initial_balance:.2f} USDT")
            
            # Enforce 1x Leverage
            self._enforce_leverage(symbol="BTC/USDT", leverage=1)
            self._enforce_leverage(symbol="ETH/USDT", leverage=1)
            
        except Exception as e:
            logger.critical(f"❌ FAILED TO INITIALIZE SAFETY LOCKS: {e}")
            logger.critical("❌ SHUTTING DOWN TO PROTECT FUNDS.")
            sys.exit(1)

    def _check_safety_status(self):
        """Returns True if safe to trade. Raises exception if Kill Switch hit."""
        if not self.is_live_safety_active:
            return True
            
        current_balance = self.get_balance("USDT")
        if current_balance is None:
            logger.error("Safety Check: Could not fetch balance. Blocking trade.")
            return False
            
        drawdown_pct = (self.initial_balance - current_balance) / self.initial_balance
        
        if drawdown_pct >= self.kill_switch_drop:
            logger.critical(f"💀 MISSING KILL SWITCH! Drawdown {drawdown_pct*100:.2f}% >= {self.kill_switch_drop*100}%")
            logger.critical("🛑 TERMINATING BOT IMMEDIATELY.")
            sys.exit(1)
            
        if drawdown_pct >= self.max_daily_loss:
            logger.warning(f"🛑 Max Daily Loss Hit ({drawdown_pct*100:.2f}%). Trading Halted.")
            return False
            
        return True

    def _enforce_leverage(self, symbol: str = "BTC/USDT", leverage: int = 1):
        """Enforce leverage safety."""
        try:
            # Check current leverage first to avoid API spam? 
            # CCXT set_leverage
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"🔒 Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            if self.is_live_safety_active:
                 logger.critical("❌ UNSAFE STATE: Could not set leverage. Exiting.")
                 sys.exit(1)

    def get_balance(self, asset: str = "USDT") -> Optional[float]:
        """Get available balance."""
        if not self.exchange:
            return 10000.0 # Mock balance
            
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get(asset, {}).get('free', 0.0))
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return None

    def get_position_value(self, symbol: str = "BTC/USDT") -> float:
        """Get current position value (notional)."""
        if not self.exchange:
            return 0.0
        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                if pos['symbol'] == symbol:
                    return float(pos.get('notional', 0.0))
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching position: {e}")
            return 0.0

    def get_recent_trades(self, symbol: str = "BTC/USDT", limit: int = 10) -> list:
        if not self.exchange: return []
        try:
            return self.exchange.fetch_my_trades(symbol, limit=limit)
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []

    def execute_order(self, symbol: str, side: str, amount: float, order_type: str = "market", price: Optional[float] = None, params: Dict = {}, kwargs: Dict = {}) -> Optional[Dict]:
        """Execute Order with Safety Interlocks."""
        
        # 0. Safety Checks (Live Only)
        if self.is_live_safety_active:
            if not self._check_safety_status():
                 logger.warning(f"⚠️ Safety Lock: Rejecting {side} order.")
                 return None
            
            # Re-enforce leverage periodically or before open?
            # Doing it here adds latency. Let's assume initialized.
            
            # Check size limit? (Optional)
            
        if not self.exchange:
            logger.info(f"DRY-RUN: Executing {side.upper()} {amount} {symbol} @ {order_type}")
            return {"id": "mock_order_id", "status": "closed", "filled": amount}
            
        try:
            logger.info(f"🚀 Executing {side.upper()} {amount} {symbol}...")
            
            # Extract stops
            stops = params.pop("stops", None)
            
            if price is None and "price" in kwargs:
                price = kwargs["price"]
            
            # Create Order
            order = self.exchange.create_order(symbol, order_type, side, amount, price, params=params)
            logger.info(f"✅ Order Placed: {order['id']} | Status: {order['status']}")
            
            # Handle Stops
            if stops and order.get('status') in ['open', 'closed']:
                sl = stops.get("stop_loss")
                if sl:
                    sl_side = "sell" if side == "buy" else "buy"
                    try:
                        self.exchange.create_order(
                            symbol, "STOP_MARKET", sl_side, amount, 
                            params={"stopPrice": sl, "closePosition": True}
                        )
                        logger.info(f"🛑 Stop Loss Placed at {sl}")
                    except Exception as e_sl:
                        logger.error(f"Failed to place SL: {e_sl}")

            return order
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return None
