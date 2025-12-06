import logging
import ccxt
import os
from typing import Dict, Any, Optional

logger = logging.getLogger("binance_executor")

class BinanceExecutor:
    def __init__(self, api_key: str = None, secret_key: str = None):
        # Security: Prefer env vars over args
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.secret_key = os.getenv("BINANCE_SECRET_KEY")
        
        # Mode Selection
        # Default to TESTNET unless GO_LIVE is explicitly set to "true"
        self.testnet = os.getenv("GO_LIVE", "false").lower() != "true"
        
        if not self.api_key or not self.secret_key:
            logger.info("ℹ️ Running in Simulation Mode (No API Keys provided).")
            self.exchange = None
        else:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future', # Assuming Futures trading
                }
            })
            if self.testnet:
                self.exchange.set_sandbox_mode(True)
                logger.info("✅ Connected to Binance TESTNET")
            else:
                logger.warning("🚨 Connected to Binance MAINNET (GO_LIVE=true)")
                self._enforce_leverage()

    def _enforce_leverage(self, symbol: str = "BTC/USDT", leverage: int = 1):
        """Enforce 1.0x leverage for safety on Mainnet."""
        try:
            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"🔒 Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")

    def get_balance(self, asset: str = "USDT") -> Optional[float]:
        """Get available balance."""
        if not self.exchange:
            return 10000.0 # Mock balance for dry-run
            
        try:
            balance = self.exchange.fetch_balance()
            return balance.get(asset, {}).get('free', 0.0)
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return None

    def get_position_value(self, symbol: str = "BTC/USDT") -> float:
        """Get current position value (notional)."""
        if not self.exchange:
            return 0.0 # Dry-run
            
        try:
            # fetch_positions is unified in ccxt, but sometimes requires specific params
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                if pos['symbol'] == symbol:
                    # notional = abs(positionAmt * markPrice)
                    return float(pos.get('notional', 0.0))
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching position: {e}")
            return 0.0

    def get_recent_trades(self, symbol: str = "BTC/USDT", limit: int = 10) -> list:
        """Fetch recent user trades to calculate PnL."""
        if not self.exchange:
            return [] # Dry-run
            
        try:
            # fetch_my_trades returns list of trades
            trades = self.exchange.fetch_my_trades(symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []

    def execute_order(self, symbol: str, side: str, amount: float, order_type: str = "market", price: Optional[float] = None, params: Dict = {}, kwargs: Dict = {}) -> Optional[Dict]:
        """
        Execute an order.
        side: 'buy' or 'sell'
        """
        if not self.exchange:
            logger.info(f"DRY-RUN: Executing {side.upper()} {amount} {symbol} @ {order_type}")
            return {"id": "mock_order_id", "status": "closed", "filled": amount}
            
        try:
            logger.info(f"🚀 Executing {side.upper()} {amount} {symbol}...")
            
            # Extract stops to handle separately if needed
            stops = params.pop("stops", None)
            
            # Handle Price (either arg or kwargs)
            if price is None and "price" in kwargs:
                price = kwargs["price"]
            
            order = self.exchange.create_order(symbol, order_type, side, amount, price, params=params)
            logger.info(f"Order Placed: {order['id']} | Status: {order['status']}")
            
            # Handle Stops
            if stops and order.get('status') in ['open', 'closed']:
                sl = stops.get("stop_loss")
                tp = stops.get("take_profit")
                
                if sl:
                    # Place Stop Market
                    sl_side = "sell" if side == "buy" else "buy"
                    try:
                        sl_order = self.exchange.create_order(
                            symbol, "STOP_MARKET", sl_side, amount, 
                            params={"stopPrice": sl, "closePosition": True} # ReduceOnly
                        )
                        logger.info(f"🛑 Stop Loss Placed at {sl}")
                    except Exception as e_sl:
                        logger.error(f"Failed to place SL: {e_sl}")

            return order
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return None
