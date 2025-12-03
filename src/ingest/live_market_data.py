import logging
import ccxt
import pandas as pd
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("live_market_data")

class LiveMarketData:
    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1h", exchange_id: str = "binance"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })
        
    def fetch_candles(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch N most recent candles."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            if not ohlcv:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # CCXT usually returns open candle as last.
            # We want to process the *last closed* candle for decision, 
            # but we need history for features.
            # We will return all fetched candles. 
            # The orchestrator will decide which one is the "target" candle (usually iloc[-2] if open is present).
            
            # For simplicity, let's just return what we get.
            # But we need to be careful about column names matching build_features.py
            # build_features expects: btc_close, btc_high, etc.
            # We are fetching BTC/USDT, so we should rename cols.
            
            # Assuming symbol is BTC/USDT
            prefix = "btc" if "BTC" in self.symbol else "eth"
            df.columns = ["timestamp"] + [f"{prefix}_{c}" for c in ["open", "high", "low", "close", "volume"]]
            
            return df
        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            return None

    def fetch_latest_candle(self) -> Optional[pd.DataFrame]:
        """
        Fetch the latest CLOSED candle.
        """
        try:
            # Fetch 2 candles to ensure we get the last closed one
            df = self.fetch_candles(limit=2)
            if df is None or len(df) < 2:
                return None
            
            # Return the second to last candle (latest closed)
            latest_closed = df.iloc[[-2]].copy().reset_index(drop=True)
            
            # Rename back for logging if needed, or keep as is.
            # The logging below expects 'close', but we renamed to 'btc_close'
            prefix = "btc" if "BTC" in self.symbol else "eth"
            close_col = f"{prefix}_close"
            
            logger.info(f"Fetched candle: {latest_closed['timestamp'].iloc[0]} | Close: {latest_closed[close_col].iloc[0]}")
            return latest_closed
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    def fetch_current_price(self) -> float:
        """Fetch current ticker price."""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return 0.0
