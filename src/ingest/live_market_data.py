import logging
import time
from typing import Any, Dict, Optional

import ccxt
import pandas as pd

logger = logging.getLogger("live_market_data")


class LiveMarketData:
    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        exchange_id: str = "binance",
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = getattr(ccxt, exchange_id)(
            {
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future"
                },  # Use futures for funding rates etc if needed
            }
        )

    def fetch_candles(self, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch N most recent candles."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            if not ohlcv:
                return None

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

            prefix = "btc" if "BTC" in self.symbol else "eth"
            df.columns = ["timestamp"] + [
                f"{prefix}_{c}" for c in ["open", "high", "low", "close", "volume"]
            ]

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

            prefix = "btc" if "BTC" in self.symbol else "eth"
            close_col = f"{prefix}_close"

            logger.info(
                f"Fetched candle: {latest_closed['timestamp'].iloc[0]} | Close: {latest_closed[close_col].iloc[0]}"
            )
            return latest_closed

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    def fetch_current_price(self) -> float:
        """Fetch current ticker price."""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker["last"]
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return 0.0

    def fetch_order_book(self, limit: int = 100) -> Optional[Dict[str, Any]]:
        """
        Fetch Level 2 Order Book (Bids/Asks).
        """
        try:
            order_book = self.exchange.fetch_order_book(self.symbol, limit=limit)
            return {
                "timestamp": order_book["timestamp"],
                "bids": order_book["bids"],  # [[price, amount], ...]
                "asks": order_book["asks"],
            }
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None

    def fetch_recent_trades(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch recent trades for CVD calculation.
        """
        try:
            trades = self.exchange.fetch_trades(self.symbol, limit=limit)
            df = pd.DataFrame(trades)
            # Keep relevant columns
            df = df[["timestamp", "side", "price", "amount", "cost"]]
            return df
        except Exception as e:
            logger.error(f"Error fetching recent trades: {e}")
            return None
