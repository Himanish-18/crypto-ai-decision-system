import asyncio
import logging
import time
from typing import Dict, List, Optional

import ccxt
import numpy as np
import pandas as pd

logger = logging.getLogger("market_router")


class MarketRouter:
    """
    Unified Market Data Engine.
    Aggregates data from multiple exchanges to provide a resilient, global view of the market.
    """

    def __init__(
        self,
        primary_exchange: str = "binance",
        secondary_exchanges: List[str] = ["bybit", "okx", "coinbasepro"],
    ):
        self.primary_name = primary_exchange
        self.secondary_names = secondary_exchanges
        self.exchanges: Dict[str, ccxt.Exchange] = {}

        # Initialize Exchanges
        self._init_exchange(self.primary_name)
        for name in self.secondary_names:
            self._init_exchange(name)

    def _init_exchange(self, name: str):
        try:
            exchange_class = getattr(ccxt, name)
            self.exchanges[name] = exchange_class(
                {
                    "enableRateLimit": True,
                    "timeout": 10000,  # Increased to 10s
                }
            )
            # Load markets to verify connection
            # self.exchanges[name].load_markets() # Can be slow, do lazy load or async
            logger.info(f"✅ Initialized Exchange: {name}")
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")

    async def fetch_unified_candles(
        self, symbol: str = "BTC/USDT", timeframe: str = "1m", limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch OHLCV from Primary. If fail, switch to Secondary.
        Returns standardized DataFrame.
        """
        # Try Primary
        df = await self._fetch_exchange_candles(
            self.primary_name, symbol, timeframe, limit
        )
        if df is not None:
            return df

        # Failover
        for name in self.secondary_names:
            logger.warning(f"⚠️ Primary {self.primary_name} failed. Trying {name}...")
            df = await self._fetch_exchange_candles(name, symbol, timeframe, limit)
            if df is not None:
                logger.info(f"✅ Failover successful using {name}")
                return df

        logger.error("❌ All exchanges failed to fetch candles.")
        return None

    async def _fetch_exchange_candles(
        self, exchange_name: str, symbol: str, timeframe: str, limit: int
    ) -> Optional[pd.DataFrame]:
        exchange = self.exchanges.get(exchange_name)
        if not exchange:
            return None

        # Retry Logic (3 attempts)
        for attempt in range(3):
            try:
                # Sync wrapper for async context if needed
                # Using asyncio.to_thread to run blocking ccxt call
                ohlcv = await asyncio.to_thread(
                    exchange.fetch_ohlcv, symbol, timeframe, limit=limit
                )

                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                # Standardization Columns
                # v16 Update: Keep generic names 'open', 'close' for easier downstream processing
                # The caller knows which symbol they asked for.

                # If we need asset specific later, we can add aliases, but primary should be standard.
                base_asset = symbol.split("/")[0].lower()
                # Add aliases for backward compatibility if needed, but primary is standard
                df[f"{base_asset}_open"] = df["open"]
                df[f"{base_asset}_high"] = df["high"]
                df[f"{base_asset}_low"] = df["low"]
                df[f"{base_asset}_close"] = df["close"]
                df[f"{base_asset}_volume"] = df["volume"]

                return df

            except Exception as e:
                logger.warning(
                    f"Fetch failed for {exchange_name} (Attempt {attempt+1}/3): {e}"
                )
                await asyncio.sleep(2)  # Backoff

        logger.error(f"❌ {exchange_name} failed after 3 attempts.")
        return None

    async def get_aggregated_depth(
        self, symbol: str = "BTC/USDT", depth_limit: int = 10
    ) -> Dict:
        """
        Fetch depth from ALL exchanges and aggregate to find global liquidity.
        Returns: {'bids': [[price, vol], ...], 'asks': ...}
        """
        tasks = []
        valid_exchanges = []

        for name, exchange in self.exchanges.items():
            tasks.append(
                asyncio.to_thread(exchange.fetch_order_book, symbol, depth_limit)
            )
            valid_exchanges.append(name)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_bids = []
        all_asks = []

        for name, res in zip(valid_exchanges, results):
            if isinstance(res, Exception):
                logger.warning(f"Depth fetch failed for {name}: {res}")
                continue

            all_bids.extend(res["bids"])
            all_asks.extend(res["asks"])

        # Sort Bids Desc, Asks Asc
        all_bids.sort(key=lambda x: x[0], reverse=True)
        all_asks.sort(key=lambda x: x[0])

        # We could aggregate volumes at same price levels, but for now raw list is fine for "Thickness" calc

        return {
            "bids": all_bids[: depth_limit * len(valid_exchanges)],
            "asks": all_asks[: depth_limit * len(valid_exchanges)],
            "source_count": len(valid_exchanges),
        }
