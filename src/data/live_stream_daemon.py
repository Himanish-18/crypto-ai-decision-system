import logging
import time
import asyncio
import pandas as pd
import ccxt.async_support as ccxt
from pathlib import Path
from datetime import datetime

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - STREAM - %(message)s")
logger = logging.getLogger("live_stream")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUFFER_DIR = PROJECT_ROOT / "data" / "live_buffer"
BUFFER_DIR.mkdir(parents=True, exist_ok=True)

class LiveStreamDaemon:
    """
    Simulates a WebSocket stream by high-frequency polling (1s).
    Buffers Tick Data (Price, Depth, Attributes) to Parquet.
    """
    def __init__(self, symbols=["BTC/USDT"], exchanges=["binance", "bybit", "okx"]):
        self.symbols = symbols
        self.exchange_names = exchanges
        self.exchanges = {}
        self.buffer = []
        self.flush_interval = 60 # Flush every 60 ticks (approx 1 min)
        
    async def _init_exchanges(self):
        for name in self.exchange_names:
            try:
                exchange_class = getattr(ccxt, name)
                self.exchanges[name] = exchange_class({
                    'enableRateLimit': True, # Important for polling
                    # 'timeout': 2000 
                })
                logger.info(f"🔌 Connected to {name}")
            except Exception as e:
                logger.error(f"Failed to connect to {name}: {e}")

    async def _fetch_ticker(self, name, exchange, symbol):
        try:
            ticker = await exchange.fetch_ticker(symbol)
            # Standardize
            return {
                "timestamp": datetime.now(), # Use local capture time for stream alignment
                "exchange": name,
                "symbol": symbol,
                "bid": ticker["bid"],
                "ask": ticker["ask"],
                "last": ticker["last"],
                "volume": ticker["baseVolume"],
                "quote_volume": ticker["quoteVolume"]
            }
        except Exception as e:
            # logger.warning(f"{name} ticker fail: {e}")
            return None

    async def _fetch_depth(self, name, exchange, symbol):
        try:
             # Depth 5 is enough for top-of-book signal
             orderbook = await exchange.fetch_order_book(symbol, limit=5)
             
             bids = orderbook['bids']
             asks = orderbook['asks']
             
             # Calculate simple Liquidity Imbalance
             bid_vol = sum([b[1] for b in bids])
             ask_vol = sum([a[1] for a in asks])
             
             # Weighted Mid Price
             best_bid = bids[0][0] if bids else None
             best_ask = asks[0][0] if asks else None
             
             return {
                 "depth_bid_vol": bid_vol,
                 "depth_ask_vol": ask_vol,
                 "best_bid": best_bid,
                 "best_ask": best_ask,
                 "spread_pct": (best_ask - best_bid) / best_bid if best_ask and best_bid else 0
             }
        except Exception:
            return None

    async def start(self):
        await self._init_exchanges()
        logger.info("🌊 Stream Started. Buffering data...")
        
        while True:
            start_time = time.time()
            
            # For each exchange, fetch data
            tasks = []
            for name, exchange in self.exchanges.items():
                for symbol in self.symbols:
                    tasks.append(self._fetch_ticker(name, exchange, symbol))
                    
            results = await asyncio.gather(*tasks)
            
            # Process Results
            snapshot = {
                "timestamp": datetime.now(),
                "tick_count": len(self.buffer)
            }
            
            # Simple aggregation for this tick
            valid_ticks = [r for r in results if r is not None]
            if valid_ticks:
                # Average Price across exchanges
                avg_price = sum([t["last"] for t in valid_ticks]) / len(valid_ticks)
                snapshot["global_price"] = avg_price
                
                # Append to buffer
                # In real high freq, we'd save every tick. 
                # Here we save the aggregated snapshot per second.
                self.buffer.append(snapshot)
            
            # Flush Check
            if len(self.buffer) >= self.flush_interval:
                await self._flush_buffer()
                
            # Sleep remainder of 1s
            elapsed = time.time() - start_time
            sleep_time = max(0.1, 1.0 - elapsed)
            await asyncio.sleep(sleep_time)

    async def _flush_buffer(self):
        if not self.buffer: return
        
        df = pd.DataFrame(self.buffer)
        filename = f"stream_{int(time.time())}.parquet"
        path = BUFFER_DIR / filename
        
        # Save async? simple to_parquet is fast enough for 60 rows
        df.to_parquet(path)
        logger.info(f"💾 Flushed {len(df)} ticks to {filename}")
        
        self.buffer = [] # Clear

    async def shutdown(self):
        for ex in self.exchanges.values():
            await ex.close()

if __name__ == "__main__":
    daemon = LiveStreamDaemon()
    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        logger.info("Stream Stopped.")
