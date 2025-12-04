import time
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from src.ingest.live_market_data import LiveMarketData

logger = logging.getLogger("l2_collector")
logging.basicConfig(level=logging.INFO)

class L2DataCollector:
    def __init__(self, symbol: str = "BTC/USDT", data_dir: str = "./data/l2"):
        self.market_data = LiveMarketData(symbol=symbol)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol.replace("/", "")

    def collect_snapshot(self):
        """
        Collect one snapshot of Order Book and Trades.
        """
        timestamp = datetime.now()
        
        # 1. Order Book
        ob = self.market_data.fetch_order_book(limit=100)
        if ob:
            # Flatten for CSV: timestamp, best_bid, best_ask, bid_vol_10, ask_vol_10...
            # For simplicity, we'll save the raw JSON-like structure or just top levels
            # Let's save top 5 levels for CSV simplicity
            snapshot = {
                'timestamp': timestamp,
                'symbol': self.symbol,
                'best_bid': ob['bids'][0][0],
                'best_ask': ob['asks'][0][0],
                'bid_vol_0': ob['bids'][0][1],
                'ask_vol_0': ob['asks'][0][1],
            }
            # Add more depth if needed
            self.save_to_csv(snapshot, "orderbook")

        # 2. Trades (Recent)
        trades = self.market_data.fetch_recent_trades(limit=50)
        if trades is not None and not trades.empty:
            # Append to trades file
            trades['collected_at'] = timestamp
            self.save_df_to_csv(trades, "trades")
            
        logger.info(f"Collected snapshot at {timestamp}")

    def save_to_csv(self, data: dict, type_: str):
        file_path = self.data_dir / f"{self.symbol}_{type_}.csv"
        df = pd.DataFrame([data])
        header = not file_path.exists()
        df.to_csv(file_path, mode='a', header=header, index=False)

    def save_df_to_csv(self, df: pd.DataFrame, type_: str):
        file_path = self.data_dir / f"{self.symbol}_{type_}.csv"
        header = not file_path.exists()
        df.to_csv(file_path, mode='a', header=header, index=False)

    def run(self, interval: int = 60):
        logger.info(f"Starting L2 Collection for {self.symbol} every {interval}s")
        try:
            while True:
                self.collect_snapshot()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Collection stopped.")

if __name__ == "__main__":
    collector = L2DataCollector()
    # Run once for testing, in production this would loop
    collector.collect_snapshot()
