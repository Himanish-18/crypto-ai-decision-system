import os
import sys
import time

import ccxt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("market_ingest")
config = load_config()


def fetch_market_data(symbol="BTC/USDT", timeframe="1h", since=None, limit=1000):
    """Fetch large-scale OHLCV data in multiple batches (e.g. 200k+ rows)."""
    exchange = ccxt.binance()
    all_data = []
    since = exchange.parse8601("2019-01-01T00:00:00Z") if not since else since
    logger.info(
        f"📡 Fetching {symbol} {timeframe} data from {pd.to_datetime(since, unit='ms')}..."
    )

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=limit
            )
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            logger.info(f"Fetched {len(all_data)} rows so far for {symbol}...")
            time.sleep(1)  # to respect rate limits
            if len(all_data) >= 200_000:
                break
        except Exception as e:
            logger.error(f"Error fetching: {e}")
            time.sleep(5)
            continue

    df = pd.DataFrame(
        all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    logger.info(f"✅ Total rows fetched for {symbol}: {len(df)}")
    return df


def save_data(df, symbol):
    pair = symbol.replace("/", "").lower()
    out_path = os.path.join(config["data_path"], "raw", f"{pair}_full.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"💾 Saved {symbol} -> {out_path}")


if __name__ == "__main__":
    for symbol in config["symbols"]:
        df = fetch_market_data(symbol, config["timeframe"])
        save_data(df, symbol)
