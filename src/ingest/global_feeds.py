import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Union

logger = logging.getLogger("global_feeds")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("⚠️ yfinance not installed. Using Simulation Mode for Macro Feeds.")

class GlobalMacroFeeds:
    """
    v24 Institutional Macro Data Provider.
    Fetches S&P 500, VIX, DXY, and Rates.
    Falls back to simulation if yfinance unavailable or connection fails.
    """
    def __init__(self, use_simulation: bool = False):
        self.use_simulation = use_simulation or not YFINANCE_AVAILABLE
        self.tickers = {
            "SP500": "^GSPC",
            "VIX": "^VIX",
            "DXY": "DX-Y.NYB",
            "US10Y": "^TNX"
        }
        
    def fetch_latest(self) -> Dict[str, float]:
        """
        Get latest prices for all macro assets.
        """
        data = {}
        for name, ticker in self.tickers.items():
            price = self._get_price(ticker, name)
            data[name] = price
            
        return data

    def fetch_history(self, days: int = 365) -> pd.DataFrame:
        """
        Get historical close prices aligned to daily frequency.
        Returns DataFrame with columns [SP500, VIX, DXY, US10Y].
        """
        if self.use_simulation:
            return self._d_simulate_history(days)
            
        try:
            tickers_list = list(self.tickers.values())
            df = yf.download(tickers_list, period=f"{days}d", interval="1d", progress=False)['Close']
            
            # Rename columns back to friendly names
            inv_map = {v: k for k, v in self.tickers.items()}
            df = df.rename(columns=inv_map)
            
            # Fill NaN
            df = df.fillna(method="ffill").fillna(method="bfill")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch macro history: {e}. Falling back to simulation.")
            return self._d_simulate_history(days)

    def _get_price(self, ticker: str, name: str) -> float:
        if self.use_simulation:
            return self._simulate_price(name)
            
        try:
            tick = yf.Ticker(ticker)
            # Fast fetch
            hist = tick.history(period="1d")
            if not hist.empty:
                return hist["Close"].iloc[-1]
            return self._simulate_price(name)
        except Exception:
            return self._simulate_price(name)

    def _simulate_price(self, name: str) -> float:
        # Reasonable defaults
        defaults = {
            "SP500": 5800.0,
            "VIX": 15.0,
            "DXY": 104.0,
            "US10Y": 4.2
        }
        base = defaults.get(name, 100.0)
        # Add random noise
        return base * (1 + np.random.normal(0, 0.005))

    def _d_simulate_history(self, days: int) -> pd.DataFrame:
        logger.info(f"🎰 Simulating {days} days of Macro Data...")
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
        
        # SP500: Drift + Vol
        sp500 = 5000 * np.cumprod(1 + np.random.normal(0.0003, 0.01, days))
        
        # VIX: Mean Reverting Ornstein-Uhlenbeck
        vix = np.zeros(days)
        vix[0] = 15.0
        for i in range(1, days):
            vix[i] = vix[i-1] + 0.1 * (15.0 - vix[i-1]) + np.random.normal(0, 1.0)
        vix = np.maximum(vix, 9.0) # Floor
        
        # DXY: Random Walk
        dxy = 100 * np.cumprod(1 + np.random.normal(0, 0.003, days))
        
        # US10Y: Random Walk
        us10y = 4.0 + np.cumsum(np.random.normal(0, 0.02, days))
        
        df = pd.DataFrame({
            "SP500": sp500,
            "VIX": vix,
            "DXY": dxy,
            "US10Y": us10y
        }, index=dates)
        
        return df
