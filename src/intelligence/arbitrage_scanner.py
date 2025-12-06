import numpy as np
import pandas as pd
import logging
from src.intelligence.kalman_smoother import KalmanSmoother

logger = logging.getLogger("arb_scanner")

class ArbitrageScanner:
    """
    Scans for Statistical Arbitrage opportunities:
    1. Pairs Trading (BTC-ETH Spread) via Kalman Filter.
    2. Cross-Exchange Arbitrage (Binance vs Bybit/OKX).
    """
    def __init__(self):
        # Kalman for Spread Beta
        # State: [Beta, Intercept]
        # simplified 1D Kalman for Spread Z-Score for now
        self.spread_kalman = KalmanSmoother(process_noise=1e-5, measurement_noise=1e-3)
        self.exchange_premiums = {} # {exchange: premium}
        
    def calculate_spread_metrics(self, btc_price: float, eth_price: float) -> dict:
        """
        Calculates BTC-ETH Spread Z-Score.
        Spread = Log(BTC) - Beta * Log(ETH)
        For simplicity in this localized scaler, we track Log(ETH/BTC) ratio.
        """
        if btc_price <= 0 or eth_price <= 0:
            return {"spread_z": 0.0, "signal": 0}
            
        # Log Ratio
        ratio = np.log(btc_price / eth_price)
        
        # Smoothed Ratio (Expected Value)
        expected_ratio = self.spread_kalman.smooth(ratio)
        
        # Z-Score (Deviation from Expected)
        # Using a rolling std dev logic or fixed assumption
        # Here we approximation: Ratio - Expected
        deviation = ratio - expected_ratio
        
        # Normalize dev by recent volatility approximation (e.g. 0.002)
        z_score = deviation / 0.002 
        
        return {
            "ratio": ratio,
            "expected_ratio": expected_ratio,
            "spread_z": z_score
        }

    def update_exchange_premiums(self, base_price: float, other_prices: dict):
        """
        Update premiums for other exchanges relative to Base (Binance).
        """
        for ex, price in other_prices.items():
            if price > 0:
                diff_pct = (price - base_price) / base_price
                self.exchange_premiums[ex] = diff_pct
                
    def get_arb_factors(self) -> dict:
        """
        Returns {
            "spread_factor": 0.0-1.0 (Strength of Mean Reversion),
            "hedge_factor": 0.0-1.0 (Need to hedge),
            "direction": 1 (Long Ratio) / -1 (Short Ratio)
        }
        """
        # Logic: If Z-Score > 2.0 -> Short Ratio (Short BTC, Long ETH)
        # If Z-Score < -2.0 -> Long Ratio (Long BTC, Short ETH)
        
        # For simplicity, we store the last computed z_score in state?
        # In this stateless design, we'd need to pass it.
        # Let's assume this is called after calculate_spread_metrics in integration.
        return {} # Placeholder for integration logic

    def analyze(self, candle_data: pd.DataFrame) -> dict:
        """
        Full analysis from candle data.
        Expects 'btc_close' and 'eth_close'.
        """
        if "btc_close" not in candle_data or "eth_close" not in candle_data:
            return {"spread_score": 0.0, "action": "NONE"}
            
        btc = candle_data["btc_close"].iloc[-1]
        eth = candle_data["eth_close"].iloc[-1]
        
        metrics = self.calculate_spread_metrics(btc, eth)
        z = metrics["spread_z"]
        
        action = "NONE"
        score = 0.0
        
        if z > 2.0:
            action = "SHORT_RATIO" # Sell BTC, Buy ETH
            score = min(abs(z)/4.0, 1.0)
        elif z < -2.0:
            action = "LONG_RATIO" # Buy BTC, Sell ETH
            score = min(abs(z)/4.0, 1.0)
            
        logger.info(f"⚖️ Arb Scan: Z-Score {z:.2f} -> {action} ({score:.2f})")
        
        return {
            "spread_z": z,
            "arb_signal": action,
            "arb_confidence": score
        }
