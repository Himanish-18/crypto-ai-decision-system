import numpy as np
import pandas as pd


class PortfolioBacktester:
    """
    Simulates Multi-Asset Portfolio Performance.
    """

    def __init__(self, check_slippage=True):
        self.check_slippage = check_slippage
        self.history = []

    def record_trade(self, timestamp, symbol, side, price, qty, fee_usd):
        self.history.append(
            {
                "ts": timestamp,
                "symbol": symbol,
                "side": side,
                "price": price,
                "qty": qty,
                "fee": fee_usd,
            }
        )

    def run_analytics(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()

        df = pd.DataFrame(self.history)
        df["value"] = df["price"] * df["qty"]

        # Calculate PnL per symbol
        # Simplified FIFO/AverageCost required for true PnL
        # Here we just return Trade Log with attribution

        return df

    def calculate_sharpe(self, returns: pd.Series) -> float:
        if len(returns) < 2:
            return 0.0
        return (
            returns.mean() / (returns.std() + 1e-9) * np.sqrt(252 * 24 * 60)
        )  # Annualized 1m
