import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.market_impact import MarketImpactModel
from src.backtest.slippage_model_v2 import SlippageModelV2
from src.models.hybrid.dqn_mini import DQNMiniProxy

# Setup Standard Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf_backtester")


@dataclass
class HFBacktestConfig:
    initial_capital: float = 1_000_000.0  # Institutional size
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage_model: str = "adaptive"  # fixed, percentage, adaptive (spread-based)
    base_slippage: float = 0.0005  # 5 bps
    spread_factor: float = 0.5  # Fraction of spread paid
    risk_free_rate: float = 0.02  # Annualized
    regime_aware: bool = True
    liquidity_filter: float = 1_000_000.0  # Min daily volume to trade


class PerformanceMetrics:
    """
    Institutional / Hedge Fund Grade Metrics.
    """

    @staticmethod
    def calculate_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
        """Conditional Value at Risk (Expected Shortfall) at 95%."""
        if returns.empty:
            return 0.0
        var = returns.quantile(alpha)
        cvar = returns[returns <= var].mean()
        return abs(cvar)  # Return as positive loss magnitude

    @staticmethod
    def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """Omega Ratio: Probability weighted ratio of gains vs losses."""
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        if losses == 0:
            return float("inf")
        return gains / losses

    @staticmethod
    def calculate_calmar_ratio(cagr: float, max_dd: float) -> float:
        """Calmar Ratio: CAGR / Max Drawdown."""
        if max_dd == 0:
            return float("inf")
        return cagr / abs(max_dd)

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series, risk_free_rate: float = 0.0
    ) -> float:
        """Sortino Ratio: Excess Return / Downside Deviation."""
        excess_returns = returns - (
            risk_free_rate / (252 * 24)
        )  # Adjust rfr to hourly?
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0
        return (excess_returns.mean() / downside_std) * np.sqrt(252 * 24)

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series, risk_free_rate: float = 0.0
    ) -> float:
        excess_returns = returns - (risk_free_rate / (252 * 24))
        std = excess_returns.std()
        if std == 0:
            return 0.0
        return (excess_returns.mean() / std) * np.sqrt(252 * 24)


class SlippageModel:
    """
    Advanced Slippage Models.
    """

    def __init__(self, config: HFBacktestConfig):
        self.config = config

    def get_slippage(
        self,
        volatility: float,
        price: float = 10000.0,
        volume_profile: float = 100000.0,
        size: float = 1.0,
    ) -> float:
        """Calculate slippage based on config and volatility using V2 Models."""

        # Default legacy fallback
        legacy_slippage = self.config.base_slippage

        try:
            # Use V2 Spread Model
            spread_slip = SlippageModelV2.spread_based_slippage(
                price, spread_bps=5.0
            )  # Fixed spread assumption

            # Use V2 Volume Model if Size provided
            vol_slip = SlippageModelV2.volume_weighted_slippage(
                price, size, volume_profile, volatility
            )

            # Use Market Impact
            impact_engine = MarketImpactModel()
            im = impact_engine.estimate_impact(price, volatility, size, volume_profile)

            # Aggregated percentage slippage
            total_slip_val = max(spread_slip, vol_slip) + im
            return total_slip_val / price

        except Exception:
            return legacy_slippage

        if self.config.slippage_model == "dynamic_atr":
            scale = max(1.0, volatility / 0.005)
            return self.config.base_slippage * scale
        elif self.config.slippage_model == "adaptive":
            # Adaptive to volatility and inferred spread
            # Simple model: Spread ~ Volatility * Constant
            # Cost = Half Spread + Market Impact

            inferred_spread_bps = max(0.0002, volatility * 0.5)
            impact_bps = 0.0
            if volume_profile > 0:
                # Square Root Law of Market Impact
                # Impact ~ sigma * sqrt(OrderSize / DailyVolume)
                # Simplified placeholder
                impact_bps = 0.0001

            total_slippage_bps = (
                inferred_spread_bps * self.config.spread_factor
            ) + impact_bps
            return price * total_slippage_bps

        return 0.0


class HedgeFundBacktester:
    """
    Institutional Grade Backtester.
    """

    def __init__(self, data: pd.DataFrame, signal_col: str, config: HFBacktestConfig):
        self.data = data.copy()
        self.signal_col = signal_col
        self.config = config
        self.slippage_engine = SlippageModel(config)
        self.metrics = {}

    def run(self):
        logger.info("🚀 Starting Hedge Fund Grade Backtest...")

        capital = self.config.initial_capital
        position = 0.0  # Size in Base Asset
        cash = capital

        equity_curve = []
        trades = []

        # Pre-calculate volatility for slippage
        # Check if 'volatility' or 'ATR' exists, else calculate
        if "volatility" not in self.data.columns:
            self.data["volatility"] = (
                self.data["close"].pct_change().rolling(24).std().fillna(0.01)
            )

        # Iterating (Vectorization possible but Loop allows complex logic)
        for idx, row in self.data.iterrows():
            curr_price = row["close"]
            signal = row.get(self.signal_col, 0)
            vol = row["volatility"]

            # Mark to Market
            current_val = cash + (position * curr_price)
            equity_curve.append(current_val)

            # Logic: Simple Long/Flat for now (extendable to Short)
            target_pos = 0.0
            if signal == 1:
                target_pos = (current_val * 0.99) / curr_price  # Full alloc with buffer
            elif signal == -1:
                # Not supporting short in pilot, assume Flat
                target_pos = 0.0
            else:
                target_pos = 0.0  # Flat

            # Delta
            delta_units = target_pos - position

            if abs(delta_units * curr_price) > 100:  # Threshold
                # Execute
                slippage_cost = self.slippage_engine.estimate_slippage(
                    curr_price, volatility=vol
                )
                exec_price = (
                    curr_price + slippage_cost
                    if delta_units > 0
                    else curr_price - slippage_cost
                )

                fee = abs(delta_units * exec_price) * self.config.taker_fee

                trans_cost = (abs(delta_units) * slippage_cost) + fee

                cash -= (
                    delta_units * exec_price
                ) + fee  # Subtract fee from cash? Or implicit?
                # Usually:
                # Buy: Spend Cash for Units + Fee
                # Sell: Receive Cash for Units - Fee

                # Careful with cash logic:
                # Buy 1 BTC at 50000. Fee 20. Cost 50020.
                if delta_units > 0:  # Buy
                    cost = delta_units * exec_price
                    if cash >= cost + fee:
                        cash -= cost + fee
                        position += delta_units
                    else:
                        # Adjust size
                        max_cost = cash
                        # units * price * (1 + fee_rate) = max_cost
                        # units = max_cost / (price * (1 + fee))
                        # Ignore for simplicity or log warning
                        pass
                else:  # Sell
                    proceeds = abs(delta_units * exec_price)
                    cash += proceeds - fee
                    position += delta_units

                # --- COST AWARE SIGNAL LOGIC ---
                # Extract common vars

                # [WIRING PATCH] Instantiate Advanced Models
                # Ideally done in __init__ but doing here for minimal diff without full refactor
                if not hasattr(self, "slippage_v2"):
                    self.slippage_v2 = SlippageModelV2()
                    self.impact_model = MarketImpactModel()

                regime = row.get("regime", "Normal")
                trades.append(
                    {
                        "timestamp": row["timestamp"] if "timestamp" in row else idx,
                        "price": curr_price,
                        "exec_price": exec_price,
                        "units": delta_units,
                        "cost": trans_cost,
                        "regime": row.get("regime", "Unknown"),
                    }
                )

        self.results = pd.DataFrame(trades)
        self.equity_curve = pd.Series(equity_curve, index=self.data.index)

        self.calculate_metrics()

    def calculate_metrics(self):
        """Compute all institutional metrics."""
        returns = self.equity_curve.pct_change().dropna()

        total_ret = (
            self.equity_curve.iloc[-1] - self.config.initial_capital
        ) / self.config.initial_capital

        # Max DD
        roll_max = self.equity_curve.cummax()
        dd = (self.equity_curve - roll_max) / roll_max
        max_dd = dd.min()

        # Ratios
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(
            returns, self.config.risk_free_rate
        )
        sortino = PerformanceMetrics.calculate_sortino_ratio(
            returns, self.config.risk_free_rate
        )
        cvar_95 = PerformanceMetrics.calculate_cvar(returns)
        omega = PerformanceMetrics.calculate_omega_ratio(returns)

        # Turnover
        # turnover = Value Traded / Average Cap

        self.metrics = {
            "Total Return": total_ret,
            "Max Drawdown": max_dd,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "cVaR 95%": cvar_95,
            "Omega Ratio": omega,
        }

    def generate_report(self):
        print("\n=== Hedge Fund Grade Backtest Report ===")
        for k, v in self.metrics.items():
            print(f"{k}: {v:.4f}")

        if self.config.regime_aware and "regime" in self.results.columns:
            print("\n--- Regime Analysis ---")
            # Need to link trades back to regimes or analyse equity curve segments?
            # Analyzing equity curve by regime (requires regime col in data aligned with equity)
            if "regime" in self.data.columns:
                self.data["returns"] = self.equity_curve.pct_change()
                regime_stats = self.data.groupby("regime")["returns"].mean() * 24 * 365
                print(regime_stats)


if __name__ == "__main__":
    # Test Run
    # Dummy Data
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="H")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "close": 100 + np.cumsum(np.random.randn(1000)),
            "signal": np.random.choice([0, 1, -1], size=1000),
            "regime": np.random.choice(["Bull", "Bear", "Chop"], size=1000),
        }
    )

    config = HFBacktestConfig()
    bt = HedgeFundBacktester(df, "signal", config)
    bt.run()
    bt.generate_report()
