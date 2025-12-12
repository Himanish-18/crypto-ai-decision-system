from typing import Any, Dict, List

import numpy as np
import pandas as pd


class PerformanceAnalytics:
    """
    Centralized Performance Metrics Calculation.
    """

    @staticmethod
    def calculate_metrics(
        equity_curve: pd.Series, risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        returns = equity_curve.pct_change().dropna()
        if returns.empty:
            return {}

        # Cumulative Return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Sharpe
        excess_returns = returns - (risk_free_rate / 252)
        sharpe = (
            (excess_returns.mean() / returns.std()) * np.sqrt(252)
            if returns.std() > 0
            else 0
        )

        # Sortino
        downside = returns[returns < 0]
        sortino = (
            (excess_returns.mean() / downside.std()) * np.sqrt(252)
            if len(downside) > 0 and downside.std() > 0
            else 0
        )

        # Max Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()

        # Calmar
        cagr = ((1 + total_return) ** (252 / len(returns))) - 1
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # cVaR (95%)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "cvar_95": cvar_95,
        }

    @staticmethod
    def generate_win_loss_matrix(trades: pd.DataFrame) -> Dict[str, float]:
        """
        Win/Loss Analysis.
        """
        if trades.empty:
            return {}

        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]

        win_rate = len(wins) / len(trades)
        loss_rate = 1 - win_rate

        avg_win = wins["pnl"].mean() if not wins.empty else 0
        avg_loss = losses["pnl"].mean() if not losses.empty else 0

        return {
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": (win_rate * avg_win) + (loss_rate * avg_loss),
            "profit_factor": (
                abs(wins["pnl"].sum() / losses["pnl"].sum())
                if losses["pnl"].sum() != 0
                else float("inf")
            ),
        }
