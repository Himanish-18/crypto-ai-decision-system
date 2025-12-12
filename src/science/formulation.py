from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

"""
Scientific Formulation of the Prediction Problem
------------------------------------------------
This module defines the mathematical framework for the prediction problem,
including explicit modeling assumptions and custom loss functions for training.

Problem Definitions:
1. Feature Space (X): R^d
2. Target Space (Y): {0, 1} (Binary Direction) or R (Continuous Return)
3. Prediction Function (f): f(X) -> P(Y=1|X) or E[Y|X]

Assumptions:
- Stationarity: Valid within regime windows (Regime-Switching assumption).
- IID: Violated in time-series; requiring walk-forward validation and gap purging.
- Slippage: Modeled as a stochastic cost function C(s) ~ Gamma(k, theta).
"""


class MathematicalFormulation:
    """
    Formal definitions of the prediction problem.
    """

    @staticmethod
    def geometric_brownian_motion_assumption(
        S0: float, mu: float, sigma: float, T: float, dt: float
    ) -> np.ndarray:
        """
        Simulate price path under GBM assumption to test theoretical bounds.
        dS_t = mu * S_t * dt + sigma * S_t * dW_t
        """
        N = int(T / dt)
        t = np.linspace(0, T, N)
        W = np.random.standard_normal(size=N)
        W = np.cumsum(W) * np.sqrt(dt)  ### Standard Brownian Motion ###
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        S = S0 * np.exp(X)
        return S

    @staticmethod
    def calculate_information_coefficient(
        predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        """
        Calculate Information Coefficient (IC).
        IC = corr(predictions, targets)
        Fundamental Law of Active Management: IR = IC * sqrt(Breadth)
        """
        return np.corrcoef(predictions, targets)[0, 1]


class ScienceLossFunctions:
    """
    Custom loss functions derived from financial utility theory.
    """

    @staticmethod
    def sharpe_loss(
        y_true: np.ndarray, y_pred: np.ndarray, transaction_cost: float = 0.0
    ) -> float:
        """
        Differentiable Sharpe Ratio Loss (Negative Sharpe).

        L = - (E[R_p] / Std[R_p])
        Where R_p = y_pred * y_true - cost * abs(y_pred - y_prev)

        Simplified (Batch): Maximizes mean return / volatility.
        """
        # Assume y_pred is position size [-1, 1], y_true is return
        # Approximate Returns
        returns = y_pred * y_true

        # Penalize turnover (simplified)
        # Note: In a true batch loss, we need state. Here we assume independent samples approx.
        # Or cost is just a constant penalty on magnitude.

        adj_returns = returns - (transaction_cost * np.abs(y_pred))

        mean_ret = np.mean(adj_returns)
        std_ret = np.std(adj_returns) + 1e-6  # Stability

        sharpe = mean_ret / std_ret
        return -sharpe

    @staticmethod
    def sortino_loss(
        y_true: np.ndarray, y_pred: np.ndarray, target_return: float = 0.0
    ) -> float:
        """
        Sortino Ratio Loss.
        L = - (E[R_p - T] / DownsideDev[R_p])
        """
        returns = y_pred * y_true
        excess_returns = returns - target_return

        downside_returns = np.where(excess_returns < 0, excess_returns, 0)
        downside_dev = np.sqrt(np.mean(downside_returns**2)) + 1e-6

        mean_excess = np.mean(excess_returns)

        sortino = mean_excess / downside_dev
        return -sortino

    @staticmethod
    def direction_magnitude_loss(
        y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.5
    ) -> float:
        """
        Composite loss: MSE for Magnitude + BCE for Direction.
        Suitable for regression models predicting returns.

        L = alpha * MSE + (1 - alpha) * DirectionPenalty
        """
        # MSE
        mse = np.mean((y_true - y_pred) ** 2)

        # Direction Penalty (0 if signs match, linear otherwise? Or Log Loss?)
        # Let's use negative correlation proxy or sign mismatch
        sign_match = np.sign(y_true) == np.sign(y_pred)
        direction_loss = 1 - np.mean(sign_match)  # 0 to 1

        return alpha * mse + (1 - alpha) * direction_loss
