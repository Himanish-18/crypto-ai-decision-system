import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger("hybrid_optimizer")


class HybridPortfolioOptimizer:
    """
    v24 Institutional Hybrid Optimizer.
    Blends Markowitz Mean-Variance Optimization (MVO) with Deep RL (PPO) Allocation.
    w_final = alpha * w_mvo + (1 - alpha) * w_rl
    """

    def __init__(self, risk_free_rate: float = 0.04):
        self.rf = risk_free_rate

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        ppo_weights: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Calculate Optimal Weights.
        expected_returns: Vector of E[R].
        cov_matrix: Covariance Matrix.
        ppo_weights: Weights suggested by RL Agent.
        alpha: Weight given to Classical MVO (0.0 to 1.0).
        """
        n_assets = len(expected_returns)

        # 1. Classical MVO (Max Sharpe)
        # We solve for weights that Maximize: (w.T * R - rf) / sqrt(w.T * Cov * w)
        # Or Minimize Negative Sharpe.

        def neg_sharpe(weights):
            p_ret = np.sum(weights * expected_returns)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if p_vol == 0:
                return 0
            return -(p_ret - self.rf) / p_vol

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1.0 / n_assets] * n_assets)

        try:
            result = minimize(
                neg_sharpe,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if result.success:
                w_mvo = result.x
            else:
                logger.warning(
                    f"MVO Solver Failed: {result.message}. Using Equal Weight fallback."
                )
                w_mvo = initial_guess
        except Exception as e:
            logger.error(f"Optimizer Crash: {e}")
            w_mvo = initial_guess

        # 2. Hybrid Blending
        # Ensure ppo_weights sum to 1
        ppo_weights = ppo_weights / (np.sum(ppo_weights) + 1e-8)

        # Blend
        w_final = alpha * w_mvo + (1 - alpha) * ppo_weights

        # Re-Normalize (just in case)
        w_final = np.clip(w_final, 0, 1)
        w_final = w_final / np.sum(w_final)

        return w_final

    def calculate_metrics(self, weights: np.ndarray, ret: np.ndarray, cov: np.ndarray):
        p_ret = np.sum(weights * ret)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        sharpe = (p_ret - self.rf) / (p_vol + 1e-8)
        return {"expected_return": p_ret, "volatility": p_vol, "sharpe": sharpe}
