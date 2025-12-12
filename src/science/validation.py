import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("science_validation")


class StatisticalValidator:
    """
    Suite of statistical tests to validate model superiority over random chance.
    """

    @staticmethod
    def bootstrap_confidence_interval(
        data: np.ndarray,
        metric_func: callable,
        n_bootstraps: int = 1000,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """
        Calculate (1-alpha) Confidence Interval using Bootstrap sampling.
        """
        bootstrapped_metrics = []
        n = len(data)

        for _ in range(n_bootstraps):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrapped_metrics.append(metric_func(sample))

        lower = np.percentile(bootstrapped_metrics, 100 * (alpha / 2))
        upper = np.percentile(bootstrapped_metrics, 100 * (1 - alpha / 2))

        return lower, upper

    @staticmethod
    def diebold_mariano_test(
        real_values: np.ndarray,
        pred_1: np.ndarray,
        pred_2: np.ndarray,
        h: int = 1,
        loss_func: str = "MSE",
    ) -> Tuple[float, float]:
        """
        Diebold-Mariano test for predictive accuracy comparison.
        H0: Model 1 and Model 2 have equal predictive accuracy.

        Args:
            h: Forecast horizon (h>1 requires adjustment)
            loss_func: 'MSE' or 'MAE'

        Returns:
            DM Statistic, p-value
        """
        T = len(real_values)

        if loss_func == "MSE":
            e1 = (real_values - pred_1) ** 2
            e2 = (real_values - pred_2) ** 2
        else:  # MAE
            e1 = np.abs(real_values - pred_1)
            e2 = np.abs(real_values - pred_2)

        d = e1 - e2
        mean_d = np.mean(d)

        # Autocovariance for h > 1
        gamma0 = np.var(d)
        gamma = gamma0

        if h > 1:
            for lag in range(1, h):
                cov = np.cov(d[lag:], d[:-lag])[0, 1]
                gamma += 2 * cov

        dm_stat = mean_d / np.sqrt((gamma / T))
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        return dm_stat, p_value

    @staticmethod
    def monte_carlo_permutation_test(
        returns_strategy: np.ndarray, n_permutations: int = 1000
    ) -> float:
        """
        Test if strategy returns are significantly different from random shuffle.
        H0: Strategy sequence is random (no time-dependence/alpha).

        Returns:
            p-value (prob that random shuffle > strategy return)
        """
        true_metric = np.sum(returns_strategy)  # Or Sharpe
        count_better = 0

        dummy = returns_strategy.copy()

        for _ in range(n_permutations):
            np.random.shuffle(dummy)
            perm_metric = np.sum(dummy)
            if perm_metric >= true_metric:
                count_better += 1

        p_value = (count_better + 1) / (n_permutations + 1)
        return p_value

    @staticmethod
    def white_reality_check(
        strategies_returns: pd.DataFrame, benchmark_returns: np.ndarray
    ):
        """
        White's Reality Check for Data Snooping.
        (Simplified implementation placeholder)
        """
        pass


class ValidationReport:
    """
    Generates a scientific report of the model.
    """

    def generate_report(self, model_returns: np.ndarray, benchmark_returns: np.ndarray):
        validator = StatisticalValidator()

        # 1. Bootstrap Sharpe
        sharpe_func = lambda x: (np.mean(x) / np.std(x)) * np.sqrt(252 * 24)  # Hourly
        ci_lower, ci_upper = validator.bootstrap_confidence_interval(
            model_returns, sharpe_func
        )

        # 2. Monte Carlo
        p_val_mc = validator.monte_carlo_permutation_test(model_returns)

        # 3. T-Test vs Benchmark
        t_stat, p_val_t = stats.ttest_ind(model_returns, benchmark_returns)

        report = {
            "bootstrap_sharpe_95_ci": (ci_lower, ci_upper),
            "monte_carlo_p_value": p_val_mc,
            "ttest_vs_benchmark_p_value": p_val_t,
        }
        return report
