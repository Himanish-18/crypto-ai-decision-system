import argparse
import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.compliance.compliance_manager import ComplianceManager
# v24 Imports
from src.ingest.global_feeds import GlobalMacroFeeds
from src.portfolio.hybrid_optimizer import HybridPortfolioOptimizer
from src.risk.pca_factor_model import PCAFactorModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inst_backtest")


def calculate_metrics(returns):
    cum_ret = (1 + returns).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    ann_ret = np.mean(returns) * 252
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-8)

    # Sortino
    downside = returns[returns < 0]
    sortino = ann_ret / (np.std(downside) * np.sqrt(252) + 1e-8)

    # Max DD
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak
    max_dd = dd.min()

    return {
        "total_return": total_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
    }


def main():
    logger.info("🏛 Starting v24 Institutional Backtest (3 Years)...")

    # 1. Initialize Components
    feeds = GlobalMacroFeeds(use_simulation=True)
    pca_model = PCAFactorModel(n_components=3)
    optimizer = HybridPortfolioOptimizer()
    compliance = ComplianceManager("data/compliance/backtest_log.csv")

    # 2. Data Fetch (3 Years)
    logger.info("Fetching Macro Data...")
    macro_df = feeds.fetch_history(days=365 * 3)

    # Simulate Crypto Data (correlated with Macro)
    logger.info("Simulating correlated Crypto Data...")
    dates = macro_df.index
    crypto_assets = ["BTC", "ETH", "SOL"]

    # Correlation matrix stub
    # BTC ~ 0.4 * SP500 + Noise
    sp500_ret = macro_df["SP500"].pct_change().fillna(0)

    crypto_returns = {}
    for asset in crypto_assets:
        noise = np.random.normal(0, 0.04, len(dates))  # High vol
        asset_ret = 1.5 * sp500_ret + noise  # Beta 1.5
        crypto_returns[asset] = asset_ret

    crypto_df = pd.DataFrame(crypto_returns, index=dates)

    # Combined Universe
    full_df = pd.concat([macro_df.pct_change(), crypto_df], axis=1).fillna(0)

    # 3. Simulation Loop
    portfolio_value = 1_000_000.0
    equity_curve = []
    current_weights = np.array(
        [1.0 / len(crypto_assets)] * len(crypto_assets)
    )  # Equal weight crypto

    logger.info("Running Simulation Loop...")
    # Rebalance weekly (every 5 days)
    rebalance_freq = 5

    for i in range(50, len(dates)):  # Warmup 50 days for PCA
        current_date = dates[i]

        # Returns for today
        day_rets = crypto_df.iloc[i].values
        port_ret = np.sum(current_weights * day_rets)
        portfolio_value *= 1 + port_ret

        equity_curve.append(portfolio_value)

        if i % rebalance_freq == 0:
            # 1. Fit PCA (Rolling Window)
            window = full_df.iloc[i - 50 : i]
            pca_model.fit(window)

            # 2. Optimize
            # Only optimize Crypto weights
            window_crypto = crypto_df.iloc[i - 50 : i]
            exp_ret = window_crypto.mean().values
            cov_mat = window_crypto.cov().values

            # Simulated RL signal (momentum)
            mom_signal = window_crypto.iloc[-1] > window_crypto.iloc[-10].mean()
            rl_weights = mom_signal.astype(float).values
            rl_weights /= np.sum(rl_weights) + 1e-8

            new_weights = optimizer.optimize(exp_ret, cov_mat, rl_weights, alpha=0.6)

            # 3. Compliance Check
            # Check notional of BTC position
            # Simply check if we pass basic valid checks (notional < 2x NAV)
            # Creating dummy trade dict for check
            trade_check = {
                "size": new_weights[0] * portfolio_value / 50000.0,  # BTC units approx
                "price": 50000.0,
                "symbol": "BTC",
                "portfolio_value": portfolio_value,
            }
            if compliance.check_compliance(trade_check):
                current_weights = new_weights
            else:
                logger.warning("Compliance Blocked Rebalance. Keeping old weights.")

    # 4. Results
    final_curve = pd.Series(equity_curve)
    metrics = calculate_metrics(final_curve.pct_change().dropna())

    logger.info("📊 Institutional Backtest Results 📊")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    logger.info(f"Final Portfolio Value: ${portfolio_value:,.2f}")


if __name__ == "__main__":
    main()
