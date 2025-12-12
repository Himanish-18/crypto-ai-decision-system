import argparse
import logging
import sys

import numpy as np

from src.rl.portfolio_env import PortfolioEnv
from src.rl.ppo_portfolio import PPOPortfolioAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backtest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    assets = ["BTC", "ETH", "SOL"]
    env = PortfolioEnv(assets)
    agent = PPOPortfolioAgent(env.state_dim, env.action_dim)

    # Run Rollout
    obs = env.reset()
    total_reward = 0
    portfolio_values = [env.initial_balance]

    weights_history = []

    logger.info("🚀 Starting Portfolio Backtest...")

    for i in range(args.steps):
        action, _ = agent.select_action(obs)
        weights_history.append(action)

        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Track Value (simulated return in info)
        # info["return"] is net PnL amount?
        # My env implementation info return was: net_return (amount)
        current_val = portfolio_values[-1] + info["return"]
        portfolio_values.append(current_val)

        if i % 10 == 0:
            logger.info(
                f"Step {i}: Val={current_val:.2f}, Reward={reward:.4f}, Turnover={info['turnover']:.4f}"
            )

    logger.info("✅ Backtest Complete.")
    final_return = (portfolio_values[-1] - env.initial_balance) / env.initial_balance
    logger.info(f"Final Return: {final_return*100:.2f}%")
    logger.info(
        f"Avg Turnover: {np.mean([np.sum(np.abs(w - weights_history[max(0, i-1)])) for i, w in enumerate(weights_history)]):.4f}"
    )


if __name__ == "__main__":
    main()
