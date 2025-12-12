import unittest

import numpy as np

from src.rl.portfolio_env import PortfolioEnv
from src.rl.ppo_portfolio import PPOPortfolioAgent


class TestPortfolioEnv(unittest.TestCase):
    def setUp(self):
        self.assets = ["BTC", "ETH", "SOL"]
        self.env = PortfolioEnv(
            self.assets, initial_balance=10000, transaction_cost_bps=5.0
        )
        self.agent = PPOPortfolioAgent(
            state_dim=self.env.state_dim, action_dim=self.env.action_dim
        )

    def test_dimensions(self):
        obs = self.env.reset()
        # n_assets=3. State: 3 avg returns + 3 vols + 4 weights = 10 dim.
        self.assertEqual(len(obs), 3 * 2 + 4)  # 10
        self.assertEqual(self.env.action_dim, 4)  # 3 assets + Cash

    def test_step_mechanics(self):
        self.env.reset()
        # Action: [0.25, 0.25, 0.25, 0.25]
        action = np.array([0.25, 0.25, 0.25, 0.25])
        obs, reward, done, info = self.env.step(action)

        self.assertIsInstance(reward, float)
        self.assertFalse(done)
        self.assertIn("turnover", info)

        # Test Turnover Cost
        # Previous weights (reset): [0, 0, 0, 1]
        # New weights: [0.25, 0.25, 0.25, 0.25]
        # Diff: |0.25| + |0.25| + |0.25| + |-0.75| = 1.5
        # Cost: 1.5 * 5bps = 7.5bps = 0.00075
        self.assertAlmostEqual(info["turnover"], 1.5)
        self.assertAlmostEqual(info["cost"], 1.5 * 0.0005)

    def test_agent_forward(self):
        obs = self.env.reset()
        action, _ = self.agent.select_action(obs)
        self.assertEqual(len(action), 4)
        self.assertAlmostEqual(np.sum(action), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
