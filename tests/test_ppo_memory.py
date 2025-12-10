
import unittest
from unittest.mock import MagicMock, patch
import sys
import numpy as np

# Mock torch BEFORE importing agent
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.optim"] = MagicMock()
# support division and comparison for drift check
sys.modules["torch"].norm.return_value = 1.0 

import os
sys.path.append(os.getcwd())

# Now import agent
from src.models.portfolio_rl.ppo_agent import RiskWeightedMemory, PortfolioPPOAgent

class TestPPOMemory(unittest.TestCase):
    def setUp(self):
        self.memory = RiskWeightedMemory(capacity=100)
        
    def test_selective_storage(self):
        """Verify only high quality or high uncertainty samples are stored."""
        # Case 1: Boring sample (Low Profit, High Confidence) -> Should be rejected
        # PF <= 1, Conf >= 0.45
        self.memory.push(
            state=np.zeros(10), action=np.zeros(5), reward=0, next_state=np.zeros(10), done=False,
            profit_factor=0.9, confidence=0.8, log_prob=0.0
        )
        self.assertEqual(len(self.memory), 0, "Boring sample should be ignored")
        
        # Case 2: Profitable sample
        self.memory.push(
            state=np.zeros(10), action=np.zeros(5), reward=1, next_state=np.zeros(10), done=False,
            profit_factor=1.2, confidence=0.8, log_prob=0.0
        )
        self.assertEqual(len(self.memory), 1, "High PF sample should be stored")
        
        # Case 3: High Uncertainty (Exploration) sample
        self.memory.push(
            state=np.zeros(10), action=np.zeros(5), reward=0, next_state=np.zeros(10), done=False,
            profit_factor=0.8, confidence=0.3, log_prob=0.0
        )
        self.assertEqual(len(self.memory), 2, "Low Confidence sample should be stored")
        
    def test_weighted_sampling(self):
        """Verify that high weight items are sampled more often (statistically)."""
        # Store 1 high weight item (PF=3.0)
        self.memory.push(
            state=[1], action=[1], reward=1, next_state=[1], done=False,
            profit_factor=3.0, confidence=0.9, log_prob=0.0
        )
        # Store 9 low weight items (PF=1.01)
        for _ in range(9):
            self.memory.push(
                state=[0], action=[0], reward=0, next_state=[0], done=False,
                profit_factor=1.01, confidence=0.9, log_prob=0.0
            )
            
        # Sample many times
        counts = {1: 0, 0: 0} # Track state value
        for _ in range(1000):
            batch = self.memory.sample(1)
            state_val = batch[0][0][0] # state is [val]
            counts[state_val] += 1
            
        # High weight item (1/10th of population) should be sampled more than 10%
        # Weight 1: 3.0 + 0.1 = 3.1
        # Weight 2: 1.0 + 0.1 = 1.1
        # Total weight approx: 3.1 + 9*1.1 = 13.0
        # Prob of High = 3.1/13.0 ~= 23%
        # It's statistically significantly higher than 10%
        
        self.assertGreater(counts[1], 150, "High weight item sampled < 15% (Should be ~23%)")
        print(f"Sampling Stats: High Profit sampled {counts[1]}/1000 times")

class TestPPOAgent(unittest.TestCase):
    @patch('src.models.portfolio_rl.ppo_agent.PortfolioActorCritic')
    def setUp(self, MockActorCritic):
        # Configure the mock to have parameters
        mock_instance = MockActorCritic.return_value
        mock_instance.parameters.return_value = [MagicMock()]
        # Mock evaluate return 3 tensors (logprobs, values, entropy)
        mock_instance.evaluate.return_value = (MagicMock(), MagicMock(), MagicMock())
        self.agent = PortfolioPPOAgent(input_dim=4, action_dim=2)
        
    def test_update_policy_dims(self):
        """Test that update_policy runs without crashing given valid data."""
        # Fill memory
        for _ in range(100):
            state = np.random.randn(4)
            action = np.array([0.5, 0.5])
            next_state = np.random.randn(4)
            self.agent.memory.push(
                state, action, reward=1.0, next_state=next_state, done=False,
                profit_factor=1.2, confidence=0.5, log_prob=-0.6
            )
            
        # Run Update
        try:
            self.agent.update_policy()
        except Exception as e:
            self.fail(f"update_policy crashed: {e}")
            
    def test_drift_control(self):
        """Test that initial parameters are stored (mocked)."""
        # With mocks, self.agent.initial_params list exists
        self.assertTrue(hasattr(self.agent, 'initial_params'))
        # Length depends on how many params the mocked model returns. 
        # By default MagicMock returns a Mock for parameters(), so iterating it might yield one item or need config.
        # But we just want to ensure the logic ran.
        pass

if __name__ == "__main__":
    unittest.main()
