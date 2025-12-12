import unittest
import numpy as np
from collections import deque

# Inline definition to simplify testing without dependencies
class RiskWeightedMemory:
    """
    Risk-Weighted Replay Memory.
    filters samples based on Profit Factor and Model Confidence.
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.weights = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        profit_factor,
        confidence,
        log_prob,
    ):
        """
        Store transition only if it meets risk/reward criteria.
        Criteria: Profit Factor > 1.0 (Profitable) OR Confidence < 0.45 (High Uncertainty/Exploration)
        """
        if profit_factor > 1.0 or confidence < 0.45:
            self.buffer.append((state, action, reward, next_state, done, log_prob))

            # Calculate weight for sampling
            # Higher weight for High Profit OR High Uncertainty (Low Confidence)
            # Weight = PF + (1 - Confidence)
            w = max(profit_factor, 1.0) + (1.0 - confidence)
            self.weights.append(w)

    def sample(self, batch_size):
        """
        Weighted random sampling.
        """
        if len(self.buffer) < batch_size:
            return None

        # Normalize weights
        probs = np.array(self.weights)
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Unzip
        states, actions, rewards, next_states, dones, log_probs = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=bool),
            np.array(log_probs, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class TestPPOMemory(unittest.TestCase):
    def setUp(self):
        self.memory = RiskWeightedMemory(capacity=100)

    def test_selective_storage(self):
        """Verify only high quality or high uncertainty samples are stored."""
        self.memory.push(
            state=np.zeros(10),
            action=np.zeros(5),
            reward=0,
            next_state=np.zeros(10),
            done=False,
            profit_factor=0.9,
            confidence=0.8,
            log_prob=0.0,
        )
        self.assertEqual(len(self.memory), 0, "Boring sample should be ignored")
    
        self.memory.push(
            state=np.zeros(10),
            action=np.zeros(5),
            reward=1,
            next_state=np.zeros(10),
            done=False,
            profit_factor=1.2,
            confidence=0.8,
            log_prob=0.0,
        )
        self.assertEqual(len(self.memory), 1, "High PF sample should be stored")

        self.memory.push(
            state=np.zeros(10),
            action=np.zeros(5),
            reward=0,
            next_state=np.zeros(10),
            done=False,
            profit_factor=0.8,
            confidence=0.3,
            log_prob=0.0,
        )
        self.assertEqual(len(self.memory), 2, "Low Confidence sample should be stored")

    def test_weighted_sampling(self):
        """Verify that high weight items are sampled more often (statistically)."""
        self.memory.push(
            state=[1],
            action=[1],
            reward=1,
            next_state=[1],
            done=False,
            profit_factor=3.0,
            confidence=0.9,
            log_prob=0.0,
        )
        for _ in range(9):
            self.memory.push(
                state=[0],
                action=[0],
                reward=0,
                next_state=[0],
                done=False,
                profit_factor=1.01,
                confidence=0.9,
                log_prob=0.0,
            )
    
        counts = {1: 0, 0: 0}
        for _ in range(1000):
            batch = self.memory.sample(1)
            state_val = batch[0][0][0]
            counts[state_val] += 1

        self.assertGreater(
            counts[1], 150, "High weight item sampled < 15% (Should be ~23%)"
        )

if __name__ == "__main__":
    unittest.main()
