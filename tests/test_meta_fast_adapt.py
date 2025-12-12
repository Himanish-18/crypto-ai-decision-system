
import unittest
import torch
import numpy as np
from src.rl.policy_meta import MetaPolicy
from src.rl.adapt_api import AdaptationAPI

class TestMetaFastAdapt(unittest.TestCase):
    def setUp(self):
        self.obs_dim = 8
        self.act_dim = 2
        self.policy = MetaPolicy(self.obs_dim, self.act_dim, inner_lr=0.5) # High LR for test
        self.api = AdaptationAPI(self.policy)

    def test_adaptation_step(self):
        # Create dummy experience
        batch = []
        for i in range(10):
            obs = np.random.randn(self.obs_dim)
            act = np.random.randn(self.act_dim)
            rew = 1.0 if i < 5 else -1.0 # Variable reward
            val = 0.0
            logp = 0.0
            batch.append((obs, act, rew, val, logp))
            
        # Check weights before
        old_weights = self.api.current_policy.model.actor.weight.data.clone()
        
        # Adapt
        self.api.adapt(batch, k_steps=1)
        
        # Check weights after
        new_weights = self.api.current_policy.model.actor.weight.data
        
        # Weights should change
        self.assertFalse(torch.allclose(old_weights, new_weights), "Weights did not update after adaptation")

    def test_rollback(self):
        self.api.rollback()
        # Ensure we have a fresh clone
        self.assertIsNot(self.api.current_policy, self.api.base_policy)
