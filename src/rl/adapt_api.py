
import torch
import copy
import logging
from src.rl.policy_meta import MetaPolicy
from src.rl.buffer_adapt import AdaptationBuffer

# v42 Meta-RL: Runtime Adaptation API
# Allows the live trading system to adapt the policy to recent market data in real-time.

class AdaptationAPI:
    def __init__(self, base_policy: MetaPolicy):
        self.base_policy = base_policy
        self.current_policy = base_policy.clone()
        self.buffer = AdaptationBuffer()
        self.logger = logging.getLogger("rl.adapt")

    def adapt(self, recent_experience_batch, k_steps=1):
        """
        Adapts the policy to the provided batch of experience.
        experience_batch: dict/list of (obs, act, rew, val, logp)
        """
        self.buffer.reset()
        self.buffer.set_mode("support")
        
        # Load data into buffer
        for item in recent_experience_batch:
            self.buffer.store(*item)
            
        # Get Batch
        data = self.buffer.get_batch("support")
        if len(data['obs']) == 0:
            return
            
        # Perform K gradient steps
        # In MAML-PPO, we adapt the *Current* policy (which might already be adapted)
        # Or resetting to Base and adapting K steps. MAML usually resets to Base.
        
        # Reset to Base for pure MAML inference
        self.current_policy = self.base_policy.clone()
        
        for _ in range(k_steps):
            _, logp_new, _ = self.current_policy.evaluate(data['obs'], data['act'])
            # Simplified Loss (Policy Gradient)
            loss = -(logp_new.squeeze() * data['adv']).mean()
            self.current_policy.adapt(loss)
            
        self.logger.info(f"✅ Adapted policy on {len(data['obs'])} samples for {k_steps} steps.")

    def get_action(self, obs):
        return self.current_policy.model.get_action(obs)

    def rollback(self):
        """Revert to safe base policy."""
        self.current_policy = self.base_policy.clone()
        self.logger.warning("↺ Rolled back to Base Policy.")
