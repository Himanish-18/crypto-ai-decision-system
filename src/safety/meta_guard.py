
import logging

# v42 Safety: Meta-Guard
# Monitors adapted policy performance and triggers rollback if it degrades.

class MetaGuard:
    def __init__(self, adapt_api):
        self.api = adapt_api
        self.logger = logging.getLogger("safety.meta")
        self.cumulative_reward = 0.0
        self.baseline_reward_threshold = -0.5 # Dummy threshold

    def check_health(self, step_reward):
        self.cumulative_reward += step_reward
        
        # Simple heuristic: If reward drops too fast, rollback
        if self.cumulative_reward < self.baseline_reward_threshold:
            self.logger.error(f"🚨 Performance Drift! Reward: {self.cumulative_reward}")
            self.api.rollback()
            self.cumulative_reward = 0.0 # Reset
            
    def reset(self):
        self.cumulative_reward = 0.0
