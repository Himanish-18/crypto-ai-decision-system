
import torch
import numpy as np

# v42 Meta-RL: Adaptation Buffer
# Stores trajectories separated into Support Set (Adaptation) and Query Set (Evaluation).

class AdaptationBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.support_set = [] # List of transitions
        self.query_set = []
        self.mode = "support" # or "query"

    def set_mode(self, mode: str):
        assert mode in ["support", "query"]
        self.mode = mode

    def store(self, obs, act, rew, val, logp):
        transition = (obs, act, rew, val, logp)
        if self.mode == "support":
            self.support_set.append(transition)
        else:
            self.query_set.append(transition)

    def get_batch(self, mode="support"):
        data = self.support_set if mode == "support" else self.query_set
        
        obs_buf = torch.as_tensor([x[0] for x in data], dtype=torch.float32)
        act_buf = torch.as_tensor([x[1] for x in data], dtype=torch.float32)
        rew_buf = torch.as_tensor([x[2] for x in data], dtype=torch.float32)
        val_buf = torch.as_tensor([x[3] for x in data], dtype=torch.float32)
        logp_buf = torch.as_tensor([x[4] for x in data], dtype=torch.float32)
        
        # Simple Advantage Estimation (Reward - Value)
        # In full PPO we'd use GAE-Lambda, keeping it simple for MAML POC
        adv_buf = rew_buf - val_buf
        
        # Normalize Advantages
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
        
        return dict(obs=obs_buf, act=act_buf, ret=rew_buf, adv=adv_buf, logp=logp_buf)

    def clear(self):
        self.reset()
