import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger("ppo_agent")

class PortfolioActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PortfolioActorCritic, self).__init__()
        # Shared Backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
        # Actor Head (Action Logits)
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1) # Output valid weights directly
        )
        
        # Critic Head (Value)
        self.critic = nn.Linear(64, 1)
        
    def forward(self, x):
        feat = self.backbone(x)
        return self.actor(feat), self.critic(feat)

class PortfolioPPOAgent:
    """
    PPO Agent for Portfolio Allocation.
    Action: 5 Assets [BTC, ETH, CASH, SPREAD, HEDGE]
    """
    def __init__(self, input_dim=20, action_dim=5, lr=3e-4):
        self.model = PortfolioActorCritic(input_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.eps_clip = 0.2
        self.gamma = 0.99
        
    def get_action(self, state):
        """
        Returns: weights (np.array), log_prob (placeholder), value
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            weights, value = self.model(state_t)
            
        return weights.squeeze(0).numpy(), value.item()
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
