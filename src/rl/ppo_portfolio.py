import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger("ppo_agent")

class PPOPortfolioAgent(nn.Module):
    """
    v23 PPO Agent for Portfolio Allocation.
    Actor: State -> Allocation Weights (Softmax).
    Critic: State -> Value Estimate.
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99, eps_clip: float = 0.2):
        super(PPOPortfolioAgent, self).__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1) # Outputs valid weights sums to 1
        )
        
        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        
    def forward(self):
        raise NotImplementedError
        
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Returns action (weights) and log_prob (placeholder for deterministic/simple stochastic).
        For simplicity in Portfolio, we often use Deterministic Actor during deployment, 
        but Stochastic (Dirichlet) during training.
        Here we implement a simple Stochastic policy: adding noise to logits or using Dirichlet?
        Lets use basic Gaussian noise on logits before softmax? No, that breaks simplex.
        Simple approach: Actor outputs mean weights. We sample from Dirichlet(alpha=weights*concentration).
        But Dirichlet is tricky.
        
        Let's use a simpler Gaussian on logits, then Softmax.
        """
        state_t = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs = self.actor(state_t)
            
        # Exploration: Sample from Categorical? NO. Action is continuous vector.
        # We need a distribution over simplex. 
        # Deep Reinforcement Learning in Portfolio Management typically uses Deterministic Policy Gradient (DDPG).
        # But user asked for PPO.
        # PPO for continuous action usually outputs Mean and Std of Gaussian.
        # But we have constraint Sum(w)=1.
        # Solution: Output Gaussian Logits, then Softmax.
        # PPO works on the Logits.
        
        # simplified for this task:
        # Just return the weights (Deterministic) for deployment.
        # For training, we need gradients.
        
        return action_probs.numpy(), 0.0 # Placeholder log_prob
    
    def update(self, memory):
        # Stub for PPO update loop
        # Calculate Advantages
        # Optimize Policy Loss + Value Loss
        pass

    def save(self, path):
         torch.save(self.state_dict(), path)
         
    def load(self, path):
         self.load_state_dict(torch.load(path))
