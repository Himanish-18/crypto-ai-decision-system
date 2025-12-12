
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np

# v42 Meta-RL: Policy Network supporting Fast Adaptation (MAML)
# Standard Actor-Critic architecture with cloning capabilities for inner-loop updates.

class MetaActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Shared Feature Extractor
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor Head (Mean of diagonal Gaussian)
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim)) # Learnable log_std
        
        # Critic Head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.shared_net(obs)
        mean = self.actor(features)
        value = self.critic(features)
        return mean, value

    def get_action(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        mean, _ = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, obs, action):
        mean, value = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1)
        
        return value, log_prob, entropy

class MetaPolicy:
    def __init__(self, obs_dim, act_dim, lr=3e-4, inner_lr=0.01):
        self.model = MetaActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.inner_lr = inner_lr

    def clone(self):
        """Creates a deep copy for inner loop adaptation."""
        cloned = copy.deepcopy(self)
        # Re-init optimizer for the clone (usually SGD for inner loop)
        cloned.optimizer = optim.SGD(cloned.model.parameters(), lr=self.inner_lr)
        return cloned

    def adapt(self, loss):
        """Performs a single gradient step (Inner Loop)."""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, obs):
        return self.model.get_action(obs)

    def evaluate(self, obs, action):
        return self.model.evaluate(obs, action)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
