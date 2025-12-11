import numpy as np
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger("portfolio_env")

class PortfolioEnv:
    """
    v23 Portfolio RL Environment.
    Simulates portfolio allocation.
    Compatible interface with standard RL loops (reset, step).
    """
    def __init__(self, asset_names: List[str], initial_balance: float = 10000.0, transaction_cost_bps: float = 5.0):
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.initial_balance = initial_balance
        self.cost_bps = transaction_cost_bps
        self.state_dim = self.n_assets * 3 + 1 # Returns, Vol, Weights, time? No just simplest
        self.action_dim = self.n_assets + 1 # +1 for Cash
        
        self.current_weights = None
        self.current_step = 0
        self.data_len = 0
        self.market_data = None # TBD: Load or Simulate
        
    def reset(self) -> np.ndarray:
        self.current_weights = np.zeros(self.action_dim)
        self.current_weights[-1] = 1.0 # 100% Cash
        self.current_step = 0
        # Mock Observation
        return self._get_obs()
        
    def step(self, action_weights: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        action_weights: Softmax normalized vector of size n_assets + 1 (last is cash).
        """
        # 1. Normalize Action
        action_weights = np.clip(action_weights, 0, 1)
        action_weights /= (np.sum(action_weights) + 1e-8)
        
        # 2. Transaction Cost
        # Change in weights cost
        turnover = np.sum(np.abs(action_weights - self.current_weights))
        cost = turnover * (self.cost_bps / 10000.0)
        
        # 3. Simulation (Mock Return)
        # Assume generic drift + noise
        returns = np.random.normal(0.0005, 0.01, self.n_assets) # 5bps drift, 1% vol
        
        # Portfolio Return
        # Asset part
        asset_weights = action_weights[:-1]
        port_ret = np.sum(asset_weights * returns)
        
        # Cash return (0)
        
        net_return = port_ret - cost
        
        # 4. Reward Calculation (Sharpe-like)
        # Penalize Volatility (Risk)
        risk_penalty = np.std(returns) * np.sum(asset_weights) * 0.1 # Simple proxy
        reward = net_return - risk_penalty
        
        # 5. Update State
        self.current_weights = action_weights
        self.current_step += 1
        
        done = self.current_step > 100 # Mock episode length
        
        return self._get_obs(), reward, done, {"turnover": turnover, "cost": cost, "return": net_return}
        
    def _get_obs(self):
        # [AssetReturns(n), AssetVol(n), CurrentWeights(n+1)]
        # Faking it
        returns = np.random.normal(0, 0.01, self.n_assets)
        vols = np.random.uniform(0.01, 0.05, self.n_assets)
        
        return np.concatenate([returns, vols, self.current_weights])
