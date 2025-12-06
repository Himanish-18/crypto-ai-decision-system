import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger("portfolio_env")

class PortfolioEnv:
    """
    Multi-Asset Portfolio Environment for RL Training.
    State: [Market Features (BTC, ETH, Spread), Portfolio State (Balances), Regime]
    Action: Weights [BTC, ETH, CASH, SPREAD, HEDGE] (Sum=1.0)
    """
    def __init__(self, df_btc: pd.DataFrame, df_eth: pd.DataFrame, initial_balance=10000.0):
        self.df_btc = df_btc.reset_index(drop=True)
        self.df_eth = df_eth.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.done = False
        
        # Portfolio Weights: [BTC, ETH, CASH, SPREAD, HEDGE]
        # Initial: All Cash
        self.weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0]) 
        self.portfolio_value = initial_balance
        self.prev_portfolio_value = initial_balance
        
        # Costs
        self.trading_fee = 0.0006 # 0.06% Taker
        
    def reset(self):
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.prev_portfolio_value = self.initial_balance
        self.weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        self.current_step = 50 # Warmup
        self.done = False
        return self._get_observation()
        
    def _get_observation(self):
        # Construct Feature Vector
        # 1. Market Features (from DF)
        # Using simple numeric cols for now
        # Assuming DFs are aligned
        if self.current_step >= len(self.df_btc):
            return np.zeros(20) # End of episode padding
            
        btc_row = self.df_btc.iloc[self.current_step]
        eth_row = self.df_eth.iloc[self.current_step]
        
        # Mock features: [BTC_Ret, ETH_Ret, BTC_Vol, ETH_Vol, Spread]
        obs = np.array([
            btc_row.get("ret_1", 0),
            eth_row.get("ret_1", 0),
            btc_row.get("volatility", 0),
            eth_row.get("volatility", 0),
            # Current Weights
            *self.weights
        ], dtype=np.float32)
        
        # Pad to fixed size if needed (e.g. 20)
        padding = np.zeros(20 - len(obs))
        return np.concatenate([obs, padding])

    def step(self, action_weights: np.ndarray):
        """
        Action: Target Weights [BTC, ETH, CASH, SPREAD, HEDGE]
        """
        if self.done:
            return self._get_observation(), 0.0, True, {}
            
        # 1. Normalize Action (Softmax or Clip)
        # Assuming action comes from Softmax policy already? 
        # Or raw logits? Let's assume normalized for Env interface
        target_weights = np.clip(action_weights, 0.0, 1.0)
        target_weights /= (np.sum(target_weights) + 1e-8)
        
        # 2. Calculate Transaction Costs (Turnover)
        turnover = np.sum(np.abs(target_weights - self.weights))
        cost = turnover * self.portfolio_value * self.trading_fee
        
        # 3. Step Market (Price Change)
        current_btc_price = self.df_btc.iloc[self.current_step]["close"]
        next_btc_price = self.df_btc.iloc[self.current_step + 1]["close"]
        
        current_eth_price = self.df_eth.iloc[self.current_step]["close"]
        next_eth_price = self.df_eth.iloc[self.current_step + 1]["close"]
        
        btc_ret = (next_btc_price - current_btc_price) / current_btc_price
        eth_ret = (next_eth_price - current_eth_price) / current_eth_price
        cash_ret = 0.0
        
        # Spread Trade logic: Long BTC / Short ETH (Example)
        # Hedge logic: Short ETH (Example)
        
        # Asset Returns Vector
        # [BTC, ETH, CASH, SPREAD, HEDGE]
        # Spread = BTC_Ret - ETH_Ret (simplified)
        # Hedge = -1 * ETH_Ret (simplified short)
        asset_returns = np.array([
            btc_ret,
            eth_ret,
            cash_ret,
            btc_ret - eth_ret, 
            -1 * eth_ret 
        ])
        
        # 4. Update Portfolio
        weighted_return = np.sum(target_weights * asset_returns)
        
        new_value = self.portfolio_value * (1 + weighted_return) - cost
        
        # 5. Reward Calculation
        # Instant PnL
        reward = (new_value - self.prev_portfolio_value) / self.prev_portfolio_value
        
        # Update State
        self.portfolio_value = new_value
        self.prev_portfolio_value = new_value
        self.weights = target_weights
        self.current_step += 1
        
        if self.current_step >= len(self.df_btc) - 1:
            self.done = True
            
        return self._get_observation(), reward, self.done, {"val": self.portfolio_value}
