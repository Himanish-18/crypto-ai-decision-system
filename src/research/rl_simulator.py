import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from pathlib import Path
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rl_simulator")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"
OUTPUT_DIR = DATA_DIR / "research"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Environment ---
class CryptoEnv:
    def __init__(self, df: pd.DataFrame, initial_balance=10000):
        self.df = df
        self.initial_balance = initial_balance
        
        # Feature columns (excluding non-numeric)
        exclude = ["timestamp", "y_direction_up", "btc_ret_fwd_1"]
        self.available_features = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
        
        # L2 Features we want to simulate
        self.l2_features = ["ofi", "vpin", "spread"]
        
        # Total Observation Space
        self.observation_space = len(self.available_features) + len(self.l2_features) + 2 # +2 for Position State
        
        # Action Space:
        # 0: Hold
        # 1: Market Buy
        # 2: Market Sell
        # 3: Limit Buy (Best Bid)
        # 4: Limit Sell (Best Ask)
        self.action_space = 5 
        
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0 # 0: Flat, 1: Long
        self.entry_price = 0.0
        self.current_step = 0
        self.equity_curve = [self.initial_balance]
        self.pending_orders = [] # List of (type, price, step_created)
        return self._get_obs()

    def _get_obs(self):
        # Base Features (Available in DF)
        obs = self.df.iloc[self.current_step][self.available_features].values.astype(np.float32)
        
        # Mock L2 Features (since they are not in historical DF yet)
        # In a real scenario, we would load a DF that has these columns
        mock_l2 = np.random.normal(0, 1, len(self.l2_features)).astype(np.float32)
        
        # Combine
        obs = np.concatenate([obs, mock_l2])
            
        # Add State Features
        state_features = np.array([self.position, self.entry_price], dtype=np.float32)
        
        return torch.FloatTensor(np.concatenate([obs, state_features]))

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["btc_close"]
        
        # Simulate L2 Data (Spread)
        spread = 0.0005 * current_price # 5 bps spread
        best_bid = current_price - (spread / 2)
        best_ask = current_price + (spread / 2)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        if done:
            return self._get_obs(), 0, done, {}

        next_price = self.df.iloc[self.current_step]["btc_close"]
        price_change = (next_price - current_price) / current_price
        
        reward = 0
        
        # --- Handle Pending Orders ---
        # Simple fill logic: If price crosses limit price
        # Buy Limit: Low <= Limit Price
        # Sell Limit: High >= Limit Price
        # Here we only have Close, so we approximate with Low/High if available, else Close
        low = self.df.iloc[self.current_step].get("btc_low", next_price)
        high = self.df.iloc[self.current_step].get("btc_high", next_price)
        
        new_pending = []
        for order in self.pending_orders:
            otype, oprice, created = order
            filled = False
            
            if otype == "BUY_LIMIT":
                if low <= oprice: # Filled
                    if self.position == 0:
                        self.position = 1
                        self.entry_price = oprice
                        filled = True
            elif otype == "SELL_LIMIT":
                if high >= oprice: # Filled
                    if self.position == 1:
                        self.position = 0
                        filled = True
            
            if not filled and (self.current_step - created < 10): # Expire after 10 steps
                new_pending.append(order)
                
        self.pending_orders = new_pending

        # --- Handle New Action ---
        if action == 1: # Market Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = best_ask # Pay spread
                reward -= 0.0005 # Transaction cost
                
        elif action == 2: # Market Sell
            if self.position == 1:
                self.position = 0
                reward -= 0.0005 # Transaction cost
                # PnL realized
                pnl = (best_bid - self.entry_price) / self.entry_price
                reward += pnl * 100
                
        elif action == 3: # Limit Buy (at Best Bid)
            if self.position == 0:
                self.pending_orders.append(("BUY_LIMIT", best_bid, self.current_step))
                
        elif action == 4: # Limit Sell (at Best Ask)
            if self.position == 1:
                self.pending_orders.append(("SELL_LIMIT", best_ask, self.current_step))
        
        # --- Calculate Reward (Unrealized PnL for holding) ---
        if self.position == 1:
            step_pnl = (next_price - current_price) / current_price
            reward += step_pnl * 100
            self.balance *= (1 + step_pnl)
            
        self.equity_curve.append(self.balance)
        
        # Drawdown Penalty
        max_equity = max(self.equity_curve)
        drawdown = (max_equity - self.balance) / max_equity
        if drawdown > 0.1:
            reward -= 1
            
        return self._get_obs(), reward, done, {}

# --- DQN Agent ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, input_dim, output_dim):
        self.model = DQN(input_dim, output_dim)
        self.target_model = DQN(input_dim, output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.action_size = output_dim

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            
            target_f = self.model(state)
            target_f = target_f.clone().detach() # Avoid in-place op error
            target_f[action] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

def main():
    logger.info("🤖 Starting RL Simulation...")
    
    # Load Data
    if not FEATURES_FILE.exists():
        logger.error("Features file not found.")
        return
        
    df = pd.read_parquet(FEATURES_FILE)
    df = df.dropna().reset_index(drop=True)
    
    # Split Data (Train on first 80%, Test on last 20%)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    env = CryptoEnv(train_df)
    agent = Agent(env.observation_space, env.action_space)
    
    episodes = 5 # Short run for demo
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
            
        agent.update_target_model()
        logger.info(f"Episode {e+1}/{episodes} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}")
        
    # Test Run
    logger.info("🧪 Running Evaluation on Test Set...")
    env = CryptoEnv(test_df)
    state = env.reset()
    done = False
    
    while not done:
        # Pure exploitation
        with torch.no_grad():
            action = torch.argmax(agent.model(state)).item()
        state, _, done, _ = env.step(action)
        
    final_balance = env.balance
    logger.info(f"Final Balance (Test): ${final_balance:.2f}")
    
    # Save Results
    results = pd.DataFrame({"equity": env.equity_curve})
    results.to_csv(OUTPUT_DIR / "rl_results.csv", index=False)
    logger.info("✅ RL Simulation Complete.")

if __name__ == "__main__":
    main()
