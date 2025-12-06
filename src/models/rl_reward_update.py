import numpy as np

def calculate_rl_reward(log_returns: np.ndarray, 
                       actions: np.ndarray, 
                       hedge_ratios: np.ndarray = None, 
                       risk_free_rate: float = 0.0) -> float:
    """
    Enhanced Reward Function for PPO Agent (v11).
    Optimizes for Sharpe Ratio and Hedge Success.
    
    Reward = TotalReturn + 0.5 * SharpeRatio + 0.2 * HedgeBonus - VolPenalty
    """
    if len(log_returns) < 2:
        return 0.0
        
    # 1. Total Cumulative Return
    # Actions are position sizes (0.0 to 1.0)
    strategy_returns = log_returns * actions
    total_return = np.sum(strategy_returns)
    
    # 2. Sharpe Ratio Component
    std_dev = np.std(strategy_returns)
    sharpe = 0.0
    if std_dev > 1e-6:
        sharpe = (np.mean(strategy_returns) - risk_free_rate) / std_dev
        
    # 3. Hedge Bonus (Simulated)
    # If we hedged (hedge_ratio > 0) during negative returns, we get a bonus.
    hedge_bonus = 0.0
    if hedge_ratios is not None:
        # Bonus if Hedge > 0 AND Market Return < 0
        # This teaches agent to hedge downside.
        successful_hedges = np.where((actions > 0) & (log_returns < 0) & (hedge_ratios > 0), 1, 0)
        hedge_bonus = np.sum(successful_hedges) * 0.01
        
    # 4. Volatility Penalty (Unhedged)
    # Penalize high variance if unhedged
    # penalty = std_dev * (1 - mean_hedge_ratio)
    
    final_reward = total_return + (0.1 * sharpe) + hedge_bonus
    
    return float(final_reward)
