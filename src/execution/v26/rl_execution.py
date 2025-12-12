
import numpy as np
import random
from typing import Dict

class RLExecutionAgent:
    """
    Q-Learning Agent to decide Order Type (Limit vs Market).
    State: [Urgency(0-2), SpreadRegime(0-2)]
    Actions: [0: Limit, 1: Market]
    """
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((3, 3, 2)) # State Space (3x3), Action Space (2)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def get_action(self, urgency: int, spread_regime: int) -> str:
        """
        urgency: 0 (Low), 1 (Med), 2 (High)
        spread_regime: 0 (Tight), 1 (Normal), 2 (Wide)
        """
        if random.uniform(0, 1) < self.epsilon:
            return "LIMIT" if random.randint(0, 1) == 0 else "MARKET"
            
        action_idx = np.argmax(self.q_table[urgency, spread_regime])
        return "LIMIT" if action_idx == 0 else "MARKET"
    
    def update(self, urgency: int, spread_regime: int, action: str, reward: float, next_urgency: int, next_spread: int):
        action_idx = 0 if action == "LIMIT" else 1
        
        current_q = self.q_table[urgency, spread_regime, action_idx]
        max_next_q = np.max(self.q_table[next_urgency, next_spread])
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[urgency, spread_regime, action_idx] = new_q
