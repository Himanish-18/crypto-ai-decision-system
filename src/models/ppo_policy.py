from typing import Any, Dict

import numpy as np
import pandas as pd


class PPOPolicy:
    """
    Continuous PPO RL Policy for Position Sizing.
    Action Space: [0.0, 1.5] (Leverage Scalar).
    """

    def __init__(self, model_path=None):
        self.model_path = model_path
        # Placeholder for loading actual RL weights

    def get_action(self, state: Dict[str, float]) -> float:
        """
        Inference step to get position size scalar.
        State expects:
          - prob: Model probability
          - volatility: Market vol
          - trend_depth: TD score
          - panic_score: PEM score
          - pnl_state: Current floating pnl (optional)
        """
        # --- Contextual Logic (Approximating Trained Agent Behavior) ---

        prob = state.get("prob", 0.5)
        vol = state.get("volatility", 0.01)
        td = state.get("trend_depth", 0.5)
        panic = state.get("panic_score", 0.0)

        # 1. Base Sizing (Confidence based)
        # Agent learns to bet more on high prob
        # Sigmoid mapping roughly: 0.5->0, 0.6->0.5, 0.8->1.0
        base_action = max(0, (prob - 0.5) * 4.0)

        # 2. Volatility Scaling (Risk Aversion)
        # PPO learns to reduce size in high vol to normalize variance
        # If vol > 2%, reduce size
        vol_scalar = np.clip(0.01 / (vol + 0.001), 0.2, 1.5)

        # 3. Trend Depth Bonus
        # Agent learns "Trend following" reward
        td_bonus = 1.0 + (max(0, td - 0.5) * 0.5)

        # 4. Panic Penalty
        # Agent heavily penalized for holding during crash
        # If panic > 0.7, force 0
        if panic > 0.7:
            panic_penalty = 0.0
        else:
            panic_penalty = 1.0 - np.clip(panic, 0, 1)

        # Combine
        raw_action = base_action * vol_scalar * td_bonus * panic_penalty

        # Clip to Action Space [0, 1.0] (User Cap: 1.0x for Pilot)
        final_action = np.clip(raw_action, 0.0, 1.0)

        return float(final_action)
