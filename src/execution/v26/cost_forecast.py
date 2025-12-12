
import numpy as np
from sklearn.linear_model import BayesianRidge
from typing import List, Tuple

class ExecutionCostForecaster:
    """
    Predicts slippage/execution cost using Bayesian Regression.
    Features: [Volatility, Spread, TradeSizePctOfVol]
    """
    
    def __init__(self):
        self.model = BayesianRidge()
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        X: [[volatility, spread, size_ratio], ...]
        y: [realized_slippage_bps, ...]
        """
        self.model.fit(X, y)
        self.is_trained = True
        
    def predict_slippage(self, volatility: float, spread: float, size_ratio: float) -> Tuple[float, float]:
        """
        Returns (Prediction, StdDev) i.e., uncertainty.
        """
        if not self.is_trained:
            # Fallback heuristic
            return (spread * 0.5 + size_ratio * 10.0, 5.0)
            
        return self.model.predict([[volatility, spread, size_ratio]], return_std=True)
