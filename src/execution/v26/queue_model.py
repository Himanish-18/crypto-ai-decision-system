
import numpy as np
from types import SimpleNamespace

class QueuePositionModel:
    """
    Estimates probability of fill for a Limit Order based on Queue Position.
    Model: P(Fill) = exp(-k * QueueDepth / VolumeRate)
    """
    
    def __init__(self, decay_rate: float = 0.5):
        self.k = decay_rate
        
    def estimate_fill_probability(self, queue_depth_qty: float, volume_rate_per_sec: float, time_horizon_sec: float) -> float:
        """
        Estimate prob of fill within time horizon.
        
        Args:
            queue_depth_qty: Amount of liquidity ahead of us in the book.
            volume_rate_per_sec: Average matching engine turnover per second.
            time_horizon_sec: How long we are willing to wait.
        """
        if volume_rate_per_sec <= 0: return 0.0
        
        # Expected time to eat through queue
        expected_wait_time = queue_depth_qty / volume_rate_per_sec
        
        # If we wait longer than expected time, prob approaches 1
        # Simple Poisson-like decay for waiting time
        # Prob that arrival time < horizon
        # P(T < t) = 1 - exp(-lambda * t) where lambda = 1/expected_wait
        
        if expected_wait_time == 0: return 1.0
        
        lam = 1.0 / expected_wait_time
        prob = 1.0 - np.exp(-lam * time_horizon_sec)
        
        return min(max(prob, 0.0), 1.0)
