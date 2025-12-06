import numpy as np

class FillProbabilityModel:
    """
    Estimates the probability of a Limit Order being filled within N seconds.
    Based on:
    1. Distance from Mid
    2. Order Book Imbalance
    3. Recent Volatility
    4. Queue Position (Simulated via Depth Volume)
    """
    def __init__(self):
        pass

    def estimate_fill_prob(self, 
                          side: str, 
                          price: float, 
                          mid_price: float, 
                          imbalance: float, 
                          volatility: float, 
                          depth_vol_ahead: float = 0.0) -> float:
        """
        Returns probability (0.0 - 1.0) of fill.
        """
        # 1. Base Probability based on Distance
        # Closer to mid = Higher prob. 
        # Inside spread (improving best) = Very High.
        
        dist_pct = 0.0
        if mid_price > 0:
            dist_pct = abs(price - mid_price) / mid_price
            
        # Heuristic: 
        # If we are improving Best Bid/Ask (aggressive Maker), Prob is high (~0.9)
        # If we are passive (behind best), Prob decays with distance and queue.
        
        # Simple Decay Model
        # Base: 0.8
        base_prob = 0.8
        
        # Adjust for Imbalance
        # If Buying (side='buy') and Imbalance is Positive (Buying Pressure), 
        # Price is likely to move UP away from our limit order. Prob Decreases.
        # If Selling and Imbalance Positive, Price moves UP into our limit. Prob Increases.
        
        imb_factor = 0.0
        if side == 'buy':
            # High buy pressure -> Price runs away -> Harder to fill passive buy
            imb_factor = -0.2 * imbalance 
        else: # sell
            # High buy pressure -> Price runs up -> Easier to fill passive sell
            imb_factor = 0.2 * imbalance
            
        # Adjust for Volatility
        # High Vol = Price moves more -> Higher chance of touching order (Execution Risk aside)
        # But also higher chance of Adverse Selection.
        # For pure "fill prob", High Vol increases chance of touch.
        vol_factor = min(0.2, volatility * 100) 
        
        # Adjust for Queue
        # More volume ahead = Lower prob
        # Decays exponentially
        queue_factor = -0.1 * np.log1p(depth_vol_ahead)
        
        final_prob = base_prob + imb_factor + vol_factor + queue_factor
        return float(np.clip(final_prob, 0.0, 1.0))

    def recommend_execution(self, fill_prob: float, urgency: float = 0.5) -> str:
        """
        Decides Execution Mode.
        urgency: 0.0 (Patient) to 1.0 (Panic)
        """
        # If probabilistic fill < Threshold, force Taker
        # Threshold depends on urgency.
        # High urgency = Require very high fill prob to stay Maker.
        
        threshold = 0.4 + (0.4 * urgency) # 0.4 to 0.8
        
        if fill_prob > threshold:
            return "MAKER"
        else:
            return "TAKER"
