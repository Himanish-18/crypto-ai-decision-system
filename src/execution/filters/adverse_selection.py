
import logging

# v39 Adverse Selection Filter
# Blocks trades if toxic flow probability is high (Software Only).

class AdverseSelectionFilter:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.logger = logging.getLogger("execution_guard")
        
    def check_toxicity(self, flow_imbalance, volatility, spread_width):
        """
        Returns True if trade should be blocked.
        """
        # Feature 1: Extreme Imbalance (Order Book Tilt)
        f1 = abs(flow_imbalance) # 0 to 1
        
        # Feature 2: Volatility Spike
        f2 = min(1.0, volatility * 10)
        
        # Feature 3: Widening Spread (MMs pulling quotes)
        f3 = spread_width > 50.0 # Bps check
        
        # Heuristic Probability
        prob_toxic = (f1 * 0.5) + (f2 * 0.3) + (f3 * 0.2)
        
        if prob_toxic > self.threshold:
            self.logger.warning(f"🛡️ Toxic Flow Detected (P={prob_toxic:.2f}). Blocking Trade.")
            return True
        
        return False
