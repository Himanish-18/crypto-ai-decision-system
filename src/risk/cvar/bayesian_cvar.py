
import numpy as np
from scipy.stats import t

# v39 Bayesian CVaR Estimator
# Uses t-distribution posterior to estimate tail risk more purely than historical sim.

class BayesianCVaR:
    def __init__(self, confidence=0.95):
        self.confidence = confidence
        
    def estimate(self, returns):
        """
        Returns CVaR estimate at given confidence level.
        """
        returns = np.array(returns)
        if len(returns) < 10: return 0.0
        
        # Fit Student's t distribution to returns (Heavy tails)
        df, loc, scale = t.fit(returns)
        
        # Monte Carlo Simulation from Posterior
        simulated = t.rvs(df, loc=loc, scale=scale, size=10000)
        
        cutoff = np.percentile(simulated, (1 - self.confidence) * 100)
        tail_losses = simulated[simulated <= cutoff]
        
        cvar = -np.mean(tail_losses)
        return cvar

if __name__ == "__main__":
    bvar = BayesianCVaR()
    dummy = np.random.normal(0, 0.01, 1000)
    print(f"Bayesian CVaR (95%): {bvar.estimate(dummy):.2%}")
