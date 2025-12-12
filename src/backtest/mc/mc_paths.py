
import numpy as np

# v39 Monte Carlo Path Generator
# Generates 10,000 alternate price histories using Geometric Brownian Motion + Jump Diffusion.

class MCPaths:
    def __init__(self, num_sims=1000):
        self.num_sims = num_sims

    def generate_paths(self, start_price, mu, sigma, days, dt=1/365):
        """
        Returns: [Sims, Steps] array
        """
        num_steps = days
        
        # GBM: dS = mu*S*dt + sigma*S*dW
        S = np.zeros((self.num_sims, num_steps + 1))
        S[:, 0] = start_price
        
        for t in range(1, num_steps + 1):
            rand = np.random.standard_normal(self.num_sims)
            
            # Jump Process (Poisson)
            # 1% chance of 5% drop
            jumps = (np.random.random(self.num_sims) < 0.01) * -0.05
            
            drift = (mu - 0.5 * sigma**2) * dt
            shock = sigma * np.sqrt(dt) * rand
            
            S[:, t] = S[:, t-1] * np.exp(drift + shock + jumps)
            
        return S

if __name__ == "__main__":
    mc = MCPaths(10)
    paths = mc.generate_paths(50000, 0.05, 0.6, 30)
    print(f"Generated {paths.shape} paths")
