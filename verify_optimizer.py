import numpy as np
from src.portfolio.optimizer_v3 import InstitutionalOptimizer

def test_opt():
    print("Testing Optimizer v3...")
    mu = np.array([0.05, 0.03, 0.02])
    cov = np.array([[0.01, 0.002, 0.001],
                    [0.002, 0.01, 0.002],
                    [0.001, 0.002, 0.01]])
    
    opt = InstitutionalOptimizer(3, ['A', 'B', 'C'])
    w = opt.optimize(mu, cov)
    
    print(f"Optimal Weights: {w}")
    assert np.abs(np.sum(w) - 1.0) < 1e-5, "Weights must sum to 1"
    assert all(w >= -1e-5), "Long only constraint failed" # Small tolerance
    
    print("✅ Optimizer v3 Verified.")

if __name__ == "__main__":
    test_opt()
