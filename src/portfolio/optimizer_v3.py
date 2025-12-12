import logging
import numpy as np
from typing import Dict, List, Optional
import warnings

# Try to import cvxpy, else fallback
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

logger = logging.getLogger("optimizer_v3")

class InstitutionalOptimizer:
    """
    v28 Institutional Logic.
    Supports Convex Optimization for mathematically optimal portfolios.
    """
    def __init__(self, n_assets: int, asset_names: List[str]):
        self.n = n_assets
        self.assets = asset_names
        
    def optimize(self, mu: np.ndarray, cov: np.ndarray, 
                 current_weights: Optional[np.ndarray] = None,
                 max_turnover: float = 0.20,
                 target_beta: float = 0.0) -> np.ndarray:
        
        if HAS_CVXPY:
            return self._optimize_cvxpy(mu, cov, current_weights, max_turnover, target_beta)
        else:
            logger.warning("CVXPY not found. Using Scipy Fallback with Penalty Functions.")
            return self._optimize_scipy(mu, cov, current_weights, max_turnover, target_beta)

    def _optimize_cvxpy(self, mu, cov, w_prev, turnover, beta_target):
        # Variables
        w = cp.Variable(self.n)
        
        # Objectives: Max Return - Lambda * Risk
        # Institutional firms maximize Utility U = w.T*mu - gamma * w.T*Sigma*w
        gamma = cp.Parameter(nonneg=True)
        gamma.value = 1.0 # Risk aversion
        
        ret = mu @ w
        risk = cp.quad_form(w, cov)
        obj = cp.Maximize(ret - gamma * risk)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
        ]
        
        if w_prev is not None:
            constraints.append(cp.norm(w - w_prev, 1) <= turnover)
            
        # Problem
        prob = cp.Problem(obj, constraints)
        prob.solve()
        
        return w.value

    def _optimize_scipy(self, mu, cov, w_prev, turnover, beta_target):
        # Fallback implementation
        from scipy.optimize import minimize
        
        def objective(w):
            # Utility = Mean - Variance
            return -(np.dot(w, mu) - 0.5 * np.dot(w.T, np.dot(cov, w)))
            
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n))
        
        # Turnover Constraint penalty
        if w_prev is not None:
             # Scipy constraint: sum(|w - w_prev|) <= turnover
             constraints.append({'type': 'ineq', 'fun': lambda w: turnover - np.sum(np.abs(w - w_prev))})

        res = minimize(objective, np.ones(self.n)/self.n, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x
