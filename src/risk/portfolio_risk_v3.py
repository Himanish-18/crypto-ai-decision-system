import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("portfolio_risk_v3")

class PortfolioRiskEngine:
    """
    v20 Institutional Portfolio Risk Engine.
    Scales factor-based risk analysis and creates VaR/ES metrics.
    """
    def __init__(self, confidence_level: float = 0.99, lookback: int = 252):
        self.conf = confidence_level
        self.lookback = lookback
        self.positions = {} # {Symbol: {"amt": float, "pv": float, "beta": float}}
        self.equity = 0.0
        
        # Factor Loadings Cache (Symbol -> {Factor: Beta})
        # Factors: MKT (BTC), MOM (Momentum), VOL (Volatility)
        self.factor_loadings = {} 
        
    def update_portfolio(self, positions_dict: Dict[str, float], prices: Dict[str, float], total_equity: float):
        """
        positions_dict: {Symbol: Amount} (e.g. BTC: 1.5, ETH: -10.0)
        prices: {Symbol: Price_USD}
        """
        self.positions = {}
        self.equity = total_equity
        
        for sym, amt in positions_dict.items():
            price = prices.get(sym, 0.0)
            pv = amt * price
            
            # Default Factors if unknown
            # Beta to BTC: 1.0 for BTC/BTC paired, 1.2 for ETH, 0 for Stable
            if "BTC" in sym:
                beta_mkt = 1.0
            elif "ETH" in sym:
                beta_mkt = 1.2
            else:
                beta_mkt = 0.0
            
            self.positions[sym] = {
                "amt": amt,
                "price": price,
                "usd_value": pv,
                "weight": pv / total_equity if total_equity > 0 else 0.0,
                "beta_mkt": beta_mkt
            }
            
    def calculate_risk_metrics(self, history: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate VaR and ES based on historical returns.
        history: DataFrame of asset returns (columns = symbols)
        """
        if not self.positions:
            return {"var_99": 0.0, "es_99": 0.0}
            
        # Construct Portfolio Return Series
        port_ret = pd.Series(0.0, index=history.index)
        
        for sym, data in self.positions.items():
            if sym in history.columns:
                w = data["weight"]
                port_ret += w * history[sym]
            else:
                # Fallback: Use Beta * BTC_Ret
                if "BTC" in history.columns:
                    w = data["weight"]
                    beta = data["beta_mkt"]
                    port_ret += w * beta * history["BTC"]
        
        # Metrics
        losses = -port_ret
        var = losses.quantile(self.conf)
        es = losses[losses > var].mean()
        
        # Factor Exposures
        net_delta = sum(d["usd_value"] for d in self.positions.values())
        beta_weighted_exp = sum(d["usd_value"] * d["beta_mkt"] for d in self.positions.values())
        
        return {
            "var_99": var,
            "es_99": es,
            "var_usd": var * self.equity,
            "es_usd": es * self.equity,
            "net_delta": net_delta,
            "beta_adjusted_exposure": beta_weighted_exp,
            "leverage": sum(abs(d["usd_value"]) for d in self.positions.values()) / self.equity if self.equity > 0 else 0
        }

    def check_constraints(self, metrics: Dict[str, float], constraints: Dict[str, float]) -> List[str]:
        """
        Check if portfolio violates limits.
        constraints: {"max_var": 0.05, "max_leverage": 3.0}
        """
        violations = []
        
        if metrics.get("var_99", 0) > constraints.get("max_var", 1.0):
            violations.append(f"VaR Violation: {metrics['var_99']:.2%} > {constraints['max_var']:.2%}")
            
        if metrics.get("leverage", 0) > constraints.get("max_leverage", 10.0):
            violations.append(f"Leverage Violation: {metrics['leverage']:.2f} > {constraints['max_leverage']}")
            
        return violations
