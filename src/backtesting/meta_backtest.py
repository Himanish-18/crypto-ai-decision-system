import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List
from src.models.meta_model import MetaModel
from src.execution.inventory_manager import InventoryManager
from src.execution.regime_router import RegimeRouter

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("meta_backtest")

class MetaBacktester:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.df = None
        self.inventory_manager = InventoryManager(max_capital=10000)
        self.regime_router = RegimeRouter()
        self.meta_model = MetaModel()
        
    def load_data(self):
        if not self.data_path.exists():
            logger.warning("Data file not found. Generating mock data.")
            self.df = self._generate_mock_data()
        else:
            self.df = pd.read_parquet(self.data_path)
            
    def _generate_mock_data(self, n=5000):
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=n, freq="1H")
        price = 100 + np.cumsum(np.random.normal(0, 1, n))
        
        # Mock Base Predictions
        y_true = (np.roll(price, -1) > price).astype(int)
        pred_xgb = np.clip(y_true * 0.6 + np.random.normal(0, 0.4, n), 0, 1)
        pred_lgb = np.clip(y_true * 0.65 + np.random.normal(0, 0.35, n), 0, 1)
        
        # Mock Regime
        regime = np.random.choice([0, 1, 2], n, p=[0.4, 0.5, 0.1])
        
        return pd.DataFrame({
            "timestamp": dates,
            "close": price,
            "pred_xgb": pred_xgb,
            "pred_lgb": pred_lgb,
            "regime": regime,
            "target": y_true
        })

    def run_walk_forward(self, window_size=1000, step_size=200):
        """
        Walk-Forward Validation: Train MetaModel on window, Test on step.
        """
        logger.info("🏃 Starting Walk-Forward Backtest...")
        
        equity_curve = [10000.0]
        returns = []
        
        for i in range(window_size, len(self.df) - step_size, step_size):
            train_data = self.df.iloc[i-window_size:i]
            test_data = self.df.iloc[i:i+step_size]
            
            # 1. Train Meta Model
            X_train = train_data[["pred_xgb", "pred_lgb"]]
            y_train = train_data["target"]
            self.meta_model.train(X_train, y_train)
            
            # 2. Simulate Trading on Test Step
            X_test = test_data[["pred_xgb", "pred_lgb"]]
            probs = self.meta_model.predict_proba(X_test)
            
            for j in range(len(test_data)):
                idx = test_data.index[j]
                row = test_data.iloc[j]
                prob = probs[j]
                
                # Get Regime Config
                regime_config = self.regime_router.get_execution_config(row["regime"])
                
                if not regime_config["can_trade"]:
                    returns.append(0.0)
                    continue
                    
                # Calculate Size
                size = self.inventory_manager.calculate_target_size(
                    signal_strength=prob, 
                    regime_multiplier=regime_config["size_multiplier"]
                )
                
                # Simulate PnL (Simplified: Close - Next Close)
                # Assuming we buy if prob > 0.5, sell if prob < 0.5
                # Here we only have Long logic for simplicity
                if prob > 0.6: # Long Threshold
                    ret = (self.df.iloc[idx+1]["close"] - row["close"]) / row["close"] if idx+1 < len(self.df) else 0
                    pnl = size * ret
                    
                    # Update Inventory Manager
                    self.inventory_manager.update_state(exposure=size, unrealized_pnl=pnl) # Mock unrealized as realized for step
                    
                    equity_curve.append(equity_curve[-1] + pnl)
                    returns.append(ret)
                else:
                    returns.append(0.0)
                    equity_curve.append(equity_curve[-1])
                    
        return equity_curve, returns

    def calculate_metrics(self, equity_curve, returns):
        equity = np.array(equity_curve)
        rets = np.array(returns)
        
        total_return = (equity[-1] - equity[0]) / equity[0]
        sharpe = np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(24*365) # Annualized
        
        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown)
        
        calmar = total_return / max_dd if max_dd > 0 else 0
        
        return {
            "Total Return": f"{total_return*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd*100:.2f}%",
            "Calmar Ratio": f"{calmar:.2f}",
            "Final Equity": f"${equity[-1]:.2f}"
        }

if __name__ == "__main__":
    # Use mock data path
    backtester = MetaBacktester(Path("dummy.parquet"))
    backtester.load_data()
    
    equity, rets = backtester.run_walk_forward()
    metrics = backtester.calculate_metrics(equity, rets)
    
    logger.info("📊 Backtest Results:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
