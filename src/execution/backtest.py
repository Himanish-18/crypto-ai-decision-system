import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import xgboost as xgb

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("backtest")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"
MODELS_DIR = DATA_DIR / "models"
EXECUTION_DIR = DATA_DIR / "execution"
EXECUTION_DIR.mkdir(parents=True, exist_ok=True)

class Backtester:
    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        features_path: Path,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.00075,
        slippage: float = 0.0005,
        stop_loss: float = 0.02,
        take_profit: float = 0.04,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features_path = features_path
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        self.model = None
        self.scaler = None
        self.df = None
        self.trades = []
        self.equity_curve = []

    def load_artifacts(self):
        """Load model, scaler, and data."""
        logger.info(f"📥 Loading model from {self.model_path}...")
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
            
        logger.info(f"📥 Loading scaler from {self.scaler_path}...")
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        logger.info(f"📥 Loading features from {self.features_path}...")
        self.df = pd.read_parquet(self.features_path)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

    def prepare_data(self):
        """Prepare data for inference (Test set only)."""
        # Chronological split to match training (Test is last 15%)
        n = len(self.df)
        val_end = int(n * 0.85)
        self.test_df = self.df.iloc[val_end:].copy().reset_index(drop=True)
        
        logger.info(f"🧪 Backtesting on Test Set: {len(self.test_df)} rows")
        
        # Prepare features
        exclude_cols = ["timestamp", "y_direction_up", "btc_ret_fwd_1"]
        feature_cols = [c for c in self.test_df.columns if c not in exclude_cols]
        
        X_test = self.test_df[feature_cols].values
        X_test_scaled = self.scaler.transform(X_test)
        
        # Generate Predictions
        logger.info("🔮 Generating predictions...")
        self.test_df["y_pred"] = self.model.predict(X_test_scaled)
        self.test_df["y_prob"] = self.model.predict_proba(X_test_scaled)[:, 1]

    def run_backtest(self):
        """Execute trading strategy."""
        logger.info("⚙️ Running Backtest...")
        
        capital = self.initial_capital
        position = 0  # 0: Flat, 1: Long
        entry_price = 0.0
        entry_time = None
        
        # Iterate through candles
        # Note: Iteration is slower than vectorization but allows for accurate SL/TP handling
        # For "vectorized logic", we could calculate signals first, but SL/TP requires path dependency.
        # We will use a fast loop.
        
        equity = []
        
        for i, row in self.test_df.iterrows():
            current_price = row["btc_close"]
            current_time = row["timestamp"]
            signal = row["y_pred"]
            
            # Mark to Market Equity
            current_equity = capital
            if position == 1:
                # Unrealized PnL
                pnl = (current_price - entry_price) / entry_price
                current_equity = capital * (1 + pnl)
            
            equity.append({"timestamp": current_time, "equity": current_equity})
            
            # Check Exits (SL/TP) if in position
            if position == 1:
                pct_change = (current_price - entry_price) / entry_price
                
                exit_reason = None
                if pct_change <= -self.stop_loss:
                    exit_reason = "Stop Loss"
                elif pct_change >= self.take_profit:
                    exit_reason = "Take Profit"
                elif signal == 0: # Model signal exit
                    exit_reason = "Signal Exit"
                
                if exit_reason:
                    # Execute Sell
                    exit_price = current_price * (1 - self.slippage) # Sell into bid
                    gross_pnl = (exit_price - entry_price) / entry_price
                    
                    # Apply fees (entry + exit)
                    # We deduct fees from capital
                    # Trade PnL = (Exit Value - Entry Value) - Fees
                    # Entry Value = Capital used
                    # Let's assume compounding: we use all capital
                    
                    # More precise:
                    # Position Size = Capital / Entry Price
                    # Entry Fee = Position Size * Entry Price * Fee Rate
                    # Exit Fee = Position Size * Exit Price * Fee Rate
                    # Net PnL = (Position Size * Exit Price) - (Position Size * Entry Price) - Entry Fee - Exit Fee
                    
                    pos_size = capital / entry_price # Units of BTC
                    entry_fee = pos_size * entry_price * self.fee_rate
                    exit_fee = pos_size * exit_price * self.fee_rate
                    
                    net_pnl_value = (pos_size * exit_price) - (pos_size * entry_price) - entry_fee - exit_fee
                    capital += net_pnl_value
                    
                    self.trades.append({
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "reason": exit_reason,
                        "pnl_pct": (exit_price - entry_price) / entry_price,
                        "net_pnl": net_pnl_value,
                        "capital_after": capital
                    })
                    
                    position = 0
                    entry_price = 0.0
                    entry_time = None
                    continue # Trade closed, move to next candle
            
            # Check Entries
            if position == 0 and signal == 1:
                # Execute Buy
                entry_price = current_price * (1 + self.slippage) # Buy from ask
                entry_time = current_time
                position = 1
                
                # Deduct entry fee implicitly from capital calculation on exit, 
                # or adjust 'capital' here? 
                # Let's keep 'capital' as available cash. When in position, 'capital' is locked.
                # We update 'capital' only on exit for simplicity in this loop variable, 
                # but track equity curve correctly.
                
        self.equity_df = pd.DataFrame(equity)
        self.trades_df = pd.DataFrame(self.trades)
        
        logger.info(f"🏁 Backtest Complete. Final Capital: ${capital:.2f}")

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if self.trades_df.empty:
            logger.warning("No trades executed.")
            return {}
            
        total_return = (self.equity_df["equity"].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # CAGR
        days = (self.equity_df["timestamp"].iloc[-1] - self.equity_df["timestamp"].iloc[0]).days
        cagr = ((1 + total_return) ** (365 / days)) - 1 if days > 0 else 0
        
        # Max Drawdown
        equity_series = self.equity_df["equity"]
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (assuming hourly risk free rate ~ 0)
        returns = equity_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365) if returns.std() > 0 else 0
        
        # Trade Metrics
        win_rate = len(self.trades_df[self.trades_df["net_pnl"] > 0]) / len(self.trades_df)
        avg_trade_pnl = self.trades_df["net_pnl"].mean()
        
        gross_profit = self.trades_df[self.trades_df["net_pnl"] > 0]["net_pnl"].sum()
        gross_loss = abs(self.trades_df[self.trades_df["net_pnl"] < 0]["net_pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        metrics = {
            "initial_capital": self.initial_capital,
            "final_capital": self.equity_df["equity"].iloc[-1],
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "total_trades": len(self.trades_df),
            "avg_trade_pnl": avg_trade_pnl,
            "profit_factor": profit_factor
        }
        
        return metrics

    def save_results(self, metrics: Dict[str, Any]):
        """Save results to disk."""
        # Save Trades
        trades_path = EXECUTION_DIR / "trades.csv"
        self.trades_df.to_csv(trades_path, index=False)
        
        # Save Metrics
        metrics_path = EXECUTION_DIR / "performance.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"💾 Results saved to {EXECUTION_DIR}")

def main():
    parser = argparse.ArgumentParser(description="Run Backtest")
    parser.add_argument("--model_path", type=str, default=str(MODELS_DIR / "model_xgb_v1.pkl"))
    parser.add_argument("--scaler_path", type=str, default=str(MODELS_DIR / "scaler_v1.pkl"))
    args = parser.parse_args()
    
    backtester = Backtester(
        model_path=Path(args.model_path),
        scaler_path=Path(args.scaler_path),
        features_path=FEATURES_FILE
    )
    
    backtester.load_artifacts()
    backtester.prepare_data()
    backtester.run_backtest()
    metrics = backtester.calculate_metrics()
    
    print("\n📊 Performance Summary:")
    print(json.dumps(metrics, indent=4))
    
    backtester.save_results(metrics)

if __name__ == "__main__":
    main()
