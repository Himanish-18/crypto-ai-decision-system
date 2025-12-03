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
logger = logging.getLogger("backtest_regime")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "execution" / "backtest_regime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class RegimeBacktester:
    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        regime_model_path: Path,
        features_path: Path,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.00075,
        slippage: float = 0.0005,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.regime_model_path = regime_model_path
        self.features_path = features_path
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        
        self.model = None
        self.scaler = None
        self.regime_detector = None
        self.df = None
        self.trades = []
        self.equity_curve = []

    def load_artifacts(self):
        """Load model, scaler, regime detector, and data."""
        logger.info(f"📥 Loading model from {self.model_path}...")
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
            
        logger.info(f"📥 Loading scaler from {self.scaler_path}...")
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        if self.regime_model_path.exists():
            logger.info(f"📥 Loading regime detector from {self.regime_model_path}...")
            with open(self.regime_model_path, "rb") as f:
                self.regime_detector = pickle.load(f)
        else:
            logger.warning("⚠️ Regime detector not found!")
            
        logger.info(f"📥 Loading features from {self.features_path}...")
        self.df = pd.read_parquet(self.features_path)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

    def prepare_data(self):
        """Prepare data and generate signals."""
        # Use Test split (last 15%)
        n = len(self.df)
        val_end = int(n * 0.85)
        self.test_df = self.df.iloc[val_end:].copy().reset_index(drop=True)
        
        logger.info(f"🧪 Backtesting on Test Set: {len(self.test_df)} rows")
        
        # Prepare features for model
        exclude_cols = ["timestamp", "y_direction_up", "btc_ret_fwd_1"]
        feature_cols = [c for c in self.test_df.columns if c not in exclude_cols]
        
        # Align features if needed
        if hasattr(self.model, "n_features_in_"):
             if len(feature_cols) != self.model.n_features_in_:
                 if hasattr(self.model, "feature_names_in_"):
                     feature_cols = self.model.feature_names_in_
        
        X_test = self.test_df[feature_cols].values
        X_test_scaled = self.scaler.transform(X_test)
        
        # Generate Raw Predictions
        logger.info("🔮 Generating raw predictions...")
        self.test_df["y_prob"] = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # --- Regime Detection ---
        if self.regime_detector:
            logger.info("🕵️ Detecting Regimes...")
            # Predict regime for entire test set
            # Note: RegimeDetector.predict expects a DataFrame with specific columns
            # We can pass self.test_df directly as it has the raw columns
            try:
                # We need to predict row by row or batch? 
                # The HMM predict method expects (n_samples, n_features).
                # Our RegimeDetector.predict takes a DF, prepares features, scales, and predicts.
                # It returns the LAST state if passed a DF.
                # We need to expose a method to predict sequence for backtesting.
                # Let's access the internal model directly for batch prediction to be fast.
                
                X_regime = self.regime_detector.prepare_features(self.test_df)
                X_regime_scaled = self.regime_detector.scaler.transform(X_regime)
                hidden_states = self.regime_detector.model.predict(X_regime_scaled)
                
                # Map states to labels
                state_map = self.regime_detector.state_map
                self.test_df["regime_state"] = hidden_states
                self.test_df["regime"] = self.test_df["regime_state"].map(state_map)
                
            except Exception as e:
                logger.error(f"Regime detection failed: {e}")
                self.test_df["regime"] = "Unknown"
        else:
            self.test_df["regime"] = "Unknown"

    def run_strategy(self):
        """Execute strategy with dynamic thresholds."""
        logger.info("⚙️ Running Strategy Execution...")
        
        capital = self.initial_capital
        position = 0
        entry_price = 0.0
        entry_time = None
        stop_loss_price = 0.0
        take_profit_price = 0.0
        last_trade_index = -999
        self.units = 0
        self.entry_fee = 0
        
        # Risk Config
        risk_per_trade_pct = 0.02
        initial_sl_pct = 0.018
        trailing_sl_pct = 0.01
        reward_ratio = 2.5
        
        equity_curve = []
        
        for i, row in self.test_df.iterrows():
            current_price = row["btc_close"]
            current_time = row["timestamp"]
            
            # --- Dynamic Thresholds ---
            regime = row.get("regime", "Unknown")
            threshold = 0.53 # Default
            
            if regime == "Bull":
                threshold = 0.51
            elif regime == "Bear":
                threshold = 0.60
            elif regime == "Sideways":
                threshold = 0.65
            elif regime == "Crash":
                threshold = 0.99 # Do not trade
            
            # Generate Signal
            prob_signal = 1 if row["y_prob"] > threshold else 0
            
            # Trend Filter
            rsi = row["btc_rsi_14"]
            macd = row["btc_macd"]
            is_uptrend = 1 if (rsi > 50 or macd > 0) else 0
            
            entry_signal = 1 if (prob_signal == 1 and is_uptrend == 1) else 0
            
            # Mark to Market
            current_equity = capital
            if position == 1:
                unrealized_pnl = self.units * (current_price - entry_price)
                current_equity = capital + unrealized_pnl - self.entry_fee
            
            # Check Exits
            if position == 1:
                exit_reason = None
                
                # Trailing Stop Loss
                new_sl = current_price * (1 - trailing_sl_pct)
                if new_sl > stop_loss_price:
                    stop_loss_price = new_sl
                
                if current_price <= stop_loss_price:
                    exit_reason = "Stop Loss"
                    exit_price = stop_loss_price
                elif current_price >= take_profit_price:
                    exit_reason = "Take Profit"
                    exit_price = take_profit_price
                
                if exit_reason:
                    final_exit_price = exit_price * (1 - self.slippage)
                    gross_pnl_value = self.units * (final_exit_price - entry_price)
                    exit_fee = (self.units * final_exit_price) * self.fee_rate
                    net_pnl_value = gross_pnl_value - self.entry_fee - exit_fee
                    
                    capital += net_pnl_value
                    
                    self.trades.append({
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": entry_price,
                        "exit_price": final_exit_price,
                        "reason": exit_reason,
                        "net_pnl": net_pnl_value,
                        "capital_after": capital,
                        "regime": regime
                    })
                    
                    position = 0
                    self.units = 0
                    last_trade_index = i
                    
            # Check Entries
            if position == 0:
                if (i - last_trade_index) >= 4:
                    if entry_signal == 1:
                        entry_price = current_price * (1 + self.slippage)
                        loss_per_unit = entry_price * initial_sl_pct
                        max_units_risk = (capital * risk_per_trade_pct) / loss_per_unit
                        max_leverage_units = capital / entry_price
                        self.units = min(max_units_risk, max_leverage_units)
                        
                        if self.units * entry_price < 10:
                            continue
                            
                        self.entry_fee = (self.units * entry_price) * self.fee_rate
                        entry_time = current_time
                        position = 1
                        
                        stop_loss_price = entry_price * (1 - initial_sl_pct)
                        take_profit_price = entry_price * (1 + (initial_sl_pct * reward_ratio))
            
            equity_curve.append({"timestamp": current_time, "equity": current_equity})

        self.equity_df = pd.DataFrame(equity_curve)
        self.trades_df = pd.DataFrame(self.trades)
        
        logger.info(f"🏁 Backtest Complete. Final Capital: ${capital:.2f}")

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if self.trades_df.empty:
            return {}
            
        total_return = (self.equity_df["equity"].iloc[-1] - self.initial_capital) / self.initial_capital
        
        equity_series = self.equity_df["equity"]
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        returns = equity_series.pct_change().dropna()
        downside_returns = returns[returns < 0]
        sortino = (returns.mean() / downside_returns.std()) * np.sqrt(24 * 365) if not downside_returns.empty else 0
        
        wins = self.trades_df[self.trades_df["net_pnl"] > 0]
        win_rate = len(wins) / len(self.trades_df)
        
        metrics = {
            "final_capital": self.equity_df["equity"].iloc[-1],
            "total_return_pct": total_return * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "sortino_ratio": sortino,
            "win_rate_pct": win_rate * 100,
            "total_trades": len(self.trades_df)
        }
        return metrics

    def save_results(self, metrics: Dict[str, Any]):
        self.trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
        self.equity_df.to_csv(OUTPUT_DIR / "equity_curve.csv", index=False)
        with open(OUTPUT_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"💾 Results saved to {OUTPUT_DIR}")

def main():
    backtester = RegimeBacktester(
        model_path=MODELS_DIR / "model_xgb_v1.pkl",
        scaler_path=MODELS_DIR / "scaler_v1.pkl",
        regime_model_path=MODELS_DIR / "regime_model.pkl",
        features_path=FEATURES_FILE
    )
    
    backtester.load_artifacts()
    backtester.prepare_data()
    backtester.run_strategy()
    metrics = backtester.calculate_metrics()
    
    print("\n📊 Regime Backtest Results:")
    print(json.dumps(metrics, indent=4))
    
    backtester.save_results(metrics)

if __name__ == "__main__":
    main()
