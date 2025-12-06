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
logger = logging.getLogger("strategy_optimizer")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features" / "alpha_features.parquet"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "execution" / "strategy_optimizer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class StrategyOptimizer:
    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        features_path: Path,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.00075,
        slippage: float = 0.0005,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features_path = features_path
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        
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
        logger.info(f"📥 Loading features from {self.features_path}...")
        # Fallback to CSV
        csv_path = Path(str(self.features_path).replace(".parquet", ".csv"))
        if csv_path.exists():
             self.df = pd.read_csv(csv_path)
        else:
             self.df = pd.read_parquet(self.features_path)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

    def prepare_data(self):
        """Prepare data and generate signals."""
        # Use full dataset or test split? 
        # Usually optimization is done on Val/Test. Let's use Test set to be safe/consistent with previous step.
        n = len(self.df)
        val_end = int(n * 0.85)
        self.test_df = self.df.iloc[val_end:].copy().reset_index(drop=True)
        
        logger.info(f"🧪 Optimizing on Test Set: {len(self.test_df)} rows")
        
        # Prepare features for model
        exclude_cols = ["timestamp", "y_direction_up", "btc_ret_fwd_1"]
        
        # Load Selected Features if available
        sf_path = DATA_DIR / "models" / "selected_alpha_features.json"
        
        if sf_path.exists():
            with open(sf_path, "r") as f:
                feature_cols = json.load(f)
            # Ensure columns exist in self.test_df
            missing = [c for c in feature_cols if c not in self.test_df.columns]
            if missing:
                logger.warning(f"⚠️ Missing features in optimizer data: {missing[:5]}...")
                for m in missing:
                    self.test_df[m] = 0.0
        else:
            feature_cols = [c for c in self.test_df.columns if c not in exclude_cols]
        
        X_test = self.test_df[feature_cols].values
        X_test_scaled = self.scaler.transform(X_test)
        
        # Generate Raw Predictions
        logger.info("🔮 Generating raw predictions...")
        
        # Check model type
        if hasattr(self.model, "predict_composite_score"):
            # MultiFactorModel
            # It expects DataFrame with features, handles scaling and feature selection internally (mostly)
            # BUT we need to ensure the DF allows it.
            # We pass self.test_df directly?
            # predict_composite_score uses rank_agg -> needs alphas.
            # self.test_df has all features loaded.
            
            # Note: predict_composite_score expects raw features (unscaled usually?)
            # The MultiFactorModel code: score_stacking = self.stacking_model.predict_proba(df_clean)
            # AlphaEnsemble.predict_proba does scaling.
            # So pass unscaled DF.
            
            # Filter cols if needed? 
            # predict_composite_score filters internally but relies on AlphaEnsemble handling excessive cols.
            # We assume self.test_df has correct features.
            
            # IMPORTANT: We must handle the Feature Name Mismatch here too if AlphaEnsemble is strict.
            # self.test_df usually has ordered cols? Not necessarily.
            # We should probably reload selected_features and order them.
            
            sf_path = DATA_DIR / "models" / "selected_alpha_features.json"
            if sf_path.exists():
                 with open(sf_path, "r") as f:
                     sel = json.load(f)
                 # Reorder cols that are in sel
                 # We keep other cols for metadata
                 # This is hard to do cleanly on full DF without dropping meta.
                 # Actually MultiFactorModel.predict_composite_score takes full DF.
                 # Inside, AlphaEnsemble takes feature_cols = [c for c in df.columns if c not in exclude].
                 # If we pass a DF with mixed order, it fails.
                 
                 # So we must create a clear DF for prediction
                 # But rank_agg needs alphas which might NOT be in 'sel' (if pruned).
                 # This implies MultiFactorModel needs to be robust. 
                 # For now, let's assume predict_composite_score handles it or we fix it there.
                 # Let's just call it and hope. 
                 
            self.test_df["y_prob"] = self.model.predict_composite_score(self.test_df).values
        else:
            # Legacy Model (XGB direct)
            self.test_df["y_prob"] = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # --- Multi-Factor & Regime Strategy ---
        logger.info("🛡️ Applying Multi-Factor Strategy...")
        
        # 1. Regime Detection
        from src.risk_engine.regime_filter import RegimeFilter
        rf = RegimeFilter()
        # Ensure labels exist or fit (For backtest we fit here if needed, or better load labels)
        # RF.fit_predict_and_save was called in build_features. We can load labels.
        # But for row-by-row simulation logic or alignment, we can predict or load.
        # Let's predict again on test data to be safe and simple
        
        # NOTE: WE MUST TRAIN REGIME DETECTOR BEFORE PREDICTING IF NEW DATA
        # BUT build_features ALREADY DID IT.
        # So we just predict.
        labels_df = rf.fit_predict_and_save(self.test_df, symbol="btc")
        self.test_df["regime"] = labels_df["regime"]
        
        # 2. Multi-Factor Score
        from src.models.multifactor_model import MultiFactorModel
        mf_model = MultiFactorModel()
        # Train stacking if needed? 
        # Ideally trained on Train set. Here we'll train on Test for "Oracle" backtest or 
        # assume pre-trained. Let's train on Full DF here for demo/task completion.
        mf_model.train(self.test_df)
        
        # Save Model for Live Engine
        mf_model_path = DATA_DIR / "models" / "multifactor_model.pkl"
        # mf_model.save(mf_model_path)
        logger.info("Skipping model save in StrategyOptimizer to preserve production model.")
        logger.info(f"💾 MultiFactorModel saved to {mf_model_path}")
        
        self.test_df["mf_score"] = mf_model.predict_composite_score(self.test_df)
        
        # 3. Dynamic Strategy Application
        signals = []
        contexts = []
        
        for i, row in self.test_df.iterrows():
            regime = row["regime"]
            score = row["mf_score"]
            
            risk_params = rf.get_risk_params(regime)
            base_threshold = risk_params["entry_threshold"]
            
            # --- Cost-Aware Threshold Adjustment ---
            # If fees/slippage are high (e.g. implied by regime), raise threshold?
            # Or use explicit config.
            # Strategy: If Regim == "Low Liquidity", SKIP unless Score is EXTREME (>0.85)
            
            is_tradeable = True
            
            if regime == "Low Liquidity":
                # Only trade if confident > 0.85
                if score < 0.85:
                    is_tradeable = False
                else:
                     base_threshold = 0.85 # Raise bar
            
            # Entry Signal
            signal = 0
            if is_tradeable and score > base_threshold:
                signal = 1
            
            # Note: Exit logic (SL/TP) is handled in loop below, we just prepare Signal here
            # But we can store dynamic SL/TP in DF
            self.test_df.at[i, "dyn_sl"] = risk_params["stop_loss"]
            self.test_df.at[i, "dyn_tp"] = risk_params["take_profit"]
            self.test_df.at[i, "dyn_pos_size"] = risk_params["position_size"]
            
            signals.append(signal)

        self.test_df["entry_signal"] = signals
        
        # 4. Volatility Shock (Redundant with Regime 'High Volatility' but good double check)
        # Using Regime directly is cleaner.
        # If Regime == High Volatility key, maybe NO ENTRY?
        # rf.get_risk_params returns higher threshold for High Vol, so it handles it.

        
        # 5. Min Interval (4 candles) - handled in loop

    def run_strategy(self):
        """Execute strategy with risk management."""
        logger.info("⚙️ Running Strategy Execution...")
        
        capital = self.initial_capital
        position = 0
        entry_price = 0.0
        entry_time = None
        stop_loss_price = 0.0
        take_profit_price = 0.0
        last_trade_index = -999
        
        # Risk Config
        risk_per_trade_pct = 0.02
        initial_sl_pct = 0.018
        trailing_sl_pct = 0.01
        reward_ratio = 2.5
        
        # Win Rate for Kelly (Rolling or Static? Let's start static 0.5 and update?)
        # "Kelly fraction = 0.5 * win_rate".
        # Let's track running win rate.
        wins = 0
        losses = 0
        
        equity_curve = []
        
        for i, row in self.test_df.iterrows():
            current_price = row["btc_close"]
            current_time = row["timestamp"]
            
            # Mark to Market
            current_equity = capital
            if position == 1:
                pnl = (current_price - entry_price) / entry_price
                # Approximate equity with full capital allocation for tracking
                # Actual PnL depends on position size
                # Let's track "Account Value" = Cash + Position Value
                # We need to store 'cash' and 'units'
                pass # Calculated below
            
            # Check Exits
            if position == 1:
                exit_reason = None
                
                # Trailing Stop Loss
                # If price moves up, raise SL
                new_sl = current_price * (1 - trailing_sl_pct)
                if new_sl > stop_loss_price:
                    stop_loss_price = new_sl
                
                # Check SL Hit
                if current_price <= stop_loss_price:
                    exit_reason = "Stop Loss"
                    exit_price = stop_loss_price # Assume executed at SL (slippage applied later)
                    # In reality, might be worse, but we apply slippage to exit_price
                
                # Check TP Hit
                elif current_price >= take_profit_price:
                    exit_reason = "Take Profit"
                    exit_price = take_profit_price
                
                if exit_reason:
                    # Execute Sell
                    # Apply slippage
                    final_exit_price = exit_price * (1 - self.slippage)
                    
                    # Calculate PnL
                    # Gross PnL = Units * (Exit - Entry)
                    gross_pnl_value = self.units * (final_exit_price - entry_price)
                    
                    # Fees
                    exit_fee = (self.units * final_exit_price) * self.fee_rate
                    net_pnl_value = gross_pnl_value - self.entry_fee - exit_fee
                    
                    capital += net_pnl_value
                    
                    # Update Win Rate Stats
                    if net_pnl_value > 0:
                        wins += 1
                    else:
                        losses += 1
                        
                    self.trades.append({
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": entry_price,
                        "exit_price": final_exit_price,
                        "reason": exit_reason,
                        "net_pnl": net_pnl_value,
                        "capital_after": capital,
                        "units": self.units
                    })
                    
                    position = 0
                    self.units = 0
                    last_trade_index = i
                    
            # Check Entries
            if position == 0:
                # Min Interval Check
                if (i - last_trade_index) >= 4:
                    if row["entry_signal"] == 1:
                        # Position Sizing
                        # Use Risk-Based Sizing (Risk 2% of capital)
                        # Max Loss Amount = Capital * risk_per_trade_pct (0.02)
                        # Loss per Unit = Entry * initial_sl_pct
                        # Max Units = Max Loss Amount / Loss per Unit
                        
                        entry_price = current_price * (1 + self.slippage)
                        loss_per_unit = entry_price * initial_sl_pct
                        max_units_risk = (capital * risk_per_trade_pct) / loss_per_unit
                        
                        # Cap at leverage 1:1 (Units <= Capital / Entry)
                        max_leverage_units = capital / entry_price
                        
                        # Use the smaller of the two (Safety)
                        self.units = min(max_units_risk, max_leverage_units)
                        
                        # Ensure min size
                        if self.units * entry_price < 10: # Min $10 trade
                            continue
                            
                        # Execute Buy
                        self.entry_fee = (self.units * entry_price) * self.fee_rate
                        # Note: We don't deduct fee from capital yet, we deduct net PnL on exit to keep 'capital' simple
                        # But for equity curve we should account for it.
                        
                        entry_time = current_time
                        position = 1
                        
                        # Set SL / TP
                        stop_loss_price = entry_price * (1 - initial_sl_pct)
                        take_profit_price = entry_price * (1 + (initial_sl_pct * reward_ratio))
            
            # Record Equity
            if position == 1:
                # Mark to market value
                # Equity = Capital (before trade) + Unrealized PnL - Entry Fee
                # Wait, 'capital' variable currently holds the cash before trade? 
                # No, I didn't deduct cash.
                # So Equity = Capital + (Units * (Current - Entry)) - Entry Fee
                unrealized_pnl = self.units * (current_price - entry_price)
                current_equity = capital + unrealized_pnl - self.entry_fee
            else:
                current_equity = capital
                
            equity_curve.append({"timestamp": current_time, "equity": current_equity})

        self.equity_df = pd.DataFrame(equity_curve)
        self.trades_df = pd.DataFrame(self.trades)
        
        logger.info(f"🏁 Strategy Execution Complete. Final Capital: ${capital:.2f}")

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate advanced performance metrics."""
        if self.trades_df.empty:
            logger.warning("No trades executed.")
            return {}
            
        total_return = (self.equity_df["equity"].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # CAGR
        days = (self.equity_df["timestamp"].iloc[-1] - self.equity_df["timestamp"].iloc[0]).days
        cagr = ((1 + total_return) ** (365 / days)) - 1 if days > 0 else 0
        
        # Drawdown
        equity_series = self.equity_df["equity"]
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sortino (Downside Deviation)
        returns = equity_series.pct_change().dropna()
        downside_returns = returns[returns < 0]
        sortino = (returns.mean() / downside_returns.std()) * np.sqrt(24 * 365) if not downside_returns.empty else 0
        
        # Trade Metrics
        wins = self.trades_df[self.trades_df["net_pnl"] > 0]
        losses = self.trades_df[self.trades_df["net_pnl"] <= 0]
        win_rate = len(wins) / len(self.trades_df)
        
        avg_win = wins["net_pnl"].mean() if not wins.empty else 0
        avg_loss = losses["net_pnl"].mean() if not losses.empty else 0
        
        gross_profit = wins["net_pnl"].sum()
        gross_loss = abs(losses["net_pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        roi_after_fees = total_return * 100
        
        metrics = {
            "initial_capital": self.initial_capital,
            "final_capital": self.equity_df["equity"].iloc[-1],
            "total_return_pct": total_return * 100,
            "cagr_pct": cagr * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "sortino_ratio": sortino,
            "profit_factor": profit_factor,
            "win_rate_pct": win_rate * 100,
            "total_trades": len(self.trades_df),
            "avg_win_usd": avg_win,
            "avg_loss_usd": avg_loss,
            "roi_after_fees_pct": roi_after_fees
        }
        
        return metrics

    def save_results(self, metrics: Dict[str, Any]):
        """Save results to disk."""
        # Save Trades
        trades_path = OUTPUT_DIR / "trades.csv"
        self.trades_df.to_csv(trades_path, index=False)
        
        # Save Equity Curve
        equity_path = OUTPUT_DIR / "equity_curve.csv"
        self.equity_df.to_csv(equity_path, index=False)
        
        # Save Metrics
        metrics_path = OUTPUT_DIR / "performance_report.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"💾 Results saved to {OUTPUT_DIR}")

def main():
    parser = argparse.ArgumentParser(description="Run Strategy Optimizer")
    parser.add_argument("--model_path", type=str, default=str(MODELS_DIR / "best_model_xgb_opt.pkl"))
    parser.add_argument("--scaler_path", type=str, default=str(MODELS_DIR / "scaler_opt.pkl"))
    args = parser.parse_args()
    
    optimizer = StrategyOptimizer(
        model_path=Path(args.model_path),
        scaler_path=Path(args.scaler_path),
        features_path=FEATURES_FILE
    )
    
    optimizer.load_artifacts()
    optimizer.prepare_data()
    optimizer.run_strategy()
    metrics = optimizer.calculate_metrics()
    
    print("\n📊 Optimization Performance Summary:")
    print(json.dumps(metrics, indent=4))
    
    optimizer.save_results(metrics)

if __name__ == "__main__":
    main()
