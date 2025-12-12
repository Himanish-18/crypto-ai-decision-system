import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.execution.strategy_optimizer import StrategyOptimizer
from src.models.multifactor_model import MultiFactorModel
from src.risk_engine.risk_module import RiskEngine

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("stress_test")

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "stress_tests"
FEATURES_FILE = DATA_DIR / "features" / "alpha_features.parquet"
MODEL_PATH = DATA_DIR / "models" / "multifactor_model.pkl"
SCALER_PATH = DATA_DIR / "models" / "optimized_scaler.pkl"  # Use optimized scaler


class StressAudit:
    def __init__(self):
        self.metrics = {}
        self.scores = {}
        self.df = None
        self.load_data()

    def load_data(self):
        logger.info("Loading Data & Model...")
        if not FEATURES_FILE.exists():
            # Fallback
            f = DATA_DIR / "features" / "features_1H_mega_alpha.parquet"
            self.df = pd.read_parquet(f)
        else:
            self.df = pd.read_parquet(FEATURES_FILE)

        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        # Compute Alphas (New features needed for model)
        from src.features.alpha_signals import AlphaSignals

        logger.info("🔧 Computing Alpha Features (On-the-fly)...")
        alpha_eng = AlphaSignals()
        self.df = alpha_eng.compute_all(self.df, "btc")
        if "eth_close" in self.df.columns:
            self.df = alpha_eng.compute_all(self.df, "eth")

        # Load Model for Data Stress Tests
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        # Apply Regime Filter (To ensure model uses correct regime logic)
        from src.risk_engine.regime_filter import RegimeFilter

        logger.info("🏷️ Applying Regime Filter for Stress Test...")
        rf = RegimeFilter()
        # This might take a moment but ensures accuracy
        # fit_predict_and_save also updates the labels file, which is fine.
        labels = rf.fit_predict_and_save(self.df, "btc")
        self.df["regime"] = labels["regime"]

    def evaluate_model(self, df_subset, name):
        """Evaluate model on a specific data subset."""
        if len(df_subset) < 50:
            logger.warning(f"⚠️ Subset {name} too small ({len(df_subset)}). Skipping.")
            return {}

        # Prepare X, y
        target_col = "y_direction_up"
        exclude_cols = [
            "timestamp",
            target_col,
            "btc_ret_fwd_1",
            "y_pred",
            "y_prob",
            "signal_prob",
            "regime",
            "symbol",
            "mf_score",
            "entry_signal",
            "dyn_sl",
            "dyn_tp",
            "dyn_pos_size",
        ]

        # Prepare features
        exclude_cols = [
            "timestamp",
            target_col,
            "btc_ret_fwd_1",
            "y_pred",
            "y_prob",
            "signal_prob",
            "regime",
            "symbol",
            "mf_score",
            "entry_signal",
            "dyn_sl",
            "dyn_tp",
            "dyn_pos_size",
        ]

        # Load Selected Features if available
        sf_path = DATA_DIR / "models" / "selected_alpha_features.json"

        # We need to construct a DF that allows predict_composite_score to work AND AlphaEnsemble to work.
        # AlphaEnsemble selects all cols except exclude_cols.
        # So we must ensure df_for_pred contains ONLY: selected_features + exclude_cols (if present)

        # Prepare Data
        # IMPORTANT: For MultiFactorModel (Regime Upgrade), we must pass ALL features,
        # including new ones not in selected_alpha_features.json
        if hasattr(self.model, "predict_composite_score"):
            df_for_pred = df_subset.copy()
        elif sf_path.exists():
            with open(sf_path, "r") as f:
                selected = json.load(f)

            # Identify metadata cols to keep
            meta_cols = [c for c in df_subset.columns if c in exclude_cols]
            ordered_cols = selected + meta_cols

            # Ensure cols exist
            missing = [c for c in ordered_cols if c not in df_subset.columns]
            if missing:
                for m in missing:
                    df_subset[m] = 0

            df_for_pred = df_subset[ordered_cols].copy()
        else:
            feature_cols = [c for c in df_subset.columns if c not in exclude_cols]
            df_for_pred = df_subset[feature_cols + exclude_cols].copy()

        # Use Composite Score
        probs = self.model.predict_composite_score(df_for_pred).values
        preds = (probs > 0.55).astype(int)

        try:
            res = {
                "auc": roc_auc_score(df_subset[target_col], probs),
                "precision": precision_score(df_subset[target_col], preds, zero_division=0),
                "recall": recall_score(df_subset[target_col], preds, zero_division=0),
                "f1": f1_score(df_subset[target_col], preds, zero_division=0),
                "count": len(df_subset),
            }
        except Exception as e:
            logger.error(f"Error eval {name}: {e}")
            res = {
                "auc": 0.5,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "count": len(df_subset),
            }

        logger.info(f"📊 {name} Metrics: AUC={res['auc']:.4f}, F1={res['f1']:.4f}")
        return res

    def run_data_stress_tests(self):
        logger.info("🔎 Running 1. Data Stress Tests...")
        results = {}

        # a) High Volatility (> 95th ATR)
        if "btc_atr_14" in self.df.columns:
            limit = self.df["btc_atr_14"].quantile(0.95)
            high_vol_df = self.df[self.df["btc_atr_14"] > limit]
            results["High Volatility"] = self.evaluate_model(
                high_vol_df, "High Volatility"
            )

        # b) Low Liquidity (< 15th Vol)
        if "btc_volume" in self.df.columns:
            limit = self.df["btc_volume"].quantile(0.15)
            low_liq_df = self.df[self.df["btc_volume"] < limit]
            results["Low Liquidity"] = self.evaluate_model(low_liq_df, "Low Liquidity")

        # c) Macro Events (> 5% move)
        if "btc_close" in self.df.columns:
            # Approx returns if not present
            rets = self.df["btc_close"].pct_change().abs()
            macro_df = self.df[rets > 0.05]
            results["Macro Events"] = self.evaluate_model(macro_df, "Macro Events")

        self.metrics["data_stress"] = results

    def run_strategy_simulations(self):
        logger.info("💰 Running 2. Trading Strategy Stress Simulations...")
        sim_results = {}

        base_fee = 0.00075
        base_slip = 0.0005

        # 1. Base Case
        logger.info("--- Base Case ---")
        opt = StrategyOptimizer(
            MODEL_PATH,
            SCALER_PATH,
            FEATURES_FILE,
            fee_rate=base_fee,
            slippage=base_slip,
        )
        opt.load_artifacts()
        opt.prepare_data()
        opt.run_strategy()
        sim_results["Base Case"] = opt.calculate_metrics()

        # 2. 2x Slippage
        logger.info("--- Stress: 2x Slippage ---")
        opt = StrategyOptimizer(
            MODEL_PATH,
            SCALER_PATH,
            FEATURES_FILE,
            fee_rate=base_fee,
            slippage=base_slip * 2,
        )
        opt.load_artifacts()
        opt.prepare_data()
        opt.run_strategy()
        sim_results["2x Slippage"] = opt.calculate_metrics()

        # 3. 3x Fees
        logger.info("--- Stress: 3x Fees ---")
        opt = StrategyOptimizer(
            MODEL_PATH,
            SCALER_PATH,
            FEATURES_FILE,
            fee_rate=base_fee * 3,
            slippage=base_slip,
        )
        opt.load_artifacts()
        opt.prepare_data()
        opt.run_strategy()
        sim_results["3x Fees"] = opt.calculate_metrics()

        # 4. Delayed Entry (1 Candle)
        logger.info("--- Stress: Delayed Entry ---")
        opt = StrategyOptimizer(
            MODEL_PATH,
            SCALER_PATH,
            FEATURES_FILE,
            fee_rate=base_fee,
            slippage=base_slip,
        )
        opt.load_artifacts()
        opt.prepare_data()
        # Hack to shift signals
        # We need to access test_df after prepare_data and shift 'entry_signal'
        # But 'entry_signal' is generated inside prepare_data.
        # So we shift it right after.
        opt.test_df["entry_signal"] = opt.test_df["entry_signal"].shift(1).fillna(0)
        opt.run_strategy()
        sim_results["Delayed Entry"] = opt.calculate_metrics()

        # 5. Price Gaps (+1% on Buy Entry)
        # We can simulate this by increasing slippage massively for this test or injecting price modification?
        # Let's say "Sudden price gaps" means we pay 1% more.
        logger.info("--- Stress: Price Gap (1%) ---")
        # 1% gap = 0.01 slippage effectively on entry
        opt = StrategyOptimizer(
            MODEL_PATH, SCALER_PATH, FEATURES_FILE, fee_rate=base_fee, slippage=0.01
        )
        opt.load_artifacts()
        opt.prepare_data()
        opt.run_strategy()
        sim_results["Price Gap 1%"] = opt.calculate_metrics()

        self.metrics["strategy_stress"] = sim_results

    def run_risk_tests(self):
        logger.info("🛡️ Running 3. Risk Engine Stress Tests...")
        re_results = {}
        risk_engine = RiskEngine(10000)

        # 1. Kill Switch / Drawdown Logic
        # RiskEngine doesn't have a stateful 'kill switch' in the snippet shown (it's stateless mostly).
        # But we can test Position Sizing reduction on Drawdown.
        logger.info("Test: Position Sizing under Drawdown")
        size_10k = risk_engine.calculate_position_size(0.55, 50000, 0.02)

        risk_engine.capital = 5000  # 50% Drawdown
        size_5k = risk_engine.calculate_position_size(0.55, 50000, 0.02)

        # Should be roughly half or less
        # Units is units. Value is units * price.
        val_10k = size_10k * 50000
        val_5k = size_5k * 50000

        re_results["Sizing_Reduction"] = (
            "PASS" if val_5k < val_10k * 0.6 else "FAIL"
        )  # Expect ~50%

        # 2. Exposure Limit (VaR)
        logger.info("Test: VaR Breach")
        # 5% of 5000 = 250.
        # If Current Value = 5000 (Full allocation), Vol = 0.05 (High) -> VaR = 5000*1.65*0.05 = 412.5 > 250.
        breach = risk_engine.check_var_limit(5000, 0.05)
        re_results["VaR_Breach_Detection"] = "PASS" if not breach else "FAIL"

        safe = risk_engine.check_var_limit(1000, 0.01)
        re_results["VaR_Safe_Detection"] = "PASS" if safe else "FAIL"

        self.metrics["risk_stress"] = re_results

    def generate_report(self):
        logger.info("📝 Generating Audit Report...")

        # Calculate Scores (Heuristics)
        # Model Score: Base AUC * 100, penalized by stress drops
        base_auc = (
            self.metrics.get("data_stress", {})
            .get("High Volatility", {})
            .get("auc", 0.5)
        )  # Proxy
        # Strategy Score: Base Sharpe/ProfitFactor
        base_pf = (
            self.metrics.get("strategy_stress", {})
            .get("Base Case", {})
            .get("profit_factor", 1.0)
        )

        self.scores["Model Score"] = min(100, max(0, base_auc * 100))  # Simple
        self.scores["Strategy Score"] = min(100, max(0, base_pf * 30))  # PF 3.0 = 90
        self.scores["Risk Score"] = (
            100
            if all(v == "PASS" for v in self.metrics["risk_stress"].values())
            else 50
        )
        self.scores["Readiness"] = (
            self.scores["Model Score"]
            + self.scores["Strategy Score"]
            + self.scores["Risk Score"]
        ) / 3

        report = f"""# Stress Testing & Model Audit Report

## 1. Scorecard
| Metric | Score (0-100) |
| :--- | :--- |
| **Model Accuracy** | **{self.scores['Model Score']:.1f}** |
| **Strategy Robustness** | **{self.scores['Strategy Score']:.1f}** |
| **Risk Safety** | **{self.scores['Risk Score']:.1f}** |
| **Deployment Readiness** | **{self.scores['Readiness']:.1f}** |

## 2. Data Stress Tests
Performance in extreme regimes:

| Regime | AUC | Precision | Recall |
| :--- | :--- | :--- | :--- |
"""
        for regime, res in self.metrics.get("data_stress", {}).items():
            report += f"| {regime} | {res['auc']:.4f} | {res['precision']:.4f} | {res['recall']:.4f} |\n"

        report += """
## 3. Strategy Simulation Constraints
Impact of execution failures:

| Scenario | Profit Factor | Win Rate | Total Return | Max DD |
| :--- | :--- | :--- | :--- | :--- |
"""
        for scen, res in self.metrics.get("strategy_stress", {}).items():
            pf = res.get("profit_factor", 0)
            wr = res.get("win_rate_pct", 0)
            ret = res.get("total_return_pct", 0)
            mdd = res.get("max_drawdown_pct", 0)
            pf_str = f"{pf:.2f}" if pf != float("inf") else "Inf"
            report += f"| {scen} | {pf_str} | {wr:.1f}% | {ret:.1f}% | {mdd:.1f}% |\n"

        report += """
## 4. Risk Engine Validation
| Test | Result |
| :--- | :--- |
"""
        for test, res in self.metrics.get("risk_stress", {}).items():
            icon = "✅" if res == "PASS" else "❌"
            report += f"| {test} | {icon} {res} |\n"

        report += (
            """
## 5. Recommendations
- **High Volatility**: """
            + (
                "Model degrades."
                if self.scores["Model Score"] < 60
                else "Model holds up."
            )
            + """
- **Slippage Sensitivity**: """
            + (
                "Critical."
                if self.metrics["strategy_stress"]["2x Slippage"]["profit_factor"] < 1.1
                else "Managed."
            )
            + """

"""
        )
        # Save Report
        with open(OUTPUT_DIR / "stress_audit_report.md", "w") as f:
            f.write(report)

        # Save JSON
        with open(OUTPUT_DIR / "stress_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)

        logger.info(f"✅ Report saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    audit = StressAudit()
    audit.run_data_stress_tests()
    audit.run_strategy_simulations()
    audit.run_risk_tests()
    audit.generate_report()
