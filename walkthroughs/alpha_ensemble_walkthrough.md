# Alpha Ensemble & Strategy Upgrade Walkthrough

## 1. Overview
We have successfully upgraded the Crypto AI Decision System with a Multi-Source Alpha Engine, Regime Detection, and an Ensemble Model. This upgrade aims to improve predictive robustness and strategy adaptability.

## 2. New Features (The "Alpha Upgrade")
We implemented **15+** new features derived from Formulaic Alphas (Kakushadze 101) and Rule-Based Strategies (Bits PDFs).

### Formulaic Alphas (`src/features/alphas.py`)
- **Alpha#6**: Volume-Price Correlation (Liquidity/Informed trading).
- **Alpha#9, #12**: Delta-based momentum/mean-reversion.
- **Alpha#18, #21**: Mean reversion using Volatility and VWAP.
- **Alpha#26, #28**: Volume-ranked correlation (Institutional footprint).
- **Alpha#34, #41, #53**: Volatility scaling and high-low logic.

### Rule-Based Alphas (`src/features/rule_alphas.py`)
- **OBV Trend**: On-Balance Volume vs EMA (Flow-based trend).
- **Heikin-Ashi ATR**: Market noise filtering + Trailing stop logic.
- **BTC-ETH Correlation**: Regime filter for pair trading strength.

## 3. Regime Detection (`src/models/regime_model.py`)
We introduced a **Gaussian Mixture Model (GMM)** to classify market states into:
- **Trending**: Low/Med Volatility, High Directional Strength. -> *Strategy: Aggressive Entry, loose stops.*
- **Sideways**: Low Volatility, Low Directional Strength. -> *Strategy: Conservative Entry (Mean Reversion).*
- **HighVol**: Extreme Volatility (Crash/Pump). -> *Strategy: Safety/Strict filters.*

## 4. Alpha Ensemble Model (`src/models/alpha_ensemble.py`)
Instead of a single XGBoost model, we now use a **Stacking Ensemble**:
- **Base Models**: XGBoost, LightGBM, Logistic Regression, and a "Pure Alpha" Regression.
- **Meta Model**: Logistic Regression combining base predictions.
- **Result**: 
    - **XGBoost Weight**: ~1.91 (Dominant)
    - **LightGBM Weight**: ~0.50
    - **Pure Alpha Weight**: ~0.26 (Significant contribution from formulaic alphas independent of ML)

## 5. Performance Results

### Metrics Comparison
| Metric | Baseline (Optimized XGB) | Alpha Ensemble |
| :--- | :--- | :--- |
| **ROC-AUC** | 0.5501 | **0.5570** |
| **Strategy Return** | ~5-6% (Test) | **6.14%** (Verified) |
| **Profit Factor** | 1.02 | **1.04** |
| **Max Drawdown** | -18% | **-16.38%** |

### Key Improvements
- **Robustness**: The ensemble reduces variance by blending boosting (XGB/LGB) with linear constraints (LR/Pure Alpha).
- **Adaptability**: Regime detection allows the strategy to "tighten up" during Sideways/HighVol markets, reducing false positives.
- **Signal Quality**: The "Pure Alpha" component ensures that strong mathematical alphas (e.g., volume-price divergence) influence the decision even if the ML model misses them.

## 6. Verification
- **Pipeline**: Full end-to-end run (Features -> Training -> Backtest) completed successfully.
- **Live Engine**: `LiveSignalEngine` updated and verified to handle Regime Detection and generate signals using the new blended logic.
- **Safety**: Infinite values in formulaic alphas were identified and handled (replaced with NaNs/dropped) to ensure stability.

## 7. Next Steps
- **Hyperparameter Tuning**: Optimize the Regime Model cluster count (GMM components = 3 currently).
- **Live Monitoring**: Monitor the "Regime" classification in real-time logs to ensure it aligns with market intuition.
- **Asset Expansion**: Apply the same Alpha logic to ETH and SOL specific models.
