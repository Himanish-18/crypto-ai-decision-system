# Optimized Multi-Factor Model Report

## 1. Executive Summary
We successfully pruned the feature set from **200+** to **66** highly relevant features using Correlation and SHAP analysis. The model was retrained using a Stacking Ensemble (XGBoost + LightGBM -> Logistic Regression), optimized via Optuna with Stratified Time Series Cross-Validation.

**Key Result**: The new alphas (Trade Imbalance, VWAP Reversal, Cross-Asset Spread) are among the **top 10 predictors**, validating the Multi-Factor approach.

## 2. Optimization Results

### Metrics
| Metric | Value | Notes |
| :--- | :--- | :--- |
| **OOF AUC** | **0.5574** | Robustness over 5 TimeSeries splits. |
| **Feature Count** | **66** | Reduced complexity & latency. |

### Meta-Model Weights
The Logistic Regression Meta-Learner heavily favors the Tree-based models:
- **XGBoost**: `2.28` (Dominant)
- **LightGBM**: `1.10`
- **Logistic Reg (Base)**: `0.07`

## 3. Feature Importance (Top 20)
New alphas are highlighted in **Bold**.

1. `eth_rsi_14`
2. **`btc_alpha_trade_imbalance`** (Microstructure)
3. **`btc_alpha_vwap_reversal`** (Mean Reversion)
4. `btc_volume_lag_3`
5. `btc_ret_lag_3`
6. `btc_zscore_5`
7. `btc_ret_lag_1`
8. **`alpha_cross_asset_spread`** (Stat Arb)
9. `btc_volume_lag_1`
10. **`btc_alpha_vol_imbalance`**
11. `eth_zscore_10`
12. **`alpha_eth_idiosyncratic`**
13. `btc_rsi_14`
14. `btc_bb_width`
15. `btc_macd_diff`
16. `eth_roll_std_20`
17. **`eth_alpha_vwap_reversal`**
18. **`eth_alpha_vpin_proxy`**
19. `eth_ret_lag_3`
20. **`eth_alpha_vol_imbalance`**

## 4. Hyperparameters (Optuna Best)
**XGBoost**:
- `n_estimators`: 299
- `max_depth`: 5
- `learning_rate`: 0.011
- `subsample`: 0.50

**LightGBM**:
- `n_estimators`: 100
- `num_leaves`: 30
- `learning_rate`: 0.011

## 5. Latency Impact
Pruning 66% of features (approx 200 -> 66) significantly reduces:
1.  **Inference Time**: Fewer computations in Tree traversals.
2.  **Data Processing**: Lower RAM usage for history buffers.
3.  **Noise**: Removal of correlated/irrelevant features improves generalization.

## 6. Next Steps
- Deploy `optimized_model.pkl` and `selected_alpha_features.json` to production.
- Monitor `btc_alpha_trade_imbalance` real-time values as it is a top predictor.
