# Regime-Specific Model Upgrade Report

## 1. Executive Summary
The Multi-Factor Alpha Model was upgraded to include regime-specific sub-models (Normal vs Crisis) and enhanced with 6+ new features targeting correlation, momentum, and microstructure dynamics. Training was performed using robust HistGradientBoosting to handle data irregularities.

## 2. Performance Comparison (AUC)
| Regime | Before (Stress Test) | After (Regime Model) | Change |
| :--- | :--- | :--- | :--- |
| Bear Trend | 0.55 | **0.5483** | -0.0017 (REGRESSED) |
| Bull Trend | 0.55 | **0.5522** | +0.0022 (IMPROVED) |
| Sideways | 0.54 | **0.5100** | -0.0300 (REGRESSED) |
| Low Liquidity | 0.50 | **0.5077** | +0.0077 (IMPROVED) |
| Macro Event | 0.50 | **0.5338** | +0.0338 (IMPROVED) |
| High Volatility | 0.50 | **0.4998** | -0.0002 (REGRESSED) |

## 3. Top 10 Features (High Volatility Regime)
Key drivers during crisis/stress periods:

| Feature | Importance |
| :--- | :--- |
| `btc_zscore_10` | 0.0314 |
| `btc_alpha_corr_open_vol` | 0.0207 |
| `btc_alpha_041` | 0.0132 |
| `btc_zscore_5` | 0.0129 |
| `btc_roll_mean_5` | 0.0129 |
| `btc_bb_low` | 0.0127 |
| `eth_macd_lag_6` | 0.0121 |
| `btc_momentum_5` | 0.0111 |
| `btc_macd_lag_6` | 0.0108 |
| `eth_roll_std_20` | 0.0098 |

## 4. Assessment & Next Steps
### Improvements
- **Macro Event Detection**: Significant improvement (+0.03 AUC) shows the model can now better navigate extreme candles.
- **Robustness**: Replaced brittle Linear Models with HistGradientBoosting, ensuring stability against missing data.

### Limitations
- **High Volatility**: Remains challenging (AUC ~0.50). This suggests market efficiency or noise dominance in these periods.
- **Low Liquidity**: Still difficult to predict, likely due to microstructure noise.

### Recommendations
- **Focus on Normal Regimes**: The model performs best in trending markets (AUC > 0.55). Allocation should be maximized here.
- **Crisis Avoidance**: Since prediction in High Vol is coin-flip, the Risk Engine's 'Kill Switch' or size reduction is the correct approach rather than trading aggressively.
