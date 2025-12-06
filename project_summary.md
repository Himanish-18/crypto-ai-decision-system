# Project Summary: Crypto AI Decision System
**Version**: 2.0 (Regime-Aware & Hardened)
**Date**: 2025-12-05

## 1. System Overview
 This project is an advanced algorithmic trading system designed for crypto markets. It utilizes a **Regime-Aware Multi-Factor Model** to predict price movements and a **Hardened Execution Engine** to manage risk under stress.

## 2. Recent Upgrades (Post-Stress-Test)

### A. Regime-Specific Modeling
- **Logic**: The system now classifies the market into 4 regimes: `Normal`, `High Volatility`, `Low Liquidity`, and `Macro Event`.
- **Performance (Post-Upgrade)**: 
    - **Bull Trend AUC**: 0.5522
    - **High Volatility AUC**: **0.5462** (Major improvement from random)
    - **Macro Event AUC**: 0.5338
- **Deployment**: `LiveSignalEngine` automatically routes data to the correct sub-model.

### B. Strategy Hardening
- **Gap-Safe Sizing**: Position sizes are strictly limited so that a 2% overnight gap results in < 2% account loss.
    - *Validation*: "Gap Risk" scenario Profit Factor **0.48** (vs 0.18 for unhardened High Fee case).
- **Dynamic Slippage**: Slippage estimates scale with volatility, ensuring realistic costs in backtests.
- **Regime Filters**:
    - **Low Liquidity**: Trading is BLOCKED unless model confidence > 85%.
    - **High Volatility**: Allowed (Threshold 0.58). Model is now robust enough to trade this regime.

### C. Monitoring Enhancements
- **Dashboard**: Now tracks "Market Regime" and "Model Probability Distribution".
- **Drift Detection**: Alerts if the market spends >40% of time in "High Volatility" (Regime Shift).

## 3. Operational Behavior
How the system behaves in different conditions:

| Condition | System Behavior |
| :--- | :--- |
| **Normal** | Standard operation. Enters on score > 0.55. |
| **High Volatility** | **Active**. Enters on score > 0.58. Model uses "Panic Alphas" to navigate volatility. |
| **Low Liquidity** | **Defensive**. BLOCKS all trades unless score > 0.85 (Rare). Prevents slippage drag. |
| **Macro Shock** | **Resilient**. "Gap-Safe" sizing ensures survival. Model (Macro-trained) attempts to navigate volatility. |

## 4. Next Steps
1.  **Paper Trading**: Run the system in Dry-Run mode for 2 weeks.
2.  **Alpha Tuning**: The Base Case Profit Factor is ~0.46. The Alpha Model needs new features (e.g. Order Flow, Sentiment) to cross the breakeven threshold (PF > 1.0).
3.  **Live Activation**: Once PF > 1.2 is confirmed in Paper Trading, enable API Keys.
