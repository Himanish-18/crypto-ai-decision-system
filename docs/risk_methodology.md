
# Risk Methodology

## 1. Exposure Limits
- **Max Leverage**: 3.0x (Hard Stop)
- **Max Asset Concentration**: 40% per asset

## 2. Scenario Testing
Portfolio is stressed daily against:
- **COVID-19 Crash (-50%)**
- **FTX Collapse (-25%)**

## 3. Hard-Veto System
Any trade violating a pre-trade check is rejected with code `REJECT_RISK_GATE`.
