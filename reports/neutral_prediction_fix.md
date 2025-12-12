
# Debug Report: Neural Prediction "Neutral" Lock

## Incident Summary
User reported that `predictions.log` was stuck showing only "Neutral" predictions (0.5000 confidence) despite market movement.

## Root Cause Analysis
1.  **Aggressive Noise Filtering**: The `MarketCleanlinessModel` (Noise Immunity v3) was calculating a Noise Score of **0.77-0.90**. The threshold was set to **0.65**, causing the system to trigger a "Block Trades" safety outcome in every cycle.
2.  **Stale State Loop**: In `src/main.py`, the noise check happened *before* the prediction update but *after* the validation logic. When blocked, the system returned early, never generating a new prediction. The validation logic then re-validated the *last known prediction* (initially "Neutral") against new prices indefinitely.

## Fixes Implemented
1.  **Relaxed Noise Threshold**: Increased `noise_threshold` from `0.65` to **0.85** in `src/ml/noise/cleanliness.py`. This reflects the higher volatility nature of crypto markets.
2.  **Prevented Stale Validation**: Modified `src/main.py` to explicitly reset `LAST_PRED_DIR` to "Blocked" and `LAST_PRED_SCORE` to `None` when noise blocking occurs. This stops the logger from spamming invalid "Neutral" checks.

## Current Status
- **System Active**: The `Arbitrator` is now successfully receiving signals from agents (`MomentumHunter`, `MeanRevGhost`, `VolOracle`).
- **Logic**: signals are being processed, and "Neutral" outcomes are now genuine computed decisions (e.g., signals canceling each other out) rather than a frozen state.
