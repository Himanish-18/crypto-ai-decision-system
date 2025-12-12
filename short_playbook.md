# Remediation Playbook: "Always Neutral" Fix

1. **Deploy Code**: Push changes to `src/main.py` (Feature wiring) and `src/fixes/retrain_model.py`.
2. **Deploy Artifacts**: Ensure `data/models/multifactor_model_v2.pkl` is present on the production server.
3. **Verify Imports**: Confirm `torch` and `joblib` are installed and imports in `main.py` do not fail.
4. **Restart Service**: Restart the `live_trading` service (or `python src/main.py`).
5. **Monitor Logs**: Tail `predictions.log` and verify "Neutral" fraction drops below 80% and "ML_Ensemble" / "DOT_Signal" appear in logs (if logging level set to DEBUG/INFO).
