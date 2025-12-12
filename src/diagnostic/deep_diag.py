
import sys
import os
import json
import logging
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix
import torch

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DEEP_DIAG")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOG_FILE = PROJECT_ROOT / "predictions.log"
MODEL_PATH = DATA_DIR / "models" / "multifactor_model.pkl"

PARAMS = {
    "root_cause": "Pending",
    "fixes_applied": [],
    "model_artifact": "Unknown",
    "metrics": {},
    "before_after_sample": "",
    "tests": {},
    "playbook": ""
}

def analyze_predictions_log():
    logger.info("A.[2,4] Analyzing predictions.log for class balance and calibration...")
    if not LOG_FILE.exists():
        logger.error("predictions.log not found.")
        return

    try:
        # Format: timestamp, level?, PredDir, PredScore, RealDir, Ret, Icon
        # Sample: 2025-12-12 21:45:26,711,Neutral,0.5000,Down,-0.016733,❌
        # It seems comma separated, but first field has spaces/commas in date?
        # 2025-12-12 21:45:26,711 is Tstamp?
        
        # Let's verify format.
        # It looks like: Date Time,Level?,PredDir,Score,RealDir,Ret,Icon
        # Actually checking `main.py`:
        # f"{LAST_PRED_DIR},{LAST_PRED_SCORE:.4f},{real_dir},{ret_since:.6f},{icon}"
        # But acc_logger also adds "%(asctime)s,%(message)s".
        # So: "2025-12-12 21:45:26,711,Neutral,0.5000,Down,-0.016733,❌"
        # Cols: Timestamp, PredDir, Score, RealDir, Ret, Icon
        
        df = pd.read_csv(LOG_FILE, names=["Timestamp", "PredDir", "Score", "RealDir", "Ret", "Icon"], sep=",")
        
        # Class Balance
        counts = df["PredDir"].value_counts(normalize=True)
        logger.info(f"Class Balance (Last 1000): {counts.to_dict()}")
        
        neutral_frac = counts.get("Neutral", 0.0)
        
        # Check Calibration (Score vs Real correctness?)
        # Scores are 0.5 mostly.
        # This confirms "Always Neutral".
        
        PARAMS["metrics"]["class_balance"] = counts.to_dict()
        PARAMS["metrics"]["neutral_fraction"] = neutral_frac
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to parse log: {e}")
        return None

def check_model_files():
    logger.info("A.[6,7] Checking Model Files...")
    if not MODEL_PATH.exists():
        logger.warning(f"Model file not found at {MODEL_PATH}")
        return False
        
    try:
        # Load Model
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded: {type(model)}")
        PARAMS["model_artifact"] = str(MODEL_PATH)
        
        # Check if it has predict
        if hasattr(model, "predict_proba"):
            # Sanity check with dummy
            dummy = np.random.rand(1, 20) # Assume 20 features?
            try:
                # Need feature count
                n_features = model.n_features_in_
                dummy = np.random.rand(1, n_features)
                prob = model.predict_proba(dummy)
                logger.info(f"Model Sanity Check (Prob): {prob}")
            except Exception as e:
                logger.warning(f"Model predict failed (Input shape mismatch?): {e}")
        return True
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return False

def check_main_py_connectivity():
    logger.info("B. Root Cause Heuristic: checking main.py connectivity...")
    main_py = PROJECT_ROOT / "src" / "main.py"
    with open(main_py, "r") as f:
        content = f.read()
        
    if "multifactor_model.predict" not in content and "dot_model" not in content.split("def job():")[1]:
        logger.error("CRITICAL: ML Models detected BUT NOT CALLED inside job() loop.")
        PARAMS["root_cause"] = "Model Disconnected. `job()` loop uses rule-based Arbitrator but ignores loaded ML models."
        return False
        
    return True

def save_report():
    with open("metrics_report.json", "w") as f:
        json.dump(PARAMS, f, indent=2)
    
    with open("root_cause.txt", "w") as f:
        f.write(PARAMS["root_cause"])
        
    logger.info("Deep Diagnostic Complete. Saved metrics_report.json and root_cause.txt")

if __name__ == "__main__":
    df = analyze_predictions_log()
    model_ok = check_model_files()
    connected = check_main_py_connectivity()
    
    if not connected:
        logger.info("Recommendation: Wire ML models into main.py job() loop.")
        
    save_report()
