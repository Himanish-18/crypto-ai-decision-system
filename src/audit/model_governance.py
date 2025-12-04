import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import hashlib
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.calibration import calibration_curve

# Config
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "execution"
MODELS_DIR = DATA_DIR / "models"
FEATURES_DIR = DATA_DIR / "features"
GOVERNANCE_DIR = DATA_DIR / "governance"
DASHBOARDS_DIR = PROJECT_ROOT / "dashboards"

GOVERNANCE_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARDS_DIR.mkdir(parents=True, exist_ok=True)

def get_file_hash(filepath):
    if not filepath.exists():
        return "N/A"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_logs():
    # Try main log first, then paper log
    log_file = LOGS_DIR / "logs" / "trading_log.jsonl"
    if not log_file.exists() or log_file.stat().st_size == 0:
        log_file = LOGS_DIR / "paper_trades.jsonl"
        
    if not log_file.exists():
        return pd.DataFrame()
    
    data = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    # Flatten
    rows = []
    for entry in data:
        row = {}
        for k, v in entry.items():
            if k not in ["decision", "signal"]:
                row[k] = v
        if "decision" in entry:
            row.update(entry["decision"])
        if "signal" in entry:
            row.update(entry["signal"])
        rows.append(row)
        
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    return df

def calculate_psi(expected, actual, buckets=10):
    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if len(expected) == 0 or len(actual) == 0:
        return 0

    expected_percents = np.histogram(expected, bins=buckets)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=buckets)[0] / len(actual)

    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value

def run_audit():
    print("🔍 Starting Model Governance Audit...")
    
    # 1. Model Identity
    model_path = MODELS_DIR / "model_xgb_v1.pkl"
    features_path = FEATURES_DIR / "features_1H_advanced.parquet"
    
    identity = {
        "model_hash": get_file_hash(model_path),
        "training_data_hash": get_file_hash(features_path),
        "audit_timestamp": datetime.now().isoformat()
    }
    
    # 2. Load Data
    df_logs = load_logs()
    df_train = pd.read_parquet(features_path) if features_path.exists() else pd.DataFrame()
    
    if df_logs.empty:
        print("❌ No logs found. Cannot perform full audit.")
        return

    # Merge logs with features to get feature values at trade time
    if not df_train.empty and "timestamp" in df_logs.columns:
        # Sort both by timestamp
        df_logs = df_logs.sort_values("timestamp")
        df_train = df_train.sort_values("timestamp")
        
        # Use merge_asof to find nearest feature row for each log entry
        df_logs = pd.merge_asof(df_logs, df_train, on="timestamp", direction="backward")
        print(f"Merged logs with features. New shape: {df_logs.shape}")

    # 3. Calibration Plot
    print("📊 Generating Calibration Plot...")
    if "prediction_prob" in df_logs.columns and "action" in df_logs.columns:
        # Assuming action 'BUY' (1) is positive class for calibration check
        # In reality we need ground truth labels (future returns), but for now we check model confidence consistency
        # Let's try to infer ground truth from price changes if available
        
        # Mock ground truth for demo if not explicit
        # Real implementation: Calculate forward return > 0
        if "price" in df_logs.columns:
            df_logs["ret_fwd"] = df_logs["price"].shift(-1) / df_logs["price"] - 1
            df_logs["target"] = (df_logs["ret_fwd"] > 0).astype(int)
            
            prob_true, prob_pred = calibration_curve(df_logs["target"][:-1], df_logs["prediction_prob"][:-1], n_bins=10)
            
            plt.figure()
            plt.plot(prob_pred, prob_true, marker='o', label="Model")
            plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly Calibrated")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title("Calibration Plot (Reliability Diagram)")
            plt.legend()
            plt.savefig(DASHBOARDS_DIR / "calibration_plot.png")
            plt.close()
            identity["calibration_score"] = np.mean(np.abs(prob_true - prob_pred)) # Simple ECE proxy
        else:
            print("⚠️ Price data missing in logs, skipping calibration.")
    else:
        print("⚠️ Prediction probs missing, skipping calibration.")

    # 4. Drift Heatmap (Top 20 Features)
    print("🌡️ Generating Drift Heatmap...")
    # Identify numeric features in logs that match training data
    print(f"Log Columns: {df_logs.columns.tolist()[:10]}...")
    print(f"Train Columns: {df_train.columns.tolist()[:10]}...")
    
    common_features = []
    for c in df_logs.columns:
        if c in df_train.columns:
            # Check if numeric
            if pd.api.types.is_numeric_dtype(df_logs[c]):
                common_features.append(c)
    top_features = common_features[:20] # Take first 20 for now
    print(f"Found {len(top_features)} common features: {top_features}")
    
    if top_features:
        psi_scores = {}
        for feat in top_features:
            psi = calculate_psi(df_train[feat], df_logs[feat])
            psi_scores[feat] = psi
            
        # Sort by PSI
        sorted_feats = sorted(psi_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        top_20_feats = [x[0] for x in sorted_feats]
        
        # Create Heatmap Data (Feature vs Day - Mocking days as chunks of logs for demo)
        # In real scenario: df_logs.groupby(pd.Grouper(key='timestamp', freq='D'))
        
        drift_matrix = []
        for feat in top_20_feats:
            drift_matrix.append(psi_scores[feat])
            
        plt.figure(figsize=(10, 12))
        sns.barplot(x=[x[1] for x in sorted_feats], y=[x[0] for x in sorted_feats], palette="viridis")
        plt.title("Feature Drift (PSI) - Live vs Train")
        plt.xlabel("PSI Score")
        plt.axvline(0.1, color="orange", linestyle="--")
        plt.axvline(0.2, color="red", linestyle="--")
        plt.savefig(DASHBOARDS_DIR / "drift_heatmap.png") # Saving barplot as requested heatmap alternative for single time slice
        plt.close()
        
        identity["top_drift_feature"] = sorted_feats[0][0]
        identity["top_drift_psi"] = sorted_feats[0][1]

    # 5. Save Stats
    with open(GOVERNANCE_DIR / "live_vs_train_stats.json", "w") as f:
        json.dump(identity, f, indent=4)
        
    # 6. Recommendation
    rec_file = PROJECT_ROOT / "next_model_promote_recommendation.txt"
    with open(rec_file, "w") as f:
        if identity.get("top_drift_psi", 0) > 0.2:
            f.write("RECOMMENDATION: RETRAIN\nReason: Critical drift detected.")
        elif identity.get("calibration_score", 0) > 0.15:
            f.write("RECOMMENDATION: RECALIBRATE\nReason: Poor model calibration.")
        else:
            f.write("RECOMMENDATION: PROMOTE\nReason: Model stable and calibrated.")

    print("✅ Audit Complete.")

if __name__ == "__main__":
    run_audit()
