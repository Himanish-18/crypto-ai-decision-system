import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_model")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_data() -> pd.DataFrame:
    """Load feature dataset."""
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")
    
    logger.info(f"📥 Loading data from {FEATURES_FILE}...")
    df = pd.read_parquet(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into Train, Validation, and Test sets chronologically.
    Train: ~70%, Val: ~15%, Test: ~15%
    """
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"✂️ Data Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


def prepare_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "y_direction_up"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    """
    Prepare features and target, apply scaling (fit on train only).
    """
    # Exclude non-feature columns
    exclude_cols = ["timestamp", target_col, "btc_ret_fwd_1"] # btc_ret_fwd_1 is the raw target, y_direction_up is the class
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler, feature_cols


def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str
) -> Dict[str, Any]:
    """Calculate evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5 # Handle case with only one class
        
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    metrics = {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }
    
    logger.info(f"📊 {model_name} Metrics: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    return metrics


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> XGBClassifier:
    """Train XGBoost Classifier."""
    logger.info("🚀 Training XGBoost...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=SEED,
        eval_metric="logloss",
        use_label_encoder=False
    )
    
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_train, y_train), (X_val, y_val)], 
        verbose=False
    )
    return model


# LSTM Model Definition
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i : i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)


def train_lstm(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, input_dim: int
) -> nn.Module:
    """Train LSTM Classifier."""
    logger.info("🧠 Training LSTM...")
    
    seq_length = 10
    hidden_dim = 64
    num_layers = 2
    batch_size = 64
    epochs = 10
    learning_rate = 0.001
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    
    # Convert to tensors
    train_data = TensorDataset(torch.from_numpy(X_train_seq).float(), torch.from_numpy(y_train_seq).float())
    val_data = TensorDataset(torch.from_numpy(X_val_seq).float(), torch.from_numpy(y_val_seq).float())
    
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size) # Shuffle=False for time series? Usually True for batches is fine if sequences are independent samples, but let's keep False to be safe/consistent
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_dim, hidden_dim, num_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
    return model


def save_artifacts(model: Any, scaler: StandardScaler, metrics: Dict, model_name: str):
    """Save model, scaler, and metrics."""
    version = "v1"
    
    # Save model
    if model_name == "xgb":
        model_path = MODELS_DIR / f"model_{model_name}_{version}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    elif model_name == "lstm":
        model_path = MODELS_DIR / f"model_{model_name}_{version}.pt"
        torch.save(model.state_dict(), model_path)
        
    # Save scaler (only once usually, but saving per run is fine)
    scaler_path = MODELS_DIR / f"scaler_{version}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    # Save metrics
    metrics_path = MODELS_DIR / f"metrics_{model_name}_{version}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"✅ Artifacts saved to {MODELS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train ML Models")
    parser.add_argument("--model", type=str, choices=["xgb", "lstm"], required=True, help="Model to train")
    args = parser.parse_args()
    
    # 1. Load Data
    df = load_data()
    
    # 2. Split Data
    train_df, val_df, test_df = split_data(df)
    
    # 3. Prepare Features
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols = prepare_features(train_df, val_df, test_df)
    
    if args.model == "xgb":
        # 4. Train XGBoost
        model = train_xgboost(X_train, y_train, X_val, y_val)
        
        # 5. Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_prob, "xgb")
        
        # 6. Save
        save_artifacts(model, scaler, metrics, "xgb")
        
    elif args.model == "lstm":
        # 4. Train LSTM
        model = train_lstm(X_train, y_train, X_val, y_val, input_dim=X_train.shape[1])
        
        # 5. Evaluate
        # Need to create sequences for test set too
        seq_length = 10
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_test_seq).float().to(device)
            y_prob = model(X_tensor).cpu().numpy().squeeze()
            y_pred = (y_prob > 0.5).astype(int)
            
        metrics = evaluate_model(y_test_seq, y_pred, y_prob, "lstm")
        
        # 6. Save
        save_artifacts(model, scaler, metrics, "lstm")


if __name__ == "__main__":
    main()
