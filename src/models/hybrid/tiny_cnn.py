import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import logging
from pathlib import Path

logger = logging.getLogger("tiny_cnn")

class TinyCNNProxy:
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='constant',
            max_iter=200,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.window_size = 20
        self.is_fitted = False
        
    def prepare_window_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Flatten last 20 candles OHLCV into a single vector.
        Vector size: 20 * 5 = 100 features.
        """
        # Ensure we have enough data
        if len(df) < self.window_size:
            return None
            
        # Sliding window generation for training
        # For inference (single row), we just take last 20
        # This function might need to handle both
        pass

    def create_dataset(self, df: pd.DataFrame, target_col="target"):
        """
        Create X, y dataset where X is flattened windows.
        """
        cols = ["btc_open", "btc_high", "btc_low", "btc_close", "btc_volume"]
        existing_cols = [c for c in cols if c in df.columns]
        
        if len(existing_cols) < 5:
            logger.warning("Missing OHLCV columns for TinyCNN")
            return None, None
            
        data = df[existing_cols].values
        param_y = df[target_col].values if target_col in df.columns else None
        
        X = []
        y = []
        
        # Vectorized or simple loop? Loop is easier for windowing
        for i in range(self.window_size, len(data)):
            window = data[i-self.window_size:i]
            # Normalize window locally? (Percent change from window start)
            # Or use global scaler? Global scaler is safer for MLP.
            X.append(window.flatten())
            if param_y is not None:
                y.append(param_y[i])
                
        return np.array(X), np.array(y)

    def fit(self, df: pd.DataFrame, target_col="target"):
        logger.info("🧠 Training Tiny-CNN Proxy (MLP)...")
        X, y = self.create_dataset(df, target_col)
        
        if X is None or len(X) == 0:
            logger.error("No data for Tiny-CNN")
            return
            
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        score = self.model.score(X_scaled, y)
        logger.info(f"✅ Tiny-CNN Training Accuracy: {score:.4f}")
        
    def predict_score(self, df_window: pd.DataFrame) -> float:
        """
        Predict for the latest window in the dataframe.
        Expects df to have at least 20 rows.
        """
        if not self.is_fitted:
            return 0.5
            
        cols = ["btc_open", "btc_high", "btc_low", "btc_close", "btc_volume"]
        existing_cols = [c for c in cols if c in df_window.columns]
        
        if len(df_window) < self.window_size:
            return 0.5
            
        latest_window = df_window[existing_cols].iloc[-self.window_size:].values
        X_flat = latest_window.flatten().reshape(1, -1)
        X_scaled = self.scaler.transform(X_flat)
        
        prob = self.model.predict_proba(X_scaled)[0, 1]
        return prob
        
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"💾 Tiny-CNN saved to {path}")
        
    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)
