import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import logging

logger = logging.getLogger("regime_model")

class RegimeDetector:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.model = GaussianMixture(n_components=n_components, random_state=42)
        self.scaler = StandardScaler()
        # Mapping regime labels to human names based on characteristics
        # This needs to be determined after fitting
        self.regime_map = {} 

    def fit(self, df: pd.DataFrame, symbol="btc"):
        """
        Fit GMM on Volatility and Trend strength features.
        Features: ATR/Price (Vol), ADX (Trend Strength), Returns Volatility.
        """
        data = self._prepare_features(df, symbol)
        data_clean = data.dropna()
        
        if data_clean.empty:
             logger.warning("Not enough data to fit Regime Model")
             return self
             
        self.scaler.fit(data_clean)
        data_scaled = self.scaler.transform(data_clean)
        
        self.model.fit(data_scaled)
        
        # Analyze regimes
        labels = self.model.predict(data_scaled)
        data_clean["regime"] = labels
        
        # Heuristic to map labels:
        # High ATR -> High Vol
        # Low ATR, High ADX -> Trending?
        # Low ATR, Low ADX -> Sideways
        
        # Calculate mean features per regime
        means = data_clean.groupby("regime").mean()
        
        # Simple sorting:
        # Sort by Volatility (ATR)
        # Lowest Vol -> Sideways or Low Vol
        # Highest Vol -> High Vol
        # Middle -> Trend? 
        # This is unsupervised, so mapping is tricky. 
        # For simplicity, we'll label by Volatility level: 0=Low, 1=Med, 2=High
        
        sorted_by_vol = means.sort_values(f"{symbol}_atr_norm")
        self.regime_map = {reg: i for i, reg in enumerate(sorted_by_vol.index)}
        # Map to strings
        names = ["LowVol", "Trend/Med", "HighVol"] 
        # This mapping assumes correlation between volatility levels and these names
        # Better: use 'Regime 0', 'Regime 1' etc and interpret later.
        
        self.labels_ = labels
        return self

    def predict(self, df: pd.DataFrame, symbol="btc"):
        """Predict regime for new data."""
        features = self._prepare_features(df, symbol)
        
        # Rows with NaNs cannot be predicted
        valid_mask = ~features.isna().any(axis=1)
        valid_data = features[valid_mask]
        
        if valid_data.empty:
            return pd.Series("Unknown", index=df.index)
            
        data_scaled = self.scaler.transform(valid_data)
        labels = self.model.predict(data_scaled)
        
        # vector mapping
        mapped_labels = [self.regime_map.get(l, l) for l in labels]
        str_map = {0: "Sideways", 1: "Trending", 2: "HighVol"}
        mapped_strs = [str_map.get(l, "Unknown") for l in mapped_labels]
        
        # Reconstruct Series
        result = pd.Series("Unknown", index=df.index)
        result.loc[valid_data.index] = mapped_strs
        return result

    def _prepare_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        # Normalized ATR
        features[f"{symbol}_atr_norm"] = df[f"{symbol}_atr_14"] / (df[f"{symbol}_close"] + 1e-9)
        
        # Rolling Vol of returns
        ret = df[f"{symbol}_close"].pct_change()
        features[f"{symbol}_vol_24"] = ret.rolling(24).std()
        
        # Trend Strength Proxy (Abs z-score of price)
        features[f"{symbol}_macd_strength"] = df[f"{symbol}_macd"].abs()
        
        # Do NOT dropna here, handle in fit/predict
        return features

    def save(self, path: Path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path):
        return joblib.load(path)
