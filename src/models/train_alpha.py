import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from src.features.microstructure import MicrostructureFeatures
from src.models.ensemble_model import EnsembleModel
from src.models.regime_detection import MarketRegimeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_alpha")

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic L2 data for initial training if real data is missing.
    """
    logger.info("Generating synthetic training data...")
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(40000, 100, n_samples),
        'high': np.random.normal(40100, 100, n_samples),
        'low': np.random.normal(39900, 100, n_samples),
        'close': np.random.normal(40000, 100, n_samples),
        'volume': np.random.normal(100, 10, n_samples),
        # Synthetic Features
        'order_imbalance_10': np.random.uniform(-1, 1, n_samples),
        'cvd_1h': np.random.normal(0, 50, n_samples),
        'vwap_deviation': np.random.normal(0, 0.01, n_samples),
        'rsi': np.random.uniform(20, 80, n_samples)
    })
    
    # Target: 1 if next close > current close
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

def train_alpha_models():
    data_dir = Path("./data/l2")
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    # 1. Load Data
    # In a real scenario, we would load from data_dir
    # df = pd.read_csv(data_dir / "BTCUSDT_merged.csv")
    
    # For now, use synthetic data to ensure pipeline works
    df = generate_synthetic_data()
    
    # 2. Train Regime Detector
    logger.info("Training Regime Detector...")
    regime_model = MarketRegimeDetector(n_components=3)
    regime_model.fit_hmm(df)
    joblib.dump(regime_model, models_dir / "regime_model.pkl")
    
    # Add regime as feature
    df['regime'] = regime_model.predict_regime(df)
    
    # 3. Train Ensemble Model
    logger.info("Training Ensemble Model...")
    ensemble = EnsembleModel()
    
    features = ['order_imbalance_10', 'cvd_1h', 'vwap_deviation', 'rsi', 'regime']
    X = df[features]
    y = df['target']
    
    ensemble.train(X, y)
    ensemble.save(models_dir / "ensemble_model.pkl")
    
    logger.info("Training Complete. Models saved to ./models/")

if __name__ == "__main__":
    train_alpha_models()
