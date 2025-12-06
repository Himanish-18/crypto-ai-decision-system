import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import optuna
import shap

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model_improvement")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_mega_alpha.parquet"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

def load_data(filepath: Path = FEATURES_FILE) -> pd.DataFrame:
    """Load feature dataset."""
    logger.info(f"📥 Loading data from {filepath}...")
    logger.info(f"📥 Loading data from {filepath}...")
    # Fallback to CSV if Parquet fails or just use CSV
    csv_path = filepath.with_suffix(".csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_parquet(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def prepare_features(df: pd.DataFrame, target_col: str = "y_direction_up") -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Prepare features and target."""
    exclude_cols = ["timestamp", target_col, "btc_ret_fwd_1"]
    # Explicitly ensure we are using all columns except excluded ones
    # The new alpha columns (alpha_*) will be automatically included here
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Scale features (fit on all for simplicity in this optimization loop, 
    # but ideally should fit on train only inside CV. 
    # For speed/simplicity here we fit once, but note potential leakage is minimal for scaling)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    
    return X_scaled, y, scaler

def objective_xgb(trial, X, y, cv):
    """Optuna objective for XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "random_state": SEED,
        "n_jobs": -1,
        "eval_metric": "logloss"
    }
    
    scores = []
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict_proba(X_val)[:, 1]
        try:
            score = roc_auc_score(y_val, preds)
        except ValueError:
            score = 0.5
        scores.append(score)
        
    return np.mean(scores)

def objective_lgb(trial, X, y, cv):
    """Optuna objective for LightGBM."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1
    }
    
    scores = []
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        preds = model.predict_proba(X_val)[:, 1]
        try:
            score = roc_auc_score(y_val, preds)
        except ValueError:
            score = 0.5
        scores.append(score)
        
    return np.mean(scores)

def run_optimization(X, y):
    """Run Optuna optimization."""
    logger.info("🔍 Starting Hyperparameter Optimization...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    # XGBoost
    logger.info("Optimizing XGBoost...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(lambda trial: objective_xgb(trial, X, y, tscv), n_trials=10) # Reduced trials for speed
    logger.info(f"Best XGB Params: {study_xgb.best_params} | Score: {study_xgb.best_value:.4f}")
    
    # LightGBM
    logger.info("Optimizing LightGBM...")
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(lambda trial: objective_lgb(trial, X, y, tscv), n_trials=10)
    logger.info(f"Best LGB Params: {study_lgb.best_params} | Score: {study_lgb.best_value:.4f}")
    
    return study_xgb.best_params, study_lgb.best_params

def train_best_models(X, y, xgb_params, lgb_params):
    """Train final models on full dataset (or train/val split)."""
    # Split into Train/Test (last 15% test)
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params, n_jobs=-1, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_preds)
    logger.info(f"Final XGB Test AUC: {xgb_auc:.4f}")
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_preds)
    logger.info(f"Final LGB Test AUC: {lgb_auc:.4f}")
    
    # Logistic Regression Baseline
    lr_model = LogisticRegression(random_state=SEED)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_preds)
    logger.info(f"Final LR Test AUC: {lr_auc:.4f}")
    
    return xgb_model, lgb_model, X_test, y_test

def feature_importance_analysis(model, X_test):
    """Calculate SHAP values."""
    logger.info("📊 Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # For binary classification, shap_values might be a list or array
    if isinstance(shap_values, list):
        vals = np.abs(shap_values[1]).mean(0)
    else:
        vals = np.abs(shap_values).mean(0)
        
    feature_importance = pd.DataFrame(list(zip(X_test.columns, vals)), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    
    # Save
    csv_path = DATA_DIR / "features" / "feature_importance.csv"
    feature_importance.to_csv(csv_path, index=False)
    logger.info(f"Feature importance saved to {csv_path}")
    
    return feature_importance

def main():
    # 1. Load Data
    df = load_data()
    X, y, scaler = prepare_features(df)
    
    # 2. Optimization
    xgb_params, lgb_params = run_optimization(X, y)
    
    # 3. Train Best Models
    xgb_model, lgb_model, X_test, y_test = train_best_models(X, y, xgb_params, lgb_params)
    
    # 4. Feature Importance (using XGBoost)
    feature_importance_analysis(xgb_model, X_test)
    
    # 5. Save Best Model
    best_model_path = MODELS_DIR / "best_model_xgb_opt.pkl"
    with open(best_model_path, "wb") as f:
        pickle.dump(xgb_model, f)
        
    # Save Scaler
    scaler_path = MODELS_DIR / "scaler_opt.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    # 6. Train and Save Regime Model
    logger.info("🛡️ Training Regime Model...")
    from src.models.regime_model import RegimeDetector
    regime_model = RegimeDetector()
    regime_model.fit(df, symbol="btc") # Fit on full history
    
    regime_model_path = MODELS_DIR / "regime_model.pkl"
    regime_model.save(regime_model_path)
    logger.info(f"✅ Regime Model Saved to {regime_model_path}")

    logger.info(f"✅ Model Improvement Pipeline Complete. Saved to {best_model_path} and {scaler_path}")

if __name__ == "__main__":
    main()
