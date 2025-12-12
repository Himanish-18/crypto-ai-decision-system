import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("feature_pruning")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features" / "alpha_features.parquet"
if not FEATURES_FILE.exists():
    FEATURES_FILE = DATA_DIR / "features" / "features_1H_mega_alpha.parquet"

MODEL_OUT = DATA_DIR / "models" / "optimized_model.pkl"
SCALER_OUT = DATA_DIR / "models" / "optimized_scaler.pkl"
FEATURES_OUT = DATA_DIR / "models" / "selected_alpha_features.json"
METRICS_OUT = DATA_DIR / "models" / "model_metrics_optimized.json"

TARGET_COL = "y_direction_up"


def load_and_clean_data():
    logger.info(f"Loading data from {FEATURES_FILE}...")
    df = pd.read_parquet(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Basic Cleaning
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def get_initial_features(df):
    exclude_cols = [
        "timestamp",
        TARGET_COL,
        "btc_ret_fwd_1",
        "y_pred",
        "y_prob",
        "signal_prob",
        "regime",
        "symbol",
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


def correlation_pruning(df, feature_cols, threshold=0.85):
    logger.info("✂️ Running Correlation Pruning...")
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    selected = [f for f in feature_cols if f not in to_drop]
    logger.info(
        f"Dropped {len(to_drop)} features due to correlation > {threshold}. Remaining: {len(selected)}"
    )
    return selected


def shap_pruning(df, feature_cols, target, threshold=0.001):
    logger.info("✂️ Running SHAP Pruning...")

    # Use a subset for speed if large
    if len(df) > 10000:
        sample_df = df.iloc[-10000:]
    else:
        sample_df = df

    X = sample_df[feature_cols]
    y = sample_df[target]

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP value
    if isinstance(shap_values, list):  # Multiclass
        shap_abs = np.abs(shap_values[1]).mean(axis=0)
    else:
        shap_abs = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.DataFrame(
        list(zip(feature_cols, shap_abs)), columns=["feature", "shap_value"]
    )
    feature_importance = feature_importance.sort_values("shap_value", ascending=False)

    selected_df = feature_importance[feature_importance["shap_value"] > threshold]
    selected = selected_df["feature"].tolist()

    logger.info(f"Dropped features with SHAP < {threshold}. Remaining: {len(selected)}")
    logger.info(f"Top 5 Features: {selected[:5]}")

    return selected


def objective_xgb(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_jobs": -1,
        "random_state": 42,
    }

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

    return np.mean(scores)


def objective_lgb(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

    return np.mean(scores)


def run_optimization(df, feature_cols):
    X = df[feature_cols]
    y = df[TARGET_COL]

    # Scale Data (StandardScaler is generally good for LR/NN, less critical for Trees but good for consistency)
    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_np, columns=feature_cols)

    logger.info("🔎 Tuning XGBoost...")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(
        lambda trial: objective_xgb(trial, X_scaled, y), n_trials=20
    )  # 20 trials for speed
    best_xgb = study_xgb.best_params

    logger.info("🔎 Tuning LightGBM...")
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(lambda trial: objective_lgb(trial, X_scaled, y), n_trials=20)
    best_lgb = study_lgb.best_params

    return best_xgb, best_lgb, scaler


def train_final_ensemble(df, feature_cols, best_xgb_params, best_lgb_params, scaler):
    X = df[feature_cols]
    y = df[TARGET_COL]

    # Use the fitted scaler from optimization phase or refit? Refit on full data.
    X_scaled_np = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_np, columns=feature_cols, index=df.index)

    # We need to construct the Stacking manually to populate the AlphaEnsemble object
    # But wait, AlphaEnsemble class structure is fixed.
    # We will instantiate it, BUT we will manually inject the fitted models.

    from src.models.alpha_ensemble import AlphaEnsemble

    ensemble = AlphaEnsemble()
    ensemble.scaler = scaler  # fitted

    # 1. Generate OOF Predictions for Meta Learner Training
    tscv = TimeSeriesSplit(n_splits=5)
    meta_X = []
    meta_y = []

    logger.info("🏋️ Training Stacked Ensemble (Step 1: Meta-Features)...")

    for train_idx, val_idx in tscv.split(X_scaled):
        X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # XGB
        xgb_m = xgb.XGBClassifier(**best_xgb_params, n_jobs=-1, random_state=42)
        xgb_m.fit(X_train, y_train)
        p_xgb = xgb_m.predict_proba(X_val)[:, 1]

        # LGB
        lgb_m = lgb.LGBMClassifier(
            **best_lgb_params, verbose=-1, n_jobs=-1, random_state=42
        )
        lgb_m.fit(X_train, y_train)
        p_lgb = lgb_m.predict_proba(X_val)[:, 1]

        # LR (Baseline)
        lr_m = LogisticRegression()
        lr_m.fit(X_train, y_train)
        p_lr = lr_m.predict_proba(X_val)[:, 1]

        meta_df = pd.DataFrame(
            {"xgb": p_xgb, "lgb": p_lgb, "lr": p_lr}, index=X_val.index
        )
        meta_X.append(meta_df)
        meta_y.append(y_val)

    meta_X = pd.concat(meta_X)
    meta_y = pd.concat(meta_y)

    # Train Meta Learner
    ensemble.meta_model.fit(meta_X, meta_y)
    coefs = dict(zip(meta_X.columns, ensemble.meta_model.coef_[0]))
    logger.info(f"Meta Model Coefficients: {coefs}")

    # Calc Metrics
    auc = roc_auc_score(meta_y, ensemble.meta_model.predict_proba(meta_X)[:, 1])
    logger.info(f"OOF AUC: {auc:.4f}")

    # 2. Retrain Base Models on Full Data
    logger.info("🏋️ Retraining Base Models on Full Data...")
    ensemble.models["xgb"] = xgb.XGBClassifier(
        **best_xgb_params, n_jobs=-1, random_state=42
    ).fit(X_scaled, y)
    ensemble.models["lgb"] = lgb.LGBMClassifier(
        **best_lgb_params, verbose=-1, n_jobs=-1, random_state=42
    ).fit(X_scaled, y)
    ensemble.models["lr"] = LogisticRegression().fit(X_scaled, y)

    # We also need to hack the AlphaEnsemble to support our custom model list if it expects specific keys
    # The default class has keys: "xgb", "lgb", "lr", "alpha_pure".
    # We are omitting "alpha_pure" for brevity or we can add it back.
    # Let's simple keep these 3.

    return ensemble, auc, coefs


def main():
    df = load_and_clean_data()

    # 1. Feature Selection
    initial_features = get_initial_features(df)
    logger.info(f"Initial Feature Count: {len(initial_features)}")

    corr_vars = correlation_pruning(df, initial_features)
    final_vars = shap_pruning(df, corr_vars, TARGET_COL)

    # 2. Optimization
    best_xgb, best_lgb, scaler = run_optimization(df, final_vars)
    logger.info(f"Best XGB: {best_xgb}")
    logger.info(f"Best LGB: {best_lgb}")

    # 3. Final Training
    ensemble, final_auc, coefs = train_final_ensemble(
        df, final_vars, best_xgb, best_lgb, scaler
    )

    # 4. Save
    logger.info("💾 Saving Artifacts...")

    # Save Model (as MultiFactorModel? Or just the AlphaEnsemble?)
    # The system uses MultiFactorModel which wraps AlphaEnsemble.
    # We should probably save it as MultiFactorModel to be safe, creating a new instance.
    from src.models.multifactor_model import MultiFactorModel

    mf_model = MultiFactorModel()
    mf_model.stacking_model = ensemble  # Injection

    mf_model.save(MODEL_OUT)  # Saves to optimized_model.pkl

    # Save Scaler (Ensemble has it, but saving separately as requested)
    with open(SCALER_OUT, "wb") as f:
        pickle.dump(scaler, f)

    # Save Feature List
    with open(FEATURES_OUT, "w") as f:
        json.dump(final_vars, f, indent=4)

    # Save Metrics
    metrics = {
        "auc": final_auc,
        "n_features": len(final_vars),
        "best_xgb_params": best_xgb,
        "best_lgb_params": best_lgb,
        "meta_coefs": coefs,
        "top_features": final_vars[:20] if len(final_vars) > 20 else final_vars,
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("✅ Optimization Complete.")


if __name__ == "__main__":
    main()
