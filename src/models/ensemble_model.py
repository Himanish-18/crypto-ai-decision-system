import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
# import catboost as cb
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger("ensemble_model")


class EnsembleModel:
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=6
        )
        self.lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05)
        # self.cb_model = cb.CatBoostClassifier(iterations=100, learning_rate=0.05, depth=6, verbose=False)
        self.meta_weights = {"xgb": 0.5, "lgb": 0.5, "cb": 0.0}

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train all base models.
        """
        logger.info("Training XGBoost...")
        self.xgb_model.fit(X, y)

        logger.info("Training LightGBM...")
        self.lgb_model.fit(X, y)

        # logger.info("Training CatBoost...")
        # self.cb_model.fit(X, y)

        logger.info("Ensemble training complete.")

    def predict_proba(self, X: pd.DataFrame) -> float:
        """
        Return weighted average probability of class 1 (Buy).
        """
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        lgb_pred = self.lgb_model.predict_proba(X)[:, 1]
        # cb_pred = self.cb_model.predict_proba(X)[:, 1]

        ensemble_pred = (
            self.meta_weights["xgb"] * xgb_pred
            + self.meta_weights["lgb"] * lgb_pred
            # self.meta_weights['cb'] * cb_pred
        )
        return ensemble_pred

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)
