
import optuna
import lightgbm as lgb
import joblib
import numpy as np
import logging

# v39 AutoML Hyperparameter Search
# Tuning LightGBM for institutional alpha using Optuna Bayesian Optimization.

class AutoMLSearch:
    def __init__(self, objective="binary"):
        self.objective = objective
        self.best_params = None
        self.logger = logging.getLogger("automl")
        
    def optimize(self, X, y, n_trials=20):
        def objective(trial):
            param = {
                'objective': self.objective,
                'metric': 'binary_logloss',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            }

            train_data = lgb.Dataset(X, label=y)
            # Simple 3-fold CV inside trial
            cv_results = lgb.cv(param, train_data, nfold=3, stratified=True, callbacks=[lgb.early_stopping(stopping_rounds=10)])
            
            return cv_results['valid binary_logloss-mean'][-1]

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.logger.info(f"AutoML Best Params: {self.best_params}")
        return self.best_params

    def train_final(self, X, y):
        if not self.best_params:
            raise ValueError("Run optimize() first.")
            
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(self.best_params, train_data)
        return model

if __name__ == "__main__":
    # Test
    automl = AutoMLSearch()
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    automl.optimize(X, y, n_trials=2)
    print("AutoML Test Passed")
