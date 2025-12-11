import unittest
import pandas as pd
import numpy as np
import logging
from src.ml.residual_model import ResidualReturnModel
from src.ml.meta_failure_v2 import MetaFailureModel
from src.ml.calibration import ProbabilityCalibration
from src.features.clustering import FeatureClustering
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO)

class TestV24Alpha(unittest.TestCase):
    
    def test_residual_model(self):
        # Create correlated data
        np.random.seed(42)
        n = 100
        market = np.random.normal(0.001, 0.01, n)
        # Asset = 1.5 * Market + Alpha
        alpha = np.random.normal(0, 0.005, n)
        asset = 1.5 * market + alpha
        
        model = ResidualReturnModel(use_pca_factors=False)
        residuals = model.fit_transform(pd.Series(asset), pd.Series(market))
        
        # Check Beta
        self.assertAlmostEqual(model.betas['Market'], 1.5, delta=0.1)
        # Check Residual = Alpha
        correlation = np.corrcoef(residuals, alpha)[0,1]
        self.assertGreater(correlation, 0.95)
        
    def test_meta_failure(self):
        model = MetaFailureModel()
        # Mock Trades
        df = pd.DataFrame({
            'volatility': np.random.rand(50),
            'spread': np.random.rand(50),
            'imbalance': np.random.rand(50), # Added missing feature
            'regime_score': np.random.randint(0, 3, 50),
            'model_confidence': np.random.rand(50),
            'pnl': np.random.choice([-0.01, 0.01], 50)
        })
        model.train(df)
        self.assertTrue(model.is_fitted)
        
        prob = model.predict_failure_proba({'volatility': 0.5})
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        
    def test_calibration(self):
        cal = ProbabilityCalibration(method='sigmoid')
        base = LogisticRegression()
        X = np.random.rand(100, 2)
        y = (X[:,0] + X[:,1] > 1).astype(int)
        
        base.fit(X[:50], y[:50])
        # Calibrate on validation
        cal_model = cal.calibrate(base, X[50:], y[50:])
        
        preds = cal_model.predict_proba(X[:5])
        self.assertEqual(preds.shape, (5, 2))
        
    def test_clustering(self):
        # Create redundant features
        df = pd.DataFrame(np.random.rand(50, 2), columns=['A', 'B'])
        # C is highly correlated to A
        df['C'] = df['A'] * 0.95 + np.random.normal(0, 0.01, 50)
        
        cluster = FeatureClustering(correlation_threshold=0.8)
        cluster.fit(df)
        
        selected = cluster.selected_features
        # distinct clusters: {A, C} and {B}
        # Should select 2 features total
        self.assertEqual(len(selected), 2)
        self.assertIn('B', selected)
        # Only one of A or C
        self.assertTrue('A' in selected or 'C' in selected)
        self.assertFalse('A' in selected and 'C' in selected)

if __name__ == '__main__':
    unittest.main()
