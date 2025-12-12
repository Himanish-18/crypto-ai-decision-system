
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple

class AlphaDiscoveryEngine:
    """
    Unsupervised learning for Feature Discovery.
    - Clustering: Groups correlated features to reduce redundancy (collinearity).
    - Stability: Tracks feature-target correlation over time.
    """
    
    def __init__(self, correlation_threshold: float = 0.9):
        self.corr_thresh = correlation_threshold
        
    def cluster_features(self, feature_matrix: pd.DataFrame) -> Dict[int, List[str]]:
        """
        Uses DBSCAN on correlation distance matrix to find groups of similar features.
        Distance = 1 - |Correlation|
        """
        corr_matrix = feature_matrix.corr().abs()
        distance_matrix = 1.0 - corr_matrix
        
        # DBSCAN: epsilon corresponds to distance threshold
        # if dist < (1 - 0.9) = 0.1, then features are > 0.9 correlated
        eps = 1.0 - self.corr_thresh
        
        db = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
        labels = db.fit_predict(distance_matrix)
        
        clusters = {}
        for idx, label in enumerate(labels):
            feat_name = feature_matrix.columns[idx]
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(feat_name)
            
        return clusters

    def check_feature_stability(self, feature: pd.Series, target: pd.Series, window: int = 100) -> float:
        """
        Calculates stability of rolling correlation.
        Returns 1.0 (perfectly stable) to 0.0 (random walk).
        Stability = 1 - StdDev(RollingCorr)/2 (approx)
        """
        if len(feature) < window: return 0.0
        
        rolling_corr = feature.rolling(window).corr(target)
        stability_score = 1.0 - rolling_corr.std()
        
        return max(0.0, stability_score)
