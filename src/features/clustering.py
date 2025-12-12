import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

logger = logging.getLogger("feature_clustering")


class FeatureClustering:
    """
    v24 Feature Redundancy Remover.
    Uses Hierarchical Clustering on Spearman Correlation to group similar features.
    Selects one representative feature per cluster.
    """

    def __init__(self, correlation_threshold: float = 0.8):
        self.threshold = correlation_threshold
        self.selected_features = []
        self.clusters = {}

    def fit(self, X: pd.DataFrame):
        """
        Identify clusters and select features.
        """
        if X.empty:
            logger.warning("Empty dataframe for clustering.")
            return

        # 1. Compute Correlation Matrix (Spearman for non-linear)
        corr_matrix = X.corr(method="spearman").abs()
        corr_matrix = corr_matrix.fillna(0)

        # 2. Distance Matrix
        # distance = 1 - correlation
        distances = 1 - corr_matrix.values
        # Ensure symmetric and positive (numerical noise fix)
        distances = np.clip(distances, 0, 2)
        np.fill_diagonal(distances, 0)

        # Squareform for linkage
        dist_linkage = squareform(distances)

        # 3. Hierarchical Clustering (Ward)
        linkage_matrix = hierarchy.watch_linkage = hierarchy.linkage(
            dist_linkage, method="ward"
        )

        # 4. Form Clusters
        # Apply threshold distance. distance=1-corr. threshold=0.2 means corr>0.8
        # Ward distance scale is different, but let's using 'distance' criterion with t = 1 - threshold?
        # Actually for Ward, it's variance.
        # Let's use 'average' linkage for direct correlation threshold mapping logic or just use fcluster
        # Re-doing linkage with 'average' matches correlation distance intuition better.

        linkage_matrix = hierarchy.linkage(dist_linkage, method="average")
        cluster_labels = hierarchy.fcluster(
            linkage_matrix, t=(1 - self.threshold), criterion="distance"
        )

        # 5. Select Representatives
        self.clusters = {}  # ID -> [features]
        for i, label in enumerate(cluster_labels):
            feat = X.columns[i]
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(feat)

        self.selected_features = []
        for label, feats in self.clusters.items():
            # Strategy: Pick feature with highest variance (proxy for signal strength)?
            # Or just first one? Or one most correlated to target (if provided)?
            # Here: Pick first one for stability.
            # Improvement: Pick one closest to cluster centroid.
            # Stub: First
            best_feat = feats[0]
            self.selected_features.append(best_feat)

        logger.info(
            f"Feature Clustering: Reduced {len(X.columns)} -> {len(self.selected_features)} features."
        )
        logger.info(
            f"Dropped {len(X.columns) - len(self.selected_features)} redundant features."
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return reduced feature set.
        """
        if not self.selected_features:
            return X
        return X[self.selected_features]
