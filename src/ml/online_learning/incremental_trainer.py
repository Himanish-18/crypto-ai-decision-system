
import numpy as np
import logging

# v39 Incremental Online Learning
# Manages model drift by warm-starting updates on new data windows.
# Supports partial_fit for linear models and booster updates for trees.

class IncrementalTrainer:
    def __init__(self, base_model, window_size=1000, drift_threshold=0.05):
        self.model = base_model
        self.buffer_X = []
        self.buffer_y = []
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.logger = logging.getLogger("online_ml")
        
    def stream_update(self, x_new, y_new):
        """Buffer new data and trigger update if window is full."""
        self.buffer_X.append(x_new)
        self.buffer_y.append(y_new)
        
        if len(self.buffer_X) >= self.window_size:
            self._update_model()
            self.buffer_X = []
            self.buffer_y = []
            
    def _update_model(self):
        X = np.array(self.buffer_X)
        y = np.array(self.buffer_y)
        
        # Check drift (Simple label mean shift for demonstration)
        mean_y = np.mean(y)
        if abs(mean_y - 0.5) > self.drift_threshold:
             self.logger.info(f"🔄 Concept Drift Detected (Mean Y: {mean_y:.2f}). Retraining...")
        else:
             # Standard update
             pass

        if hasattr(self.model, "partial_fit"):
            # SGD / Neural Nets
            self.model.partial_fit(X, y)
            self.logger.info("✅ Online Update: Partial Fit Complete.")
        elif hasattr(self.model, "update"):
            # Custom XGB/LGB updater
             self.model.update(X, y)
             self.logger.info("✅ Online Update: Booster Refit Complete.")
        else:
            self.logger.warning("⚠️ Model does not support online updates.")
