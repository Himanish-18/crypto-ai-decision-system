import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional

class UncertaintyEngine:
    """
    v18 Uncertainty Quantification Engine.
    Uses Deep Ensembles or Monte Carlo Dropout to estimate predictive uncertainty.
    
    Returns:
    - Prediction Mean (The Signal)
    - Epistemic Uncertainty (Model Ignorance/Lack of Data)
    - Aleatoric Uncertainty (Inherent Noise)
    """
    def __init__(self, model: nn.Module = None, dropout_rate: float = 0.1, n_passes: int = 10):
        self.model = model
        self.dropout_rate = dropout_rate
        self.n_passes = n_passes
        
    def enable_dropout(self, model):
        """ Force dropout layers to be active even during inference """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[float, float, float]:
        """
        Perform MC Dropout Inference.
        
        Args:
            x: Input tensor (batch_size=1, features)
            
        Returns:
            (mean_pred, epistemic_var, aleatoric_var)
        """
        if self.model is None:
            # Stub for when no model is loaded (Simulation/Fallback)
            return 0.0, 0.0, 0.0
            
        self.enable_dropout(self.model)
        
        preds = []
        with torch.no_grad():
            for _ in range(self.n_passes):
                # Forward pass with dropout active
                out = self.model(x)
                preds.append(out.item())
                
        preds = np.array(preds)
        
        # Mean Prediction
        mean_pred = np.mean(preds)
        
        # Epistemic Uncertainty = Variance of the means (Model disagreement)
        epistemic_var = np.var(preds)
        
        # Aleatoric Uncertainty 
        # (Strictly speaking, requires a second output head for LogVar. 
        # For this v18 implementation, we approximate or assume constant if not modeled)
        # We will assume the model outputs a single value for now. 
        # In v19 we can add heteroscedastic loss.
        aleatoric_var = 0.0 # Placeholder
        
        return float(mean_pred), float(epistemic_var), float(aleatoric_var)

    def is_confident(self, epistemic_var: float, threshold: float = 0.05) -> bool:
        """
        Veto Logic: If uncertainty is too high, don't trade.
        """
        return epistemic_var < threshold
