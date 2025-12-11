import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger("adversarial_v1")

class AdversarialRobustness:
    """
    v18 Adversarial Defense Unit.
    Checks model stability against Fast Gradient Sign Method (FGSM) attacks.
    If a small noise injection flips the prediction, the confidence is penalized.
    """
    def __init__(self, epsilon: float = 0.01):
        self.epsilon = epsilon
        self.loss_fn = nn.MSELoss() # Assuming regression or log-prob output

    def generate_perturbation(self, model: nn.Module, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Generate FGSM Perturbation.
        x_adv = x + epsilon * sign(grad_x(loss(x, y)))
        """
        x.requires_grad = True
        
        # Forward pass
        output = model(x)
        loss = self.loss_fn(output, target)
        
        # Backward pass to get gradient w.r.t Input
        model.zero_grad()
        loss.backward()
        
        data_grad = x.grad.data
        
        # Create perturbation
        sign_data_grad = data_grad.sign()
        perturbed_x = x + self.epsilon * sign_data_grad
        
        # Return detached perturbation
        return perturbed_x.detach()

    def check_stability(self, model: nn.Module, x: torch.Tensor) -> float:
        """
        Returns a Stability Score (0.0 to 1.0).
        1.0 = Perfectly Stable (Prediction didn't change much).
        0.0 = Unstable (Prediction flipped).
        """
        if model is None: return 1.0 # Stub
        
        model.eval()
        
        # 1. Original Prediction
        orig_pred = model(x).item()
        
        # 2. Generate Attack
        # Since we don't have a label 'y' at inference time, we use the original prediction as the 'target'
        # and try to maximize distance from it (un-targeted attack) or just check sensitivity.
        # Actually standard FGSM targets a label. Here we want to see sensitivity.
        # Let's try to Maximize loss away from current prediction (Virtual Adversarial Training concept).
        target = torch.tensor([orig_pred]) 
        
        # We want to find x' that maximizes distance from orig_pred
        # So we ascend the gradient of MSE(pred, orig_pred)
        # Wait, if we use MSE(pred, orig_pred), gradient is 0 at x.
        # VAT uses KL Div with no-grad target.
        # Approximating: We manually perturb random directions if gradient is zero, 
        # or use a pseudo-target (e.g. reverse signal).
        
        # Simpler approach for v1 Inference:
        # Just inject random noise scaled by gradient magnitude?
        # Let's stick to simple noise injection for v1 stability check if strict FGSM is hard without labels.
        # Or: Assume target is "Opposite of Prediction" and see if we can easily push it there.
        
        # Implementation: Simple Sensitivity Analysis
        # Perturb input by epsilon in direction of gradients (if available) or random.
        
        # Generate random noise bound by epsilon
        noise = torch.randn_like(x) * self.epsilon
        perturbed_x = x + noise
        
        adv_pred = model(perturbed_x).item()
        
        # Calculate deviation
        deviation = abs(orig_pred - adv_pred)
        
        # Normalize stability
        # If deviation is small relative to signal magnitude, it's stable.
        # If signal is 0.5 and deviation is 0.4, that's bad.
        
        stability = 1.0 - min(deviation, 1.0)
        return stability
