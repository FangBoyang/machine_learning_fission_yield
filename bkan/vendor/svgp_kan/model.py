"""
model.py - GPKAN model implementation

Stacks multiple GPKANLayers to form a deep probabilistic network.
"""

import torch
import torch.nn as nn
from .layers import GPKANLayer


class GPKAN(nn.Module):
    """
    The core Neural Network class.
    Stacks multiple GPKANLayers to form a deep probabilistic network.
    
    Args:
        layers_hidden: List of layer sizes, e.g., [2, 10, 1] for 2 inputs, 
                       10 hidden units, 1 output
        num_inducing: Number of inducing points per edge
        kernel_type: 'rbf' or 'cosine'
        strict_mean_field: If True, enforce mean-field approximation assumptions
            by adjusting num_inducing if necessary. Default False (backward compatible).
        z_min: Minimum inducing point location (default: -1.5)
        z_max: Maximum inducing point location (default: 1.5)
        initial_lengthscale: Initial lengthscale for strict_mean_field check (default: 1.0)
    """
    def __init__(self, layers_hidden, num_inducing=20, kernel_type='rbf',
                 strict_mean_field=False, z_min=-1.5, z_max=1.5, initial_lengthscale=1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.strict_mean_field = strict_mean_field
        
        # Build layer stack
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                GPKANLayer(
                    in_features=layers_hidden[i], 
                    out_features=layers_hidden[i+1], 
                    num_inducing=num_inducing, 
                    kernel_type=kernel_type,
                    strict_mean_field=strict_mean_field,
                    z_min=z_min,
                    z_max=z_max,
                    initial_lengthscale=initial_lengthscale
                )
            )

    def forward(self, x):
        state = x
        for layer in self.layers:
            state = layer(state)
        # Returns tuple (Mean, Variance)
        return state
    
    def compute_total_kl(self, jitter=1e-4):
        """
        Compute total KL divergence across all layers.
        
        This is the KL[q(u)||p(u)] term in the ELBO for variational inference.
        
        Returns:
            total_kl: scalar, sum of KL divergences from all layers
        """
        total_kl = 0.0
        for layer in self.layers:
            layer_kl = layer.compute_kl(jitter=jitter)
            total_kl = total_kl + layer_kl
        return total_kl
    
    def check_all_approximations(self, verbose=True):
        """
        Check mean-field approximation assumptions for all layers.
        
        Returns:
            list of diagnostic dicts, one per layer
        """
        results = []
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f"\n--- Layer {i} ---")
            results.append(layer.check_approximation(verbose=verbose))
        return results


def gaussian_nll_loss(pred_mean, pred_var, target):
    """
    Negative Log Likelihood Loss for Gaussian distributions.
    Optimizes for both accuracy (Mean) and calibration (Variance).
    
    This is the data fidelity term (reconstruction term) in the ELBO.
    """
    # Ensure positive variance
    pred_var = torch.clamp(pred_var, min=1e-6)
    
    # Loss = 0.5 * ( log(var) + (y-mu)^2 / var )
    loss_term = 0.5 * (torch.log(pred_var) + (target - pred_mean).pow(2) / pred_var)
    
    return loss_term.mean()
