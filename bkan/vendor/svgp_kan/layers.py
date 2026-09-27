"""
layers.py - SVGP-KAN Layer implementation (v0.4.4)

Each layer implements a Sparse Variational GP on each edge of a KAN,
enabling probabilistic inference with uncertainty propagation.

MODES:
- strict_mean_field=False (DEFAULT): Uses clamping heuristic for orthogonal variance.
  Backward compatible. Works well in practice even when approximation assumptions
  are not strictly satisfied.
  
- strict_mean_field=True: Adjusts num_inducing at initialization to ensure proper
  spacing relative to the initial lengthscale. Lengthscales may still adapt during
  training, with clamping used as a safety net. This provides a better starting
  point for the mean-field approximation.

UPDATE (v0.4.4): Toned down warning messages - clamping works well in practice.
"""

import torch
import torch.nn as nn
import warnings
from .kernels import (
    rbf_moment_matching, 
    cosine_moment_matching, 
    check_mean_field_assumptions,
    check_mean_field_validity,
    adjust_inducing_points_for_mean_field,
    compute_recommended_num_inducing
)


class GPKANLayer(nn.Module):
    """
    A single SVGP-KAN layer.
    
    Each edge (i,j) is an independent univariate GP with:
    - Inducing points z[i,j,:]
    - Variational mean q_mu[i,j,:]  
    - Variational variance exp(q_log_var[i,j,:])
    - Kernel lengthscale exp(log_scale[i,j])
    - Kernel signal variance exp(log_variance[i,j])
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        num_inducing: Number of inducing points per edge
        kernel_type: 'rbf' or 'cosine'
        strict_mean_field: If True, adjust num_inducing at initialization to ensure
            proper spacing. Lengthscales remain learnable. Default False (backward compatible).
        z_min: Minimum inducing point location (default: -1.5)
        z_max: Maximum inducing point location (default: 1.5)
        initial_lengthscale: Initial lengthscale for strict_mean_field check (default: 1.0)
    """
    
    def __init__(self, in_features, out_features, num_inducing=20, kernel_type='rbf',
                 strict_mean_field=False, z_min=-1.5, z_max=1.5, initial_lengthscale=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_type = kernel_type.lower()
        self.strict_mean_field = strict_mean_field
        self.z_min = z_min
        self.z_max = z_max
        
        # Store original request for reference
        self._requested_num_inducing = num_inducing
        
        # Adjust num_inducing if strict_mean_field is enabled
        if strict_mean_field:
            num_inducing, was_adjusted = adjust_inducing_points_for_mean_field(
                num_inducing=num_inducing,
                z_min=z_min,
                z_max=z_max,
                lengthscale=initial_lengthscale,
                min_separation_factor=2.0,
                verbose=True
            )
            self._was_adjusted = was_adjusted
        else:
            self._was_adjusted = False
        
        self.num_inducing = num_inducing
        
        # --- Learnable Parameters ---
        # Inducing points initialized uniformly in [z_min, z_max]
        self.z = nn.Parameter(
            torch.linspace(z_min, z_max, num_inducing)
            .reshape(1, 1, num_inducing)
            .repeat(out_features, in_features, 1)
        )
        
        # Variational parameters
        self.q_mu = nn.Parameter(torch.randn(out_features, in_features, num_inducing) * 0.05)
        self.q_log_var = nn.Parameter(torch.ones(out_features, in_features, num_inducing) * -3.0)
        
        # Kernel hyperparameters (in log space for positivity)
        self.log_scale = nn.Parameter(torch.zeros(out_features, in_features))
        self.log_variance = nn.Parameter(torch.zeros(out_features, in_features) - 1.0)
        
        # Diagnostic tracking
        self._last_diagnostics = None
        self._mean_field_warning_issued = False

    def forward(self, x, return_diagnostics=False):
        """
        Forward pass with uncertainty propagation.
        
        Args:
            x: Either a tensor (deterministic input) or tuple (mean, var)
            return_diagnostics: If True, also return approximation diagnostics
            
        Returns:
            (y_mean, y_var): Output mean and variance tensors
            diagnostics: (optional) Dict with approximation quality metrics
        """
        # --- 1. Input Parsing ---
        if isinstance(x, torch.Tensor):
            x_mean, x_var = x, torch.zeros_like(x) + 1e-6
        else:
            x_mean, x_var = x
            x_var = torch.clamp(x_var, min=1e-6)

        original_shape = x_mean.shape
        ndim = x_mean.ndim

        # --- 2. Universal Shape Handling ---
        if ndim == 2:
            x_mean_flat = x_mean
            x_var_flat = x_var
        elif ndim == 4:
            # 4D Vision Support [B, C, H, W]
            b, c, h, w = original_shape
            if c != self.in_features: 
                raise ValueError(f"Channel mismatch: {c} vs {self.in_features}")
            x_mean_flat = x_mean.permute(0, 2, 3, 1).contiguous().view(-1, c)
            x_var_flat = x_var.permute(0, 2, 3, 1).contiguous().view(-1, c)
        else:
            x_mean_flat = x_mean.reshape(-1, self.in_features)
            x_var_flat = x_var.reshape(-1, self.in_features)

        # --- 3. Core GP Logic ---
        constrained_q_var = torch.exp(self.q_log_var).clamp(min=1e-5)
        constrained_var = torch.exp(self.log_variance).clamp(min=1e-5)
        
        min_scale = 0.1 if self.kernel_type == 'rbf' else 0.5
        constrained_scale = torch.exp(self.log_scale).clamp(min=min_scale)

        # Check mean-field validity if strict mode and issue info once
        if self.strict_mean_field and not self._mean_field_warning_issued:
            is_valid, min_ratio, _ = check_mean_field_validity(self.z, constrained_scale)
            if not is_valid:
                # Toned-down message - this is expected and handled
                print(
                    f"  [info] Layer using clamped approximation (separation ratio = {min_ratio:.2f})"
                )
                self._mean_field_warning_issued = True

        if self.kernel_type == 'rbf':
            result = rbf_moment_matching(
                x_mean_flat, x_var_flat, self.z, self.q_mu, constrained_q_var, 
                constrained_scale, constrained_var,
                return_diagnostics=return_diagnostics
            )
        elif self.kernel_type == 'cosine':
            result = cosine_moment_matching(
                x_mean_flat, x_var_flat, self.z, self.q_mu, constrained_q_var, 
                constrained_scale, constrained_var,
                return_diagnostics=return_diagnostics
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        if return_diagnostics:
            edge_means, edge_vars, diagnostics = result
            self._last_diagnostics = diagnostics
        else:
            edge_means, edge_vars = result
            diagnostics = None

        # Aggregation across input dimensions
        y_mean_flat = edge_means.sum(dim=2) 
        y_var_flat = edge_vars.sum(dim=2)

        # --- 4. Output Reshaping ---
        if ndim == 2:
            y_mean, y_var = y_mean_flat, y_var_flat
        elif ndim == 4:
            y_mean = y_mean_flat.view(b, h, w, self.out_features).permute(0, 3, 1, 2)
            y_var = y_var_flat.view(b, h, w, self.out_features).permute(0, 3, 1, 2)
        else:
            out_shape = original_shape[:-1] + (self.out_features,)
            y_mean = y_mean_flat.reshape(out_shape)
            y_var = y_var_flat.reshape(out_shape)

        if return_diagnostics:
            return (y_mean, y_var), diagnostics
        return y_mean, y_var

    def compute_kl(self, jitter=1e-4):
        """
        Compute KL divergence KL[q(u)||p(u)] for this layer.
        
        This is the proper Bayesian regularization term for variational inference.
        
        Returns:
            kl: scalar KL divergence
        """
        try:
            from .kl_divergence import compute_kl_divergence
        except ImportError:
            warnings.warn(
                "kl_divergence module not found. KL term will be zero. "
                "To enable proper Bayesian inference, ensure kl_divergence.py is in the package."
            )
            return torch.tensor(0.0, device=self.q_mu.device)
        
        # Get constrained parameters (same as used in forward pass)
        constrained_var = torch.exp(self.log_variance).clamp(min=1e-5)
        min_scale = 0.1 if self.kernel_type == 'rbf' else 0.5
        constrained_scale = torch.exp(self.log_scale).clamp(min=min_scale)
        
        if self.kernel_type == 'rbf':
            kl = compute_kl_divergence(
                q_mu=self.q_mu,
                q_log_var=self.q_log_var,
                z=self.z,
                lengthscale=constrained_scale,
                variance=constrained_var,
                jitter=jitter
            )
        else:
            warnings.warn(f"KL divergence not implemented for kernel type '{self.kernel_type}'. Returning zero.")
            kl = torch.tensor(0.0, device=self.q_mu.device)
        
        return kl

    def get_relevance(self):
        """Returns the ARD variance magnitude for feature selection."""
        return torch.exp(self.log_variance)
    
    def check_approximation(self, x_var=None, verbose=True):
        """
        Check mean-field approximation status.
        
        This is informational - the clamping heuristic handles violations gracefully.
        
        Args:
            x_var: [Batch, In] input variances (optional)
            verbose: Print diagnostic messages
            
        Returns:
            dict with diagnostic information
        """
        constrained_var = torch.exp(self.log_variance).clamp(min=1e-5)
        min_scale = 0.1 if self.kernel_type == 'rbf' else 0.5
        constrained_scale = torch.exp(self.log_scale).clamp(min=min_scale)
        
        return check_mean_field_assumptions(
            self.z, constrained_scale, constrained_var, x_var, verbose
        )
    
    def get_last_diagnostics(self):
        """Return diagnostics from last forward pass (if computed)."""
        return self._last_diagnostics
    
    def get_recommended_num_inducing(self):
        """
        Get recommended number of inducing points based on current lengthscales.
        
        Returns:
            recommended: Recommended number of inducing points
            current: Current number of inducing points
        """
        min_scale = 0.1 if self.kernel_type == 'rbf' else 0.5
        constrained_scale = torch.exp(self.log_scale).clamp(min=min_scale)
        
        # Use minimum lengthscale across all edges
        min_lengthscale = constrained_scale.min().item()
        
        recommended = compute_recommended_num_inducing(
            self.z_min, self.z_max, min_lengthscale, min_separation_factor=2.0
        )
        
        return recommended, self.num_inducing
    
    def reset_mean_field_warning(self):
        """Reset the mean-field warning flag (useful for re-checking after adjustments)."""
        self._mean_field_warning_issued = False
