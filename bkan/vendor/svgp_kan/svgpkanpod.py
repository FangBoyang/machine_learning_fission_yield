"""
SVGPKanPOD: Flexible Probabilistic Neural Operator via SVGP-KANs.

A novel architecture for Reduced-Order Modeling (ROM) that replaces 
Discrete Matrix Factorization (POD/SVD) with Continuous Operator Learning.

Key Features:
    - Flexible input routing: User controls what each network sees
    - Principled uncertainty quantification via SVGP
    - Built-in loss function with KL annealing support
    - Easy access to latent outputs for interpretability/symbolic regression
    - Optional strict_mean_field mode for enforcing approximation assumptions

Topology (Branch-Trunk):
    u(branch_inputs, trunk_inputs) = Sum_k [ Branch_k(branch_inputs) * Trunk_k(trunk_inputs) ]

Usage:
    # Default mode (backward compatible, uses clamping heuristic)
    pod = SVGPKanPOD(branch_input_dim=2, trunk_input_dim=3, latent_dim=2)
    
    # Strict mode (enforces mean-field assumptions, may reduce num_inducing)
    pod = SVGPKanPOD(branch_input_dim=2, trunk_input_dim=3, latent_dim=2,
                     strict_mean_field=True)

    # Decomposed uncertainty
    result = model.predict_with_uncertainty(branch_in, trunk_in)
    print(f"Epistemic (model): {result['epistemic_var'].mean():.4f}")
    print(f"Aleatoric (noise): {result['aleatoric_var'].mean():.4f}")
"""


import torch
import torch.nn as nn
from .model import GPKAN, gaussian_nll_loss


class SVGPKanPOD(nn.Module):
    """
    Flexible Branch-Trunk SVGP-KAN for Probabilistic Operator Learning.
    
    Args:
        branch_input_dim: Dimension of branch network inputs
        trunk_input_dim: Dimension of trunk network inputs
        latent_dim: Number of latent dimensions (basis functions)
        branch_hidden: Hidden layer sizes for branch network
        trunk_hidden: Hidden layer sizes for trunk network
        num_inducing: Number of inducing points for SVGP
        kernel: Kernel type ('rbf', 'cosine')
        learn_noise: Whether to learn observation noise variance
        strict_mean_field: If True, enforce mean-field approximation assumptions
            by adjusting num_inducing if necessary. Default False (backward compatible).
        z_min: Minimum inducing point location (default: -1.5)
        z_max: Maximum inducing point location (default: 1.5)
        initial_lengthscale: Initial lengthscale for strict_mean_field check (default: 1.0)
        
    Attributes:
        branch (GPKAN): Maps branch_inputs -> latent coefficients
        trunk (GPKAN): Maps trunk_inputs -> basis functions
        log_noise: Log observation noise variance (if learn_noise=True)
    """
    
    def __init__(self, 
                 branch_input_dim, 
                 trunk_input_dim, 
                 latent_dim=10,
                 branch_hidden=None,
                 trunk_hidden=None,
                 num_inducing=50,
                 kernel='rbf',
                 learn_noise=True,
                 strict_mean_field=False,
                 z_min=-1.5,
                 z_max=1.5,
                 initial_lengthscale=1.0):
        super().__init__()
        
        # Default hidden layers - single layer like original
        if branch_hidden is None:
            branch_hidden = [32]
        if trunk_hidden is None:
            trunk_hidden = [32]
        
        self.latent_dim = latent_dim
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.strict_mean_field = strict_mean_field
        
        # --- Branch Net ---
        # Maps branch_inputs -> Coefficients
        self.branch = GPKAN(
            layers_hidden=[branch_input_dim] + list(branch_hidden) + [latent_dim],
            num_inducing=num_inducing,
            kernel_type=kernel,
            strict_mean_field=strict_mean_field,
            z_min=z_min,
            z_max=z_max,
            initial_lengthscale=initial_lengthscale
        )
        
        # --- Trunk Net ---
        # Maps trunk_inputs -> Basis functions
        self.trunk = GPKAN(
            layers_hidden=[trunk_input_dim] + list(trunk_hidden) + [latent_dim],
            num_inducing=num_inducing,
            kernel_type=kernel,
            strict_mean_field=strict_mean_field,
            z_min=z_min,
            z_max=z_max,
            initial_lengthscale=initial_lengthscale
        )
        
        # --- Observation Noise ---
        if learn_noise:
            self.log_noise = nn.Parameter(torch.tensor(-3.0))
        else:
            self.register_buffer('log_noise', torch.tensor(-3.0))
        
        # Store config for serialization
        self._config = {
            'branch_input_dim': branch_input_dim,
            'trunk_input_dim': trunk_input_dim,
            'latent_dim': latent_dim,
            'branch_hidden': branch_hidden,
            'trunk_hidden': trunk_hidden,
            'num_inducing': num_inducing,
            'kernel': kernel,
            'learn_noise': learn_noise,
            'strict_mean_field': strict_mean_field,
            'z_min': z_min,
            'z_max': z_max,
            'initial_lengthscale': initial_lengthscale
        }

    def forward(self, branch_inputs, trunk_inputs):
        """
        Probabilistic forward pass with analytic moment matching.
        
        Args:
            branch_inputs: [N, branch_input_dim] - Inputs to branch network
            trunk_inputs: [N, trunk_input_dim] - Inputs to trunk network
            
        Returns:
            pred_mean: [N, 1] - Predicted mean
            pred_var: [N, 1] - Predicted variance (epistemic + aleatoric)
        """
        # Get distributions from both networks
        # alpha: [N, latent_dim], phi: [N, latent_dim]
        alpha_mu, alpha_var = self.branch(branch_inputs)
        phi_mu, phi_var = self.trunk(trunk_inputs)
        
        # Probabilistic product of Gaussians
        # E[XY] = E[X]E[Y] for independent X, Y
        product_mean = alpha_mu * phi_mu
        
        # Var(XY) = Var(X)E[Y]^2 + Var(Y)E[X]^2 + Var(X)Var(Y)
        product_var = (alpha_var * phi_mu.pow(2) +
                      phi_var * alpha_mu.pow(2) +
                      alpha_var * phi_var)
        
        # Sum over latent dimension
        pred_mean = product_mean.sum(dim=-1, keepdim=True)
        pred_var = product_var.sum(dim=-1, keepdim=True)
        
        return pred_mean, pred_var

    def compute_kl(self, jitter=1e-4):
        """Total KL divergence from both networks."""
        return (self.branch.compute_total_kl(jitter) + 
                self.trunk.compute_total_kl(jitter))

    def loss(self, branch_inputs, trunk_inputs, targets, beta=1.0):
        """
        ELBO loss for training.
        
        Args:
            branch_inputs: [N, branch_input_dim]
            trunk_inputs: [N, trunk_input_dim]
            targets: [N, 1] - Target values
            beta: KL weight (for annealing)
            
        Returns:
            total_loss: NLL + beta * KL
            nll: Negative log-likelihood
            kl: KL divergence
        """
        pred_mean, pred_var = self.forward(branch_inputs, trunk_inputs)
        
        # Add observation noise
        noise_var = torch.exp(self.log_noise).clamp(min=1e-6)
        total_var = pred_var + noise_var
        
        # Gaussian NLL
        nll = gaussian_nll_loss(pred_mean, total_var, targets)
        
        # KL divergence
        kl = self.compute_kl()
        
        return nll + beta * kl, nll, kl

    def predict(self, branch_inputs, trunk_inputs, return_std=True):
        """
        Make predictions with uncertainty.
        
        Args:
            branch_inputs: [N, branch_input_dim]
            trunk_inputs: [N, trunk_input_dim]
            return_std: If True, return std instead of var
            
        Returns:
            mean: [N, 1]
            uncertainty: [N, 1] - std or var depending on return_std
        """
        self.eval()
        with torch.no_grad():
            pred_mean, pred_var = self.forward(branch_inputs, trunk_inputs)
            noise_var = torch.exp(self.log_noise).clamp(min=1e-6)
            total_var = pred_var + noise_var
            
        if return_std:
            return pred_mean, torch.sqrt(total_var)
        return pred_mean, total_var

    def predict_with_uncertainty(self, branch_inputs, trunk_inputs):
        """
        Make predictions with decomposed uncertainty.
        
        Returns dict with:
            'mean': Predicted mean [N, 1]
            'epistemic_var': Model uncertainty from SVGP [N, 1]
            'aleatoric_var': Observation noise variance [N, 1]
            'total_var': Sum of epistemic + aleatoric [N, 1]
            'total_std': Square root of total_var [N, 1]
        """
        self.eval()
        with torch.no_grad():
            pred_mean, epistemic_var = self.forward(branch_inputs, trunk_inputs)
            aleatoric_var = torch.exp(self.log_noise).clamp(min=1e-6)
            total_var = epistemic_var + aleatoric_var
            
        return {
            'mean': pred_mean,
            'epistemic_var': epistemic_var,
            'aleatoric_var': aleatoric_var.expand_as(epistemic_var),
            'total_var': total_var,
            'total_std': torch.sqrt(total_var)
        }

    def get_latent_outputs(self, branch_inputs, trunk_inputs):
        """
        Get latent representations for interpretability.
        
        Useful for:
            - Symbolic regression on branch/trunk outputs
            - Visualizing learned factorization
            - Analyzing uncertainty sources
            
        Returns:
            dict with:
                'branch_mean': [N, latent_dim]
                'branch_var': [N, latent_dim]
                'trunk_mean': [N, latent_dim]
                'trunk_var': [N, latent_dim]
        """
        self.eval()
        with torch.no_grad():
            branch_mu, branch_var = self.branch(branch_inputs)
            trunk_mu, trunk_var = self.trunk(trunk_inputs)
            
        return {
            'branch_mean': branch_mu,
            'branch_var': branch_var,
            'trunk_mean': trunk_mu,
            'trunk_var': trunk_var
        }

    def get_branch_outputs(self, branch_inputs):
        """
        Get branch network outputs for symbolic regression.
        
        Args:
            branch_inputs: [N, branch_input_dim]
            
        Returns:
            mean: [N, latent_dim] - Branch output means
            var: [N, latent_dim] - Branch output variances
        """
        self.eval()
        with torch.no_grad():
            return self.branch(branch_inputs)

    def get_trunk_outputs(self, trunk_inputs):
        """
        Get trunk network outputs for symbolic regression.
        
        Args:
            trunk_inputs: [N, trunk_input_dim]
            
        Returns:
            mean: [N, latent_dim] - Trunk output means
            var: [N, latent_dim] - Trunk output variances
        """
        self.eval()
        with torch.no_grad():
            return self.trunk(trunk_inputs)

    def get_noise_std(self):
        """Get learned observation noise standard deviation."""
        return torch.exp(0.5 * self.log_noise).item()

    def get_config(self):
        """Get model configuration for serialization."""
        return self._config.copy()

    @classmethod
    def from_config(cls, config):
        """Create model from configuration dict."""
        return cls(**config)
    
    def check_all_approximations(self, verbose=True):
        """
        Check mean-field approximation assumptions for all layers.
        
        Returns:
            dict with 'branch' and 'trunk' results
        """
        if verbose:
            print("\n=== Branch Network ===")
        branch_results = self.branch.check_all_approximations(verbose=verbose)
        
        if verbose:
            print("\n=== Trunk Network ===")
        trunk_results = self.trunk.check_all_approximations(verbose=verbose)
        
        return {
            'branch': branch_results,
            'trunk': trunk_results
        }


def train_svgpkanpod(model, branch_inputs, trunk_inputs, targets,
                     epochs=3000, lr=0.002, beta_max=0.0001, warmup=1000,
                     patience=500, verbose=True, device=None):
    """
    Train SVGPKanPOD model with best practices.
    
    Features:
        - Cosine annealing learning rate
        - Linear KL annealing
        - Gradient clipping
        - Early stopping with checkpointing
        
    Args:
        model: SVGPKanPOD instance
        branch_inputs: Training branch inputs
        trunk_inputs: Training trunk inputs
        targets: Training targets
        epochs: Maximum training epochs
        lr: Initial learning rate
        beta_max: Maximum KL weight
        warmup: Epochs for KL warmup
        patience: Early stopping patience
        verbose: Print progress
        device: torch device
        
    Returns:
        history: Dict with 'loss', 'rmse' lists
    """
    from copy import deepcopy
    
    if device is not None:
        model = model.to(device)
        branch_inputs = branch_inputs.to(device)
        trunk_inputs = trunk_inputs.to(device)
        targets = targets.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_rmse = float('inf')
    best_state = None
    patience_counter = 0
    
    history = {'loss': [], 'rmse': [], 'nll': [], 'kl': []}
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # KL annealing
        if epoch < warmup:
            beta = beta_max * (epoch / max(1, warmup))
        else:
            beta = beta_max
        
        loss, nll, kl = model.loss(branch_inputs, trunk_inputs, targets, beta=beta)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            pred_mean, _ = model(branch_inputs, trunk_inputs)
            rmse = torch.sqrt(nn.MSELoss()(pred_mean, targets)).item()
        
        history['loss'].append(loss.item())
        history['rmse'].append(rmse)
        history['nll'].append(nll.item())
        history['kl'].append(kl.item())
        
        # Checkpointing
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break
        
        if verbose and epoch % 500 == 0:
            print(f"  Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                  f"NLL: {nll.item():.4f} | KL: {kl.item():.4f} | RMSE: {rmse:.5f}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history
