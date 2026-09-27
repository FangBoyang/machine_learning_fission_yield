import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from .model import GPKAN, gaussian_nll_loss

class GPKANRegressor:
    """
    High-level API for GP-KAN.
    
    Set use_kl=False for no KL term.
    Set use_kl=True for proper Bayesian variational inference.
    """
    def __init__(self, hidden_layers=[1, 5, 1], kernel='rbf', num_inducing=20, device='cpu'):
        self.hidden_layers = hidden_layers
        self.kernel = kernel
        self.num_inducing = num_inducing
        self.device = device
        
        self.model = GPKAN(
            layers_hidden=hidden_layers, 
            num_inducing=num_inducing, 
            kernel_type=kernel
        ).to(device)
        
        self.likelihood_log_var = torch.nn.Parameter(
            torch.tensor(-2.0, device=device) 
        )
        
        self.history = {'loss': [], 'nll': [], 'kl': [], 'sparsity': []}
        self.trained = False
        self.noise_lower_bound = 1e-4 

    def fit(self, X, y, epochs=1000, lr=0.02, sparsity_weight=0.05, 
            kl_weight=0.01, use_kl=True, noise_lower_bound=None, verbose=True):
        """
        Train the GP-KAN model.
        
        Args:
            X: Input data [N, in_features]
            y: Target data [N, 1] or [N,]
            epochs: Number of training epochs
            lr: Learning rate
            sparsity_weight: Weight for L1 sparsity penalty
            kl_weight: Weight for KL divergence term (Bayesian regularization)
            use_kl: If True, use KL divergence (proper VI). If False, ignore KL (old behavior).
            noise_lower_bound: Minimum observation noise
            verbose: Print training progress
        
        Returns:
            self
        """
        X_ten = self._to_tensor(X)
        y_ten = self._to_tensor(y)
        
        data_var = y_ten.var().item()
        if noise_lower_bound is None:
            self.noise_lower_bound = 1e-4
        else:
            if noise_lower_bound > 0.5 * data_var:
                safe_bound = 0.5 * data_var
                if verbose:
                    print(f"WARNING: noise_lower_bound ({noise_lower_bound:.4f}) is too high.")
                    print(f"Auto-adjusting bound to {safe_bound:.4f}.")
                self.noise_lower_bound = safe_bound
            else:
                self.noise_lower_bound = noise_lower_bound

        optimizer = optim.Adam([
            {'params': self.model.parameters()},
            {'params': [self.likelihood_log_var], 'lr': 0.05} 
        ], lr=lr)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//4, gamma=0.8)
        
        self.model.train()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Training GP-KAN Regressor")
            print(f"{'='*70}")
            print(f"Epochs: {epochs}")
            print(f"KL Divergence: {'ENABLED (Proper VI)' if use_kl else 'DISABLED (Old behavior)'}")
            print(f"KL Weight: {kl_weight if use_kl else 'N/A'}")
            print(f"Sparsity Weight: {sparsity_weight}")
            print(f"{'='*70}\n")
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            f_mu, f_var = self.model(X_ten)
            
            obs_noise = torch.exp(self.likelihood_log_var).clamp(min=self.noise_lower_bound)
            y_var = f_var + obs_noise
            
            # NLL term (data fidelity)
            nll = gaussian_nll_loss(f_mu, y_var, y_ten)
            
            # KL term (Bayesian regularization)
            if use_kl:
                kl = self.model.compute_total_kl()
            else:
                kl = torch.tensor(0.0, device=self.device)
            
            # Sparsity term (ARD via L1 on kernel variance)
            l1_loss = 0
            first_layer = self.model.layers[0]
            l1_loss += torch.exp(first_layer.log_variance).sum() * sparsity_weight
            
            # Total loss: -ELBO = NLL + KL + Sparsity
            # (Minimizing -ELBO is equivalent to maximizing ELBO)
            total_loss = nll + kl_weight * kl + l1_loss
            
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Record history
            self.history['loss'].append(total_loss.item())
            self.history['nll'].append(nll.item())
            self.history['kl'].append(kl.item() if use_kl else 0.0)
            self.history['sparsity'].append(l1_loss.item())
            
            if verbose and (epoch + 1) % 100 == 0:
                if use_kl:
                    print(f"Epoch {epoch+1:4d}/{epochs} | "
                          f"NLL: {nll.item():.4f} | "
                          f"KL: {kl.item():.4f} | "
                          f"Sparse: {l1_loss.item():.4f} | "
                          f"Total: {total_loss.item():.4f} | "
                          f"Noise: {obs_noise.item():.4f}")
                else:
                    print(f"Epoch {epoch+1:4d}/{epochs} | "
                          f"NLL: {nll.item():.4f} | "
                          f"Sparse: {l1_loss.item():.4f} | "
                          f"Noise: {obs_noise.item():.4f}")

        self.trained = True
        if verbose:
            print(f"\n{'='*70}")
            print("✓ Training Complete")
            print(f"{'='*70}\n")
        return self

    def predict(self, X, return_std=True, include_likelihood=True):
        if not self.trained:
            raise RuntimeError("Model is not trained yet. Call .fit() first.")
            
        self.model.eval()
        X_ten = self._to_tensor(X)
        
        with torch.no_grad():
            f_mu, f_var = self.model(X_ten)
            
            if not return_std:
                return f_mu.cpu().numpy()

            if include_likelihood:
                obs_noise = torch.exp(self.likelihood_log_var).clamp(min=self.noise_lower_bound)
                y_var = f_var + obs_noise
                std = torch.sqrt(y_var)
            else:
                std = torch.sqrt(f_var)
            
        return f_mu.cpu().numpy(), std.cpu().numpy()

    def explain(self, threshold=0.01):
        print("\n=== GP-KAN Model Explanation ===")
        layer0 = self.model.layers[0]
        # Explicitly detach/cpu for printing
        relevance = layer0.get_relevance().detach().cpu().mean(dim=0)
        scales = torch.exp(layer0.log_scale).detach().cpu().mean(dim=0)
        
        for i in range(self.hidden_layers[0]):
            score, s = relevance[i].item(), scales[i].item()
            status = "[PRUNED]" if score < threshold else "[ACTIVE]"
            shape = "Linear" if s > 3.0 else ("High Freq" if s < 0.5 else "Smooth/Non-Linear")
            if status == "[ACTIVE]":
                print(f"Feature {i}: {status} importance={score:.3f} | Type: {shape} (scale={s:.2f})")
            else:
                print(f"Feature {i}: {status} (Irrelevant)")

    def plot_diagnosis(self, figsize=(14, 4)):
        """Plot training diagnostics including KL divergence if used."""
        n_plots = 4 if max(self.history['kl']) > 0 else 3
        
        plt.figure(figsize=figsize)
        
        plt.subplot(1, n_plots, 1)
        plt.plot(self.history['loss'])
        plt.title("Total Loss")
        plt.xlabel("Epoch")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, n_plots, 2)
        plt.plot(self.history['nll'], label='NLL')
        plt.title("Negative Log-Likelihood")
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, n_plots, 3)
        plt.plot(self.history['sparsity'])
        plt.title("Sparsity Loss")
        plt.xlabel("Epoch")
        plt.grid(True, alpha=0.3)
        
        if n_plots == 4:
            plt.subplot(1, n_plots, 4)
            plt.plot(self.history['kl'])
            plt.title("KL Divergence")
            plt.xlabel("Epoch")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(self.device)
        return x.float().to(self.device)
