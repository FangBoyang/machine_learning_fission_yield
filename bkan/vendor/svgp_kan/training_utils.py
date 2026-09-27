"""
training_utils.py - Principled approaches to SVGP-KAN training stability

This module provides several techniques to prevent training collapse:
1. Early stopping with validation monitoring
2. Learning rate scheduling
3. Gradient clipping
4. KL annealing strategies
5. Model checkpointing
"""

import torch
import torch.optim as optim
import numpy as np
from copy import deepcopy


class EarlyStopping:
    """
    Stop training when validation metric stops improving.
    
    This is principled because it directly optimizes generalization,
    not just training loss.
    """
    def __init__(self, patience=500, min_delta=1e-4, mode='min'):
        """
        Args:
            patience: Epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy-like metrics
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
            
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
        return self.should_stop


class ModelCheckpoint:
    """
    Save model state when validation metric improves.
    
    This ensures we keep the best model, not the last model.
    """
    def __init__(self, mode='min'):
        self.mode = mode
        self.best_score = None
        self.best_state = None
        self.best_epoch = 0
        
    def __call__(self, model, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_state = deepcopy(model.state_dict())
            self.best_epoch = epoch
            return True
            
        if self.mode == 'min':
            improved = score < self.best_score
        else:
            improved = score > self.best_score
            
        if improved:
            self.best_score = score
            self.best_state = deepcopy(model.state_dict())
            self.best_epoch = epoch
            return True
        return False
    
    def load_best(self, model):
        """Restore best model weights."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            return self.best_epoch
        return None


class KLAnnealingSchedule:
    """
    Various KL annealing strategies for variational inference.
    
    The KL term in ELBO can dominate early training, causing the model
    to ignore the data. Annealing helps by starting with low KL weight.
    """
    def __init__(self, strategy='linear', warmup_epochs=500, max_weight=0.001, 
                 cyclical_period=1000):
        """
        Args:
            strategy: 'linear', 'sigmoid', 'cyclical', 'constant'
            warmup_epochs: Epochs before reaching max_weight (for linear/sigmoid)
            max_weight: Maximum KL weight (β in β-VAE)
            cyclical_period: Period for cyclical annealing
        """
        self.strategy = strategy
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight
        self.cyclical_period = cyclical_period
        
    def __call__(self, epoch):
        if self.strategy == 'constant':
            return self.max_weight
            
        elif self.strategy == 'linear':
            # Linear warmup
            return min(self.max_weight, (epoch / self.warmup_epochs) * self.max_weight)
            
        elif self.strategy == 'sigmoid':
            # Smooth S-curve warmup
            x = (epoch - self.warmup_epochs / 2) / (self.warmup_epochs / 10)
            sigmoid = 1 / (1 + np.exp(-x))
            return self.max_weight * sigmoid
            
        elif self.strategy == 'cyclical':
            # Cyclical annealing (Fu et al., 2019)
            # Helps escape local optima
            cycle_pos = (epoch % self.cyclical_period) / self.cyclical_period
            return self.max_weight * min(1.0, cycle_pos * 2)
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class GradientClipper:
    """
    Clip gradients to prevent exploding gradients.
    
    Two modes:
    - 'norm': Clip by global norm (preserves direction)
    - 'value': Clip each gradient independently
    """
    def __init__(self, max_norm=1.0, mode='norm'):
        self.max_norm = max_norm
        self.mode = mode
        
    def __call__(self, model):
        if self.mode == 'norm':
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        elif self.mode == 'value':
            torch.nn.utils.clip_grad_value_(model.parameters(), self.max_norm)


def create_lr_scheduler(optimizer, strategy='cosine', total_epochs=2500, 
                        warmup_epochs=100, min_lr=1e-5):
    """
    Create learning rate scheduler.
    
    Args:
        strategy: 'cosine', 'step', 'plateau', 'warmup_cosine'
        
    Returns:
        Scheduler object
    """
    if strategy == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=min_lr
        )
        
    elif strategy == 'step':
        # Decay by 0.5 every 500 epochs
        return optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
        
    elif strategy == 'plateau':
        # Reduce LR when validation loss plateaus
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=200, min_lr=min_lr
        )
        
    elif strategy == 'warmup_cosine':
        # Linear warmup + cosine decay
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return max(min_lr / optimizer.defaults['lr'], 
                          0.5 * (1 + np.cos(np.pi * progress)))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def compute_validation_nmse(model, val_params, val_coords, val_y):
    """Compute NMSE on validation set."""
    with torch.no_grad():
        mu, _ = model(val_params, val_coords)
        nmse = torch.norm(mu - val_y)**2 / torch.norm(val_y)**2
    return nmse.item()


class StableTrainer:
    """
    Complete training loop with all stability features.
    
    Usage:
        trainer = StableTrainer(model, train_data, val_data)
        history = trainer.train(epochs=2500)
        trainer.restore_best()  # Load best model
    """
    def __init__(self, model, train_data, val_data, 
                 lr=0.005, kl_strategy='linear', kl_max=0.001,
                 early_stopping_patience=500, grad_clip=1.0,
                 lr_scheduler='cosine'):
        """
        Args:
            model: SVGPKanPOD model
            train_data: (params, coords, y) tensors
            val_data: (params, coords, y) tensors
            lr: Initial learning rate
            kl_strategy: KL annealing strategy
            kl_max: Maximum KL weight
            early_stopping_patience: Patience for early stopping
            grad_clip: Gradient clipping threshold
            lr_scheduler: LR scheduler strategy
        """
        self.model = model
        self.train_params, self.train_coords, self.train_y = train_data
        self.val_params, self.val_coords, self.val_y = val_data
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.kl_schedule = KLAnnealingSchedule(strategy=kl_strategy, max_weight=kl_max)
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.checkpoint = ModelCheckpoint(mode='min')
        self.grad_clipper = GradientClipper(max_norm=grad_clip)
        self.lr_scheduler = None  # Set in train()
        self.lr_strategy = lr_scheduler
        
    def train(self, epochs=2500, log_interval=100):
        """
        Train with all stability features.
        
        Returns:
            history: Dict with training metrics
        """
        from svgp_kan import gaussian_nll_loss
        
        # Initialize LR scheduler
        self.lr_scheduler = create_lr_scheduler(
            self.optimizer, strategy=self.lr_strategy, total_epochs=epochs
        )
        
        history = {
            'train_loss': [], 'val_nmse': [], 'kl_weight': [], 'lr': []
        }
        
        for epoch in range(epochs + 1):
            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            
            mu, var = self.model(self.train_params, self.train_coords)
            nll = gaussian_nll_loss(mu, var, self.train_y)
            kl = self.model.compute_total_kl()
            
            kl_weight = self.kl_schedule(epoch)
            loss = nll + kl_weight * kl
            
            loss.backward()
            self.grad_clipper(self.model)
            self.optimizer.step()
            
            # Update LR
            if self.lr_strategy != 'plateau':
                self.lr_scheduler.step()
            
            # Validation
            self.model.eval()
            val_nmse = compute_validation_nmse(
                self.model, self.val_params, self.val_coords, self.val_y
            )
            
            if self.lr_strategy == 'plateau':
                self.lr_scheduler.step(val_nmse)
            
            # Checkpointing
            self.checkpoint(self.model, val_nmse, epoch)
            
            # Early stopping check
            if self.early_stopping(val_nmse, epoch):
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation NMSE: {self.early_stopping.best_score:.4e} "
                      f"at epoch {self.early_stopping.best_epoch}")
                break
            
            # Logging
            if epoch % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d} | Loss: {loss.item():+.3f} | "
                      f"Val NMSE: {val_nmse:.4e} | KL_w: {kl_weight:.4f} | "
                      f"LR: {current_lr:.2e}")
                
                history['train_loss'].append(loss.item())
                history['val_nmse'].append(val_nmse)
                history['kl_weight'].append(kl_weight)
                history['lr'].append(current_lr)
        
        return history
    
    def restore_best(self):
        """Load the best model from training."""
        best_epoch = self.checkpoint.load_best(self.model)
        print(f"Restored best model from epoch {best_epoch} "
              f"(val NMSE: {self.checkpoint.best_score:.4e})")
        return best_epoch
