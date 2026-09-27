# SVGP-KAN: Sparse Variational Gaussian Process Kolmogorov-Arnold Networks

**SVGP-KAN** is a library for building interpretable, probabilistic, and scalable neural networks. It merges the architecture of **Kolmogorov-Arnold Networks (KANs)** with the uncertainty quantification of **Sparse Variational Gaussian Processes**.

Unlike the original GP-KAN (which scales cubically $O(N^3)$), this implementation scales linearly $O(N)$ with data size, making it usable for real-world scientific datasets.

## Key Features

  * **Probabilistic:** Outputs mean predictions and confidence intervals (uncertainty).
  * **Proper Bayesian Inference:** Implements full ELBO optimization with KL divergence for well-calibrated uncertainty.
  * **Scalable:** Uses inducing points and matrix-based batching to train on large datasets (100k+ samples).
  * **Scientific Discovery:** Includes **Automatic Relevance Determination (ARD)** to automatically identify signal vs. noise features.
  * **Computer Vision:** Natively supports 4D inputs (images) for probabilistic U-Nets and CNNs.
  * **Universal:** The RBF kernel naturally adapts to learn periodic, linear, or complex non-linear functions.

## Recent Updates (v0.3 and 0.4)

### ✅ Proper Bayesian Inference
The library implements **full ELBO optimization** with KL divergence regularization:

```python
# Loss = NLL + λ × KL[q(u)||p(u)]
loss = gaussian_nll_loss(pred_mean, pred_var, target) + kl_weight * model.compute_kl()
```

This ensures:
- **Well-calibrated uncertainty**: Predicted intervals match actual error rates
- **Bayesian regularization**: Prevents overfitting through principled priors
- **Theoretically grounded**: Matches standard sparse variational GP formulations

All models (`GPKANLayer`, `GPKAN`, `SVGPUNet`, `GPKANRegressor`) now include `compute_kl()` methods for easy integration.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/yourusername/svgp-kan.git
    cd svgp-kan
    ```

2.  Install the library (automatically installs PyTorch dependencies):

    ```bash
    pip install .
    ```

## Quick Start

### 1. Basic Regression with Proper Bayesian Inference

```python
import torch
import torch.nn as nn
import torch.optim as optim
from svgp_kan import GPKAN, gaussian_nll_loss

# Create model
model = GPKAN(layer_dims=[2, 5, 1], kernel='rbf')
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop with KL divergence
for epoch in range(500):
    # Forward pass
    pred_mean, pred_var = model(X_train)
    
    # Compute ELBO loss
    nll = gaussian_nll_loss(pred_mean, pred_var, y_train)
    kl = model.compute_kl()
    loss = nll + 0.01 * kl  # λ = 0.01 (adjustable)
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: NLL={nll.item():.4f}, KL={kl.item():.2f}")

# Predict with uncertainty
pred_mean, pred_var = model(X_test)
pred_std = torch.sqrt(pred_var)
```

**Key Parameters:**
- `kl_weight` (λ): Typically 0.001-0.1. Higher → stronger regularization, more conservative uncertainty
- Set to 0 to disable Bayesian inference (may not be suitable for uncertainty quantification)

### 2. Scientific Discovery (Feature Selection)

Train a model to identify which inputs matter and which are noise.

```python
import numpy as np
from svgp_kan import GPKANRegressor

# 1. Generate Data (Sine + Linear + Noise)
X = np.random.rand(1000, 3)
# Feature 0: Sine, Feature 1: Linear, Feature 2: Noise
y = np.sin(X[:, 0]*5) + 2*X[:, 1] + 0*X[:, 2] 

# IMPORTANT: Reshape y to [N, 1] for PyTorch training
y = y.reshape(-1, 1)

# 2. Train Scientist Model
# We use RBF kernels as universal approximators
model = GPKANRegressor(hidden_layers=[3, 5, 1], kernel='rbf', use_kl=True, kl_weight=0.01)
model.fit(X, y, epochs=500, sparsity_weight=0.05)

# 3. Interpret Results
model.explain()
# Output Example:
# Feature 0: [ACTIVE] Type: Non-Linear (Sine)
# Feature 1: [ACTIVE] Type: Linear (Large Scale)
# Feature 2: [PRUNED] (Irrelevant)

# 4. Predict with Uncertainty
mu, std = model.predict(X)
```

### 3. Uncertainty Quantification Studies

The repository includes three validation studies demonstrating proper Bayesian inference:

- Study A: Heteroscedastic Noise Calibration: validates that predicted uncertainties correlate with actual errors in spatially-varying noise:

- Study B: Multi-Step Prediction Uncertainty Growth: demonstrates uncertainty propagation in sequential forecasting of fluid dynamics:

- Study C: Out-of-Distribution Detection: shows elevated uncertainty on anomalous data (MNIST)

## Computer Vision Support

SVGP-KAN natively supports U-Net for **2D Computer Vision** tasks (Segmentation, Reconstruction) with **calibrated uncertainty maps**.

**Why use SVGP-KAN for Vision?**

  * **Uncertainty:** Returns a pixel-wise "confusion map" where the model is unsure (e.g., ambiguous tumor boundaries).
  * **Data Efficiency:** GP inductive biases often perform better on small datasets (common in scientific imaging).
  * **Proper Calibration:** KL divergence ensures uncertainties are well-calibrated.

### Usage Example

```python
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from svgp_kan import SVGPUNet_Fluid, gaussian_nll_loss

# 1. Setup Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Initialize the Probabilistic U-Net
model = SVGPUNet_Fluid(in_channels=1, base=32).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training Loop with KL Divergence
for epoch in range(500):
    # Forward pass
    images = torch.randn(8, 1, 64, 64).to(device)
    targets = torch.randn(8, 1, 64, 64).to(device)
    
    pred_mean, pred_var = model(images)
    
    # ELBO loss
    nll = gaussian_nll_loss(pred_mean, pred_var, targets)
    kl = model.compute_kl()
    loss = nll + 0.01 * kl
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. Inference with Uncertainty
model.eval()
with torch.no_grad():
    pred_mean, pred_var = model(test_images)
    uncertainty = torch.sqrt(pred_var)

# 5. Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(pred_mean[0, 0].cpu(), cmap='RdBu_r')
plt.title("Prediction")
plt.subplot(1, 2, 2)
plt.imshow(uncertainty[0, 0].cpu(), cmap='hot')
plt.title("Uncertainty Map")
plt.show()
```

### Custom CNN Integration

You can also drop `GPKANLayer` into *any* existing CNN. It automatically handles 4D tensors `[B, C, H, W]`.

```python
import torch.nn as nn
from svgp_kan import GPKANLayer

class MyCustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        # Replaces a standard Conv1x1 or Linear layer
        self.gp_layer = GPKANLayer(in_features=16, out_features=32)
        
    def forward(self, x):
        x = self.conv(x)
        # No flattening needed!
        # GPKANLayer automatically processes spatial grid
        x_mean, x_var = self.gp_layer(x)
        return x_mean, x_var
```

## API Reference

### Core Components

- **`GPKANLayer`**: Single sparse variational GP layer with `compute_kl()` method
- **`GPKAN`**: Multi-layer KAN architecture with automatic KL aggregation
- **`SVGPUNet`**: U-Net architecture for 2D images with uncertainty
- **`SVGPUNet_Fluid`**: Specialized U-Net for fluid dynamics
- **`GPKANRegressor`**: High-level scikit-learn-style interface
- **`gaussian_nll_loss`**: Negative log-likelihood for mean and variance
- **`compute_kl_divergence`**: KL divergence KL[q(u)||p(u)] for inducing variables

### KL Weight Selection

The KL weight (λ) controls the trade-off between data fitting and regularization:

- **λ = 0**: No explicit Bayesian regularization
- **λ = 0.001-0.01**: Weak regularization, prioritizes data fit
- **λ = 0.01-0.1**: Balanced (recommended for most applications)
- **λ = 0.1-1.0**: Strong regularization, conservative uncertainty

**Rule of thumb:** Start with λ = 0.01 and adjust based on:
- If uncertainties are too small → increase λ
- If model underfits → decrease λ
- Use validation set calibration to tune


## Citation

If you use **SVGP-KAN**, its ideas, and/or examples in your research, please cite the following papers:

```bibtex
@article{ju2025svgpkan,
  title   = {Scalable and Interpretable Scientific Discovery via Sparse Variational Gaussian Process Kolmogorov-Arnold Networks (SVGP-KAN)},
  author  = {Ju, Y. Sungtaek},
  journal = {arXiv preprint arXiv:2512.00260},
  year    = {2025},
  doi     = {10.48550/arXiv.2512.00260},
  url     = {https://arxiv.org/abs/2512.00260}
}

Ju, Y. Sungtaek. “Uncertainty Quantification for Scientific Machine Learning Using Sparse Variational Gaussian Process Kolmogorov–Arnold Networks (SVGP KAN).” Machine Learning: Science and Technology 7, no. 2 (2026): 025042. https://doi.org/10.1088/2632-2153/ae5502.

@article{ju2025svgpkanUQ,
  title   = {Uncertainty Quantification for Scientific Machine Learning using Sparse Variational Gaussian Process Kolmogorov-Arnold Networks (SVGP KAN)},
  author  = {Ju, Y. Sungtaek},
  journal = {Machine Learning: Science and Technology},
  year    = {2026},
  doi     = {10.1088/2632-2153/ae5502},
  url     = {https://doi.org/10.1088/2632-2153/ae5502}
}
```
