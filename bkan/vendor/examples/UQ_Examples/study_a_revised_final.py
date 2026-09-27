"""
Study A: Calibration of Heteroscedastic Measurement Uncertainty in Fluid Fields

REVISED: Adds proper Bayesian inference with KL divergence
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# =============================================================================
# HYPERPARAMETERS - ADJUST HERE
# =============================================================================
USE_KL = True           # Enable KL divergence (Bayesian inference)
KL_WEIGHT = 0.01        # Weight for KL term (0.001, 0.01, 0.1, 1.0)
EPOCHS = 1000
BATCH_SIZE = 16
GRID_SIZE = 64
LEARNING_RATE = 2e-3
NUM_TEST_SAMPLES = 100
OUTPUT_DIR = 'study_results'
# =============================================================================

# Import library components
try:
    from svgp_kan import gaussian_nll_loss, SVGPUNet_Fluid
except ImportError:
    print("Warning: svgp_kan not found. This script requires the library.")
    raise


def generate_calibration_flow(batch_size=16, grid_size=64):
    """Generate flow fields with spatially-varying heteroscedastic noise"""
    x = np.linspace(0, 2*np.pi, grid_size)
    y = np.linspace(0, 2*np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    
    data_t = []
    data_t_plus_1 = []

    for _ in range(batch_size):
        w = np.sin(X) * np.cos(Y) + 0.1 * np.random.randn(grid_size, grid_size)
        shift_w = np.roll(np.roll(w, 1, axis=0), 1, axis=1)
        
        # Noise proportional to field magnitude (Heteroscedastic)
        local_noise_level = 0.01 + 0.15 * np.abs(shift_w) 
        noise = local_noise_level * np.random.randn(grid_size, grid_size)
        
        w_next = 0.95 * shift_w + noise
        data_t.append(w)
        data_t_plus_1.append(w_next)

    inputs = torch.tensor(np.array(data_t), dtype=torch.float32).unsqueeze(1)
    targets = torch.tensor(np.array(data_t_plus_1), dtype=torch.float32).unsqueeze(1)
    return inputs, targets


def train_model(model, device, epochs=500, batch_size=16, grid_size=64, lr=2e-3, 
                use_kl=True, kl_weight=0.01):
    """Train with on-the-fly data generation and optional KL divergence"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\n=== Training Study A: Heteroscedastic Noise Calibration ===")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Bayesian Inference (KL): {'ENABLED' if use_kl else 'DISABLED'}")
    if use_kl:
        print(f"KL Weight: {kl_weight}")
    
    losses = []
    for epoch in range(epochs):
        # Generate fresh batch every epoch
        x, y = generate_calibration_flow(batch_size=batch_size, grid_size=grid_size)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        mu, var = model(x)
        
        # Negative log-likelihood
        nll = gaussian_nll_loss(mu, var, y)
        
        # KL divergence (NEW!)
        if use_kl:
            try:
                kl = model.compute_kl()
            except AttributeError:
                print("Warning: Model missing compute_kl(). Using old version?")
                kl = torch.tensor(0.0)
        else:
            kl = torch.tensor(0.0)
        
        # Total loss
        loss = nll + kl_weight * kl
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            if use_kl and kl.item() > 0:
                print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | NLL: {nll.item():.4f} | KL: {kl.item():.2f} (weighted: {(kl_weight*kl).item():.2f})")
            else:
                print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")
    
    print("✅ Training complete")
    return losses


def evaluate_calibration(model, device, num_test_samples=100, grid_size=64):
    """Comprehensive calibration evaluation"""
    model.eval()
    
    print("\n" + "="*70)
    print("STUDY A: CALIBRATION METRICS")
    print("="*70)
    
    # Generate test data
    print(f"\nGenerating {num_test_samples} test samples...")
    x, y_true = generate_calibration_flow(batch_size=num_test_samples, grid_size=grid_size)
    x, y_true = x.to(device), y_true.to(device)
    
    with torch.no_grad():
        mu, var = model(x)
        std = torch.sqrt(var)
    
    # Flatten for analysis
    y_true_flat = y_true.cpu().numpy().flatten()
    mu_flat = mu.cpu().numpy().flatten()
    std_flat = std.cpu().numpy().flatten()
    err_flat = np.abs(y_true_flat - mu_flat)
    
    # === METRIC 1: Correlation between predicted uncertainty and error ===
    rho, p_value = pearsonr(std_flat, err_flat)
    print(f"\n1. UNCERTAINTY-ERROR CORRELATION")
    print(f"   Pearson correlation (ρ): {rho:.4f}")
    print(f"   p-value: {p_value:.2e}")
    
    # === METRIC 2: Prediction interval coverage ===
    within_1sigma = err_flat <= std_flat
    within_2sigma = err_flat <= 2 * std_flat
    within_3sigma = err_flat <= 3 * std_flat
    
    coverage_1sigma = within_1sigma.mean() * 100
    coverage_2sigma = within_2sigma.mean() * 100
    coverage_3sigma = within_3sigma.mean() * 100
    
    print(f"\n2. PREDICTION INTERVAL COVERAGE")
    print(f"   ±1σ: {coverage_1sigma:.2f}% (expect ~68%)")
    print(f"   ±2σ: {coverage_2sigma:.2f}% (expect ~95%)")
    print(f"   ±3σ: {coverage_3sigma:.2f}% (expect ~99.7%)")
    
    # === METRIC 3: Calibration error ===
    calibration_error_2sigma = np.abs(coverage_2sigma - 95.0)
    print(f"\n3. CALIBRATION ERROR")
    print(f"   |Coverage - Nominal| at 2σ: {calibration_error_2sigma:.2f} pp")
    
    # === METRIC 4: Mean squared error ===
    mse = np.mean((y_true_flat - mu_flat)**2)
    rmse = np.sqrt(mse)
    print(f"\n4. PREDICTION RMSE")
    print(f"   RMSE: {rmse:.4f}")
    
    # === METRIC 5: Normalized residuals (should be ~N(0,1)) ===
    normalized_residuals = (y_true_flat - mu_flat) / (std_flat + 1e-6)
    print(f"\n5. NORMALIZED RESIDUALS")
    print(f"   Mean: {normalized_residuals.mean():.4f} (expect ~0)")
    print(f"   Std:  {normalized_residuals.std():.4f} (expect ~1)")
    
    print("="*70)
    
    return {
        'correlation': rho,
        'p_value': p_value,
        'coverage_1sigma': coverage_1sigma,
        'coverage_2sigma': coverage_2sigma,
        'coverage_3sigma': coverage_3sigma,
        'calibration_error': calibration_error_2sigma,
        'rmse': rmse,
        'normalized_residuals_mean': normalized_residuals.mean(),
        'normalized_residuals_std': normalized_residuals.std(),
        'std_flat': std_flat,
        'err_flat': err_flat,
        'mu': mu.cpu().numpy(),
        'var': var.cpu().numpy(),
        'y_true': y_true.cpu().numpy()
    }


def visualize_results(results, grid_size=64, save_path='examples/study_a_calibration.png'):
    """Create comprehensive visualization with bold Arial fonts and subfigure labels"""
    # Set font to bold Arial
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    
    # Sample visualization (first test case)
    mu = results['mu'][0, 0]
    var = results['var'][0, 0]
    y_true = results['y_true'][0, 0]
    err = np.abs(y_true - mu)
    std = np.sqrt(var)
    
    # Plot 1: Prediction
    im0 = axes[0, 0].imshow(mu, cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    axes[0, 0].text(0.02, 0.98, '(a)', transform=axes[0, 0].transAxes, 
                    fontsize=14, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Plot 2: Predicted Uncertainty
    im1 = axes[0, 1].imshow(std, cmap='hot', vmin=0)
    axes[0, 1].text(0.02, 0.98, '(b)', transform=axes[0, 1].transAxes, 
                    fontsize=14, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Plot 3: Actual Error
    im2 = axes[1, 0].imshow(err, cmap='inferno', vmin=0)
    axes[1, 0].text(0.02, 0.98, '(c)', transform=axes[1, 0].transAxes, 
                    fontsize=14, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Plot 4: Calibration scatter (no fit equation, no grid)
    axes[1, 1].scatter(results['std_flat'], results['err_flat'], 
                       alpha=0.1, s=1, c='purple')
    try:
        m, b = np.polyfit(results['std_flat'], results['err_flat'], 1)
        x_line = np.linspace(results['std_flat'].min(), results['std_flat'].max(), 100)
        axes[1, 1].plot(x_line, m*x_line + b, 'r-', linewidth=2)
    except:
        pass
    axes[1, 1].text(0.02, 0.98, '(d)', transform=axes[1, 1].transAxes, 
                    fontsize=14, fontweight='bold', va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1, 1].set_xlabel("Predicted Uncertainty", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("Actual Error", fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(labelsize=10)
    axes[1, 1].grid(False)  # Remove grid
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {save_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = SVGPUNet_Fluid(base=16).to(device)
    
    # Train
    losses = train_model(
        model, device,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grid_size=GRID_SIZE,
        lr=LEARNING_RATE,
        use_kl=USE_KL,
        kl_weight=KL_WEIGHT
    )
    
    # Evaluate
    results = evaluate_calibration(model, device, num_test_samples=NUM_TEST_SAMPLES, grid_size=GRID_SIZE)
    
    # Visualize
    visualize_results(results, grid_size=GRID_SIZE, save_path=f'{OUTPUT_DIR}/study_a_calibration.png')
    
    # Save results to file
    output_file = f'{OUTPUT_DIR}/study_a_metrics.txt'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("STUDY A: HETEROSCEDASTIC NOISE CALIBRATION - RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"CONFIGURATION:\n")
        f.write(f"  Bayesian Inference: {'ENABLED' if USE_KL else 'DISABLED'}\n")
        if USE_KL:
            f.write(f"  KL Weight: {KL_WEIGHT}\n")
        f.write(f"  Grid size: {GRID_SIZE}×{GRID_SIZE}\n")
        f.write(f"  Test samples: {NUM_TEST_SAMPLES}\n\n")
        f.write(f"Uncertainty-Error Correlation (ρ): {results['correlation']:.4f}\n")
        f.write(f"p-value: {results['p_value']:.2e}\n\n")
        f.write(f"Prediction Interval Coverage:\n")
        f.write(f"  ±1σ: {results['coverage_1sigma']:.2f}% (68% nominal)\n")
        f.write(f"  ±2σ: {results['coverage_2sigma']:.2f}% (95% nominal)\n")
        f.write(f"  ±3σ: {results['coverage_3sigma']:.2f}% (99.7% nominal)\n\n")
        f.write(f"Calibration error (±2σ): {results['calibration_error']:.2f} pp\n")
        f.write(f"Prediction RMSE: {results['rmse']:.4f}\n\n")
        f.write(f"Normalized Residuals:\n")
        f.write(f"  Mean: {results['normalized_residuals_mean']:.4f} (0 expected)\n")
        f.write(f"  Std: {results['normalized_residuals_std']:.4f} (1 expected)\n")
    
    print(f"\n✅ Results saved to: {output_file}")
    print("\nAll metrics ready for manuscript!")


if __name__ == "__main__":
    main()