import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from svgp_kan import GPKANRegressor

# ==========================================
# Helper Functions
# ==========================================

def compute_permutation_importance(model, X_val, y_val):
    """Computes Feature Importance via Permutation."""
    # Fast prediction (no std needed for importance score)
    mu_base = model.predict(X_val, return_std=False)
    mse_base = np.mean((mu_base - y_val)**2)
    
    importances = []
    X_val_np = X_val if isinstance(X_val, np.ndarray) else X_val.cpu().numpy()
    
    for i in range(X_val.shape[1]):
        X_perturbed = X_val_np.copy()
        np.random.shuffle(X_perturbed[:, i])
        
        mu_pert = model.predict(X_perturbed, return_std=False)
        mse_pert = np.mean((mu_pert - y_val)**2)
        
        importances.append(max(0, mse_pert - mse_base))
        
    return np.array(importances)

def plot_diagnostics(model, X_test, y_test, importances, rmse, save_dir="."):
    """Generates diagnostic plots."""
    print(f"Generating plots in {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Feature Importance
    plt.figure(figsize=(10, 6), dpi=300)
    colors = ['#2ca02c' if i < 5 else '#d62728' for i in range(10)]
    plt.bar(range(10), importances, color=colors, alpha=0.8, edgecolor='black')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance (MSE Increase)")
    plt.title(f"Feature Relevance | Test RMSE: {rmse:.3f}")
    plt.xticks(range(10))
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(save_dir, "friedman_feature_importance.png"), bbox_inches='tight')
    plt.close()

    # 2. Regression Fit
    # Use include_likelihood=True to see the prediction interval
    mu, std = model.predict(X_test, include_likelihood=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Random subset for clarity
    subset = np.random.choice(len(y_test), min(100, len(y_test)), replace=False)
    
    # Panel A: Predictions
    ax[0].errorbar(y_test.flatten()[subset], mu.flatten()[subset], 
                   yerr=std.flatten()[subset]*2, fmt='o', alpha=0.5, color='blue', label='Preds (95% CI)')
    ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax[0].set_title("Regression Accuracy")
    ax[0].set_xlabel("True Value")
    ax[0].set_ylabel("Predicted Value")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Panel B: Residuals
    residuals = y_test.flatten() - mu.flatten()
    ax[1].scatter(mu.flatten(), residuals, alpha=0.5, color='purple')
    ax[1].axhline(0, color='black', linestyle='--')
    ax[1].set_title("Residual Analysis")
    ax[1].set_xlabel("Predicted Value")
    ax[1].set_ylabel("Residual (True - Pred)")
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "friedman_regression_fit.png"), bbox_inches='tight')
    plt.close()

    # 3. Discovery Scan (Scanning individual features)
    scan_grid = np.linspace(-2, 2, 100)
    zeros = np.zeros((100, 10)) 
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=300)
    axes = axes.flatten()
    
    for i in range(10):
        X_scan = zeros.copy()
        X_scan[:, i] = scan_grid
        
        # Use include_likelihood=False to see the "clean" learned function
        mu_scan, std_scan = model.predict(X_scan, include_likelihood=False)
        
        ax = axes[i]
        is_signal = i < 5
        color = '#2ca02c' if is_signal else '#d62728'
        
        ax.plot(scan_grid, mu_scan, color=color, linewidth=2)
        ax.fill_between(scan_grid, 
                        (mu_scan - 2*std_scan).flatten(), 
                        (mu_scan + 2*std_scan).flatten(), 
                        color=color, alpha=0.1)
        
        status = "(Signal)" if is_signal else "(Noise)"
        ax.set_title(f"Feature {i} {status}")
        ax.set_ylim(-3, 3) 
        ax.grid(True, alpha=0.3)
        if not is_signal: ax.set_facecolor('#fff5f5') 
            
    plt.suptitle("Learned Functional Shapes (Discovery Scan)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "friedman_learned_functions.png"), bbox_inches='tight')
    plt.close()

def plot_interaction_analysis(model, scaler_x, scaler_y, save_dir="."):
    """Visualizes x0*x1 interaction."""
    print("Generating Interaction Surface...")
    res = 50
    x0_grid = np.linspace(0, 1, res)
    x1_grid = np.linspace(0, 1, res)
    X0, X1 = np.meshgrid(x0_grid, x1_grid)
    
    # Create input matrix (10 features)
    # We fix others to 0.5 (mean)
    X_raw = np.ones((res*res, 10)) * 0.5
    X_raw[:, 0] = X0.ravel()
    X_raw[:, 1] = X1.ravel()
    
    # Scale inputs
    X_input = scaler_x.transform(X_raw)
    
    # Predict (No likelihood noise, just function shape)
    mu_inter, _ = model.predict(X_input, include_likelihood=False)
    
    # Inverse transform output
    Z_pred = scaler_y.inverse_transform(mu_inter).reshape(res, res)
    
    # Ground Truth: 10 * sin(pi * x0 * x1) + ...
    # Note: Friedman #1 is 10 * sin(pi * x1 * x2) + 20 * (x3 - 0.5)^2 ...
    # Wait, indices are 1-based in math, 0-based in code. 
    # Friedman: 10 * sin(pi * X[:,0] * X[:,1])
    Z_true = 10 * np.sin(np.pi * X0 * X1) + 7.5 # +7.5 is approx mean of other terms
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    extent = [0, 1, 0, 1]
    
    im0 = axes[0].imshow(Z_true, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    axes[0].set_title("Ground Truth (x0 * x1)")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(Z_pred, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    axes[1].set_title("SVGP-KAN Prediction")
    plt.colorbar(im1, ax=axes[1])
    
    err = np.abs(Z_true - Z_pred)
    im2 = axes[2].imshow(err, extent=extent, origin='lower', cmap='plasma', aspect='auto')
    axes[2].set_title("Absolute Error")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "friedman_interaction_surface.png"), bbox_inches='tight')
    plt.close()

# ==========================================
# Main Execution
# ==========================================

def run_single_trial(seed=42, save_plots=False):
    """Runs one full training cycle."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Generate Friedman #1 Data
    X, y = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=seed)
    y = y.reshape(-1, 1)
    
    # Standardize
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # 2. Initialize SVGP-KAN
    model = GPKANRegressor(
        hidden_layers=[10, 5, 1], 
        kernel='rbf', 
        num_inducing=30, 
        device='cpu' 
    )
    
    # 3. Train
    print(f"Training Trial {seed}...")
    model.fit(X_train, y_train, epochs=500, lr=0.02, sparsity_weight=0.05, verbose=False)
    
    # 4. Evaluate
    mu, _ = model.predict(X_test)
    mse = np.mean((mu - y_test)**2)
    rmse = np.sqrt(mse)
    
    # 5. Interpretability Check
    imps = compute_permutation_importance(model, X_test, y_test)
    
    # Identify Selected Features
    threshold = 0.05 * np.max(imps)
    selected_features = [i for i, val in enumerate(imps) if val > threshold]
    
    # 6. Plotting (Only for the first trial usually)
    if save_plots:
        plot_diagnostics(model, X_test, y_test, imps, rmse, save_dir="examples")
        plot_interaction_analysis(model, scaler_x, scaler_y, save_dir="examples")
    
    return {
        'rmse': rmse,
        'features': selected_features,
        'importances': imps
    }

def main():
    print("=== SVGP-KAN Benchmark: Friedman #1 (Feature Selection Test) ===")
    
    n_trials = 5
    results = []
    feature_counts = np.zeros(10)
    
    for i in range(n_trials):
        print(f"\n--- Trial {i+1}/{n_trials} ---")
        # Save plots only for the first trial
        res = run_single_trial(seed=42+i, save_plots=(i==0))
        results.append(res)
        
        print(f"RMSE: {res['rmse']:.4f} | Features: {res['features']}")
        for f in res['features']:
            feature_counts[f] += 1

    # --- Final Stats ---
    rmses = [r['rmse'] for r in results]
    print("\n=== Robustness Report ===")
    print(f"Mean RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    
    print("\nFeature Stability:")
    print(f"{'Feat':<5} | {'Truth':<5} | {'Selection Rate':<15} | {'Status'}")
    print("-" * 50)
    ground_truth = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    for i in range(10):
        rate = feature_counts[i] / n_trials
        status = "✅ Robust" if (rate > 0.8 and ground_truth[i]) or (rate < 0.2 and not ground_truth[i]) else "⚠️ Unstable"
        print(f"{i:<5} | {ground_truth[i]:<5} | {rate:<15.2f} | {status}")

if __name__ == "__main__":
    main()