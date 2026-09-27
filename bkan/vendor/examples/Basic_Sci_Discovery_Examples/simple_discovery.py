import numpy as np
import matplotlib.pyplot as plt
import torch
from svgp_kan import GPKANRegressor

def main():
    print("=== Scientific Discovery Demo (High Res & Intervals) ===")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Generate Data
    N = 600
    X = np.random.rand(N, 3) * 2 - 1 # Range [-1, 1]
    
    # Feature 0: High Frequency Sine 
    f1 = np.sin(3 * np.pi * X[:, 0]) 
    # Feature 1: Linear
    f2 = 1.5 * X[:, 1]               
    # Feature 2: Noise
    f3 = 0.0 * X[:, 2]               
    
    y = f1 + f2 + f3 + np.random.randn(N) * 0.1
    y = y.reshape(-1, 1)
    
    # Noise Floor Calculation
    raw_var = np.var(y)
    target_noise = (0.1**2) / raw_var
    
    # 2. Initialize Model
    # INCREASED CAPACITY: num_inducing=50
    model = GPKANRegressor(
        hidden_layers=[3, 5, 1], 
        kernel='rbf', 
        num_inducing=50, 
        device='cpu'
    )

    # --- PHASE 1: WARM-UP ---
    print("\n[Phase 1] Warm-up (Learning Signal)...")
    model.fit(
        X, y, 
        epochs=500, 
        lr=0.03, 
        sparsity_weight=0.0,  
        noise_lower_bound=target_noise * 0.9,
        verbose=True
    )

    # --- PHASE 2: PRUNING ---
    print("\n[Phase 2] Pruning (Removing Noise)...")
    model.fit(
        X, y, 
        epochs=1500, 
        lr=0.02, 
        sparsity_weight=0.05, 
        noise_lower_bound=target_noise * 0.9,
        verbose=True
    )

    # 3. Explain
    model.explain(threshold=0.01)

    # 4. Visualization
    print("\nVisualizing...")
    
    # EXTENDED GRID: [-1.2, 1.2] to show extrapolation uncertainty
    N_TEST = 300
    grid = np.linspace(-1.2, 1.2, N_TEST)
    zeros = np.zeros(N_TEST)

    X_test_1 = np.column_stack([grid, zeros, zeros])
    X_test_2 = np.column_stack([zeros, grid, zeros])
    X_test_3 = np.column_stack([zeros, zeros, grid])

    # Get pure function predictions (Epistemic Uncertainty)
    # The bands will be tight in [-1,1] and wide outside.
    mu1, std1 = model.predict(X_test_1, include_likelihood=True)
    mu2, std2 = model.predict(X_test_2, include_likelihood=True)
    mu3, std3 = model.predict(X_test_3, include_likelihood=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # Panel 1: Sine
    # Show real data scatter (slice)
    mask1 = (np.abs(X[:,1]) < 0.2) & (np.abs(X[:,2]) < 0.2)
    # Plot data only in valid range
    axes[0].scatter(X[mask1, 0], y[mask1], c='k', s=10, alpha=0.3, label='Data Slice')
    
    axes[0].plot(grid, mu1, 'b-', lw=2, label='Learned')
    
    # Plot Truth only in valid range [-1, 1] to highlight extrapolation
    valid_idx = (grid >= -1) & (grid <= 1)
    axes[0].plot(grid[valid_idx], np.sin(3 * np.pi * grid[valid_idx]), 'r--', label='Truth')
    
    # The CONFIDENCE BAND (This is what you want to see)
    axes[0].fill_between(grid, 
                         mu1.flatten()-2*std1.flatten(), 
                         mu1.flatten()+2*std1.flatten(), 
                         color='blue', alpha=0.3, label='95% Conf') # Increased alpha
    
    axes[0].set_title("Feature 0: Periodic Law")
    axes[0].legend(fontsize='small')
    axes[0].set_xlim(-1.5, 1.5)

    # Panel 2: Linear
    mask2 = (np.abs(X[:,0]) < 0.2) & (np.abs(X[:,2]) < 0.2)
    axes[1].scatter(X[mask2, 1], y[mask2], c='k', s=10, alpha=0.3, label='Data Slice')
    
    axes[1].plot(grid, mu2, 'b-', lw=2, label='Learned')
    axes[1].plot(grid[valid_idx], 1.5 * grid[valid_idx], 'r--', label='Truth')
    
    axes[1].fill_between(grid, 
                         mu2.flatten()-2*std2.flatten(), 
                         mu2.flatten()+2*std2.flatten(), 
                         color='blue', alpha=0.3)
    
    axes[1].set_title("Feature 1: Linear Law")
    axes[1].set_xlim(-1.5, 1.5)

    # Panel 3: Noise
    axes[2].plot(grid, mu3, 'b-', lw=2, label='Learned')
    axes[2].fill_between(grid, 
                         mu3.flatten()-2*std3.flatten(), 
                         mu3.flatten()+2*std3.flatten(), 
                         color='blue', alpha=0.3)
    
    axes[2].set_ylim(-2, 2)
    axes[2].set_xlim(-1.5, 1.5)
    axes[2].set_title("Feature 2: Pruned (Noise)")

    plt.tight_layout()
    plt.savefig("examples/simple_discovery.png") 
    #plt.show()

if __name__ == "__main__":
    main()
