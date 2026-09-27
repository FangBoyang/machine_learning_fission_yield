import numpy as np
import matplotlib.pyplot as plt
import torch
from svgp_kan import GPKANRegressor

def main():
    print("=== Surface Reconstruction Demo (High Res & Intervals) ===")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Generate 2D Surface Data
    # Pattern: Checkerboard hills (Sine + Cosine)
    N_SAMPLES = 600
    X_train = np.random.rand(N_SAMPLES, 2) * 2 - 1 # [-1, 1]
    y_true = np.sin(2 * np.pi * X_train[:, 0]) + np.cos(2 * np.pi * X_train[:, 1])
    
    # Add substantial noise
    noise_level = 0.2
    y_train = y_true + np.random.randn(N_SAMPLES) * noise_level
    y_train = y_train.reshape(-1, 1)
    
    # Calculate noise bound
    raw_var = np.var(y_train)
    target_noise = (noise_level**2) / raw_var

    # 2. Initialize Model
    # Use 50 inducing points for smooth surface resolution
    model = GPKANRegressor(
        hidden_layers=[2, 5, 1], 
        kernel='rbf', 
        num_inducing=50, 
        device='cpu'
    )

    # 3. Train (Phased)
    
    # Phase 1: Warm-up (Learn the hills)
    print("\n[Phase 1] Warm-up...")
    model.fit(
        X_train, y_train, 
        epochs=500, 
        lr=0.03, 
        sparsity_weight=0.0,
        noise_lower_bound=target_noise * 0.9,
        verbose=True
    )

    # Phase 2: Refine (Tighten the fit)
    # Even though both features are real, a tiny sparsity weight helps 
    # remove any dead neurons in the hidden layer.
    print("\n[Phase 2] Refinement...")
    model.fit(
        X_train, y_train, 
        epochs=1500, 
        lr=0.02, 
        sparsity_weight=0.01,
        noise_lower_bound=target_noise * 0.9,
        verbose=True
    )

    # 4. Visualization
    print("\nVisualizing Surface & Uncertainty...")
    
    # Create grid that extends beyond training data [-1.2, 1.2]
    # This allows us to see "Extrapolation Uncertainty"
    res = 50
    x1_grid = np.linspace(-1.2, 1.2, res)
    x2_grid = np.linspace(-1.2, 1.2, res)
    X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
    X_test_grid = np.column_stack([X1_mesh.ravel(), X2_mesh.ravel()])

    # Predict Latent Function (clean surface)
    # include_likelihood=False shows Epistemic Uncertainty (Model Confidence)
    mu_pred, std_pred = model.predict(X_test_grid, include_likelihood=False)
    
    Mean_mesh = mu_pred.reshape(res, res)
    Std_mesh = std_pred.reshape(res, res)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Panel 1: Noisy Training Data
    # Only plot data within the valid [-1, 1] box
    sc = axes[0].scatter(X_train[:,0], X_train[:,1], c=y_train.ravel(), cmap='viridis', s=20, edgecolors='k', linewidth=0.5)
    axes[0].set_title("1. Noisy Training Data")
    axes[0].set_xlim(-1.2, 1.2)
    axes[0].set_ylim(-1.2, 1.2)
    # Draw box to show training boundary
    rect = plt.Rectangle((-1, -1), 2, 2, fill=False, color='red', linestyle='--', linewidth=2)
    axes[0].add_patch(rect)
    plt.colorbar(sc, ax=axes[0])

    # Panel 2: Predicted Mean (Smooth Manifold)
    cp1 = axes[1].contourf(X1_mesh, X2_mesh, Mean_mesh, levels=20, cmap='viridis')
    axes[1].set_title("2. Predicted Mean Surface")
    # Add boundary box
    rect2 = plt.Rectangle((-1, -1), 2, 2, fill=False, color='white', linestyle='--', linewidth=1.5)
    axes[1].add_patch(rect2)
    plt.colorbar(cp1, ax=axes[1])

    # Panel 3: Uncertainty (Should show extrapolation risk)
    # Notice: Low uncertainty (purple) inside the red box, High (yellow) outside.
    cp2 = axes[2].contourf(X1_mesh, X2_mesh, Std_mesh, levels=20, cmap='plasma')
    axes[2].set_title("3. Epistemic Uncertainty (Confidence)")
    # Add boundary box
    rect3 = plt.Rectangle((-1, -1), 2, 2, fill=False, color='white', linestyle='--', linewidth=1.5)
    axes[2].add_patch(rect3)
    plt.colorbar(cp2, ax=axes[2])

    plt.tight_layout()
    plt.savefig("examples/surface_reconstruction.png")
    #plt.show()

if __name__ == "__main__":
    main()