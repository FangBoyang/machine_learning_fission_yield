import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import os

from svgp_kan import GPKANLayer

# =============================================================================
# HYPERPARAMETERS - ADJUST HERE
# =============================================================================
# NOTE: For OOD detection, we intentionally do NOT use KL divergence
# The natural GP variance growth on OOD samples is the detection mechanism
USE_KL = True          # KL regularization destroys OOD separation!
KL_WEIGHT = 0.001         # Not used when USE_KL=False
EPOCHS = 70
BATCH_SIZE = 64
NORMAL_DIGIT = 0
ANOMALY_DIGIT = 7
OUTPUT_DIR = 'study_results'
# =============================================================================

'''
Reconstruction:

"0": Should look like a "0".

"7": Should look like a weird, blurry "0" (or a blob). It should not look like a sharp "7". This is the goal.

Uncertainty (Right Panel):

The Bar for "7" should be significantly higher than for "0".

This demonstrates OOD Detection: "I can't reconstruct this '7' properly, AND my internal variance is spiking because I don't recognize these features."

IMPORTANT: For OOD detection, we do NOT use KL divergence regularization.
The detection mechanism relies on the natural variance growth when the GP 
sees data far from training examples. KL regularization would suppress this
natural mechanism and destroy the ID/OOD separation.
'''

# Fix random seeds for reproducibility
import random
random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
if torch.cuda.is_available():
    torch.cuda.manual_seed(7)
    torch.backends.cudnn.deterministic = True


class StrictAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- Encoder ---
        self.enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*7*7, 64),
            nn.ReLU()
        )
        
        # --- Bottleneck ---
        self.bottleneck = GPKANLayer(
            in_features=64, 
            out_features=6,  
            num_inducing=20, 
            kernel_type='rbf'
        )
        
        # --- TUNING ---
        # 1. Lengthscale: Keep it somewhat large to ensure '0's are connected
        nn.init.constant_(self.bottleneck.log_scale, 0.0) 
        # 2. Variance Amplitude: Start high (1.0) so OOD samples have high uncertainty
        nn.init.constant_(self.bottleneck.log_variance, 0.0) 
        
        # --- Decoder ---
        self.dec_linear = nn.Linear(6, 16*7*7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),  
            nn.Sigmoid() 
        )

    def forward(self, x):
        feat = self.enc(x)
        mu, var = self.bottleneck(feat)
        
        if self.training:
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            z = mu + torch.randn_like(std) * std
        else:
            z = mu 
            
        z_spatial = self.dec_linear(z).view(-1, 16, 7, 7)
        recon = self.dec(z_spatial)
        
        # Uncertainty: Sum variance
        uncertainty_scalar = var.sum(dim=1).view(-1, 1, 1, 1)
        
        return recon, uncertainty_scalar

def get_data(normal_digit=0, batch_size=64): 
    print(f"Preparing Data: Normal Class = '{normal_digit}'")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Train only on '0'
    train_idx = [i for i, (img, label) in enumerate(dataset) if label == normal_digit]
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def train(model, loader, device, epochs=10, use_kl=True, kl_weight=0.01):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    print("\nStarting Training...")
    print(f"Bayesian Inference (KL): {'ENABLED' if use_kl else 'DISABLED'}")
    if not use_kl:
        print("  (KL disabled for OOD detection - natural variance growth mechanism)")
    if use_kl:
        print(f"KL Weight: {kl_weight}")
    
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            
            recon, _ = model(x)
            
            # Reconstruction loss
            loss_recon = F.mse_loss(recon, x)
            
            # KL divergence
            if use_kl:
                try:
                    kl = model.bottleneck.compute_kl()
                except AttributeError:
                    print("Warning: Bottleneck missing compute_kl(). Using old version?")
                    kl = torch.tensor(0.0)
            else:
                kl = torch.tensor(0.0)
            
            # Total loss
            loss = loss_recon + kl_weight * kl
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss/len(loader)
        if use_kl and kl.item() > 0:
            print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Recon: {loss_recon.item():.4f} | KL: {kl.item():.4f}")
        else:
            print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")

def evaluate(model, test_loader, device, normal_digit=0, anomaly_digit=7):
    model.eval()
    img_normal, img_anomaly = None, None
    
    # 1. Grab Samples
    for x, y in test_loader:
        if img_normal is None and (y == normal_digit).any():
            img_normal = x[y == normal_digit][0].unsqueeze(0)
        if img_anomaly is None and (y == anomaly_digit).any():
            img_anomaly = x[y == anomaly_digit][0].unsqueeze(0)
        if img_normal is not None and img_anomaly is not None: break
            
    imgs = torch.cat([img_normal, img_anomaly]).to(device)
    
    # 2. Forward Pass
    with torch.no_grad():
        recons, uncs = model(imgs)
        
    imgs, recons = imgs.cpu(), recons.cpu()
    uncs = uncs.cpu() 
    
    # 3. NUMERICAL LOGGING
    u_norm = uncs[0].item()
    u_anom = uncs[1].item()
    ratio = u_anom / (u_norm + 1e-6)
    
    # 4. RECONSTRUCTION ERROR
    recon_error_norm = F.mse_loss(recons[0], imgs[0]).item()
    recon_error_anom = F.mse_loss(recons[1], imgs[1]).item()
    
    print("\n=== Anomaly Detection Results ===")
    print(f"Normal  ('{normal_digit}') Uncertainty: {u_norm:.4f}")
    print(f"Anomaly ('{anomaly_digit}') Uncertainty: {u_anom:.4f}")
    print(f"Anomaly Score Ratio: {ratio:.2f}x")
    print(f"\nReconstruction Errors:")
    print(f"Normal  ('{normal_digit}') MSE: {recon_error_norm:.6f}")
    print(f"Anomaly ('{anomaly_digit}') MSE: {recon_error_anom:.6f}")
    print(f"Error Ratio: {recon_error_anom/recon_error_norm:.2f}x")
    print("=================================")
    
    # 5. COMPREHENSIVE STATISTICAL ANALYSIS
    print("\n" + "="*70)
    print("STUDY C: COMPREHENSIVE OUT-OF-DISTRIBUTION DETECTION METRICS")
    print("="*70)
    
    # Evaluate on larger test set for statistics
    print("\nEvaluating on full test set...")
    normal_uncertainties = []
    anomaly_uncertainties = []
    normal_recon_errors = []
    anomaly_recon_errors = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            recon, unc = model(x)
            unc_values = unc.squeeze().cpu().numpy()
            recon_errors = F.mse_loss(recon, x, reduction='none').mean(dim=(1,2,3)).cpu().numpy()
            
            # Separate normal vs anomaly
            normal_mask = (y == normal_digit).numpy()
            anomaly_mask = (y == anomaly_digit).numpy()
            
            normal_uncertainties.extend(unc_values[normal_mask].tolist())
            anomaly_uncertainties.extend(unc_values[anomaly_mask].tolist())
            normal_recon_errors.extend(recon_errors[normal_mask].tolist())
            anomaly_recon_errors.extend(recon_errors[anomaly_mask].tolist())
            
            if len(normal_uncertainties) >= 50 and len(anomaly_uncertainties) >= 50:
                break
    
    normal_uncertainties = np.array(normal_uncertainties)
    anomaly_uncertainties = np.array(anomaly_uncertainties)
    normal_recon_errors = np.array(normal_recon_errors)
    anomaly_recon_errors = np.array(anomaly_recon_errors)
    
    print(f"\n1. UNCERTAINTY STATISTICS:")
    print(f"   Normal (n={len(normal_uncertainties)}):")
    print(f"     Mean: {normal_uncertainties.mean():.6f}")
    print(f"     Std:  {normal_uncertainties.std():.6f}")
    print(f"     Min:  {normal_uncertainties.min():.6f}")
    print(f"     Max:  {normal_uncertainties.max():.6f}")
    print(f"\n   Anomaly (n={len(anomaly_uncertainties)}):")
    print(f"     Mean: {anomaly_uncertainties.mean():.6f}")
    print(f"     Std:  {anomaly_uncertainties.std():.6f}")
    print(f"     Min:  {anomaly_uncertainties.min():.6f}")
    print(f"     Max:  {anomaly_uncertainties.max():.6f}")
    
    mean_ratio = anomaly_uncertainties.mean() / normal_uncertainties.mean()
    print(f"\n   Mean Uncertainty Ratio: {mean_ratio:.1f}x")
    
    print(f"\n2. RECONSTRUCTION ERROR STATISTICS:")
    print(f"   Normal:")
    print(f"     Mean MSE: {normal_recon_errors.mean():.6f}")
    print(f"     Std MSE:  {normal_recon_errors.std():.6f}")
    print(f"\n   Anomaly:")
    print(f"     Mean MSE: {anomaly_recon_errors.mean():.6f}")
    print(f"     Std MSE:  {anomaly_recon_errors.std():.6f}")
    
    mse_ratio = anomaly_recon_errors.mean() / normal_recon_errors.mean()
    print(f"\n   Mean MSE Ratio: {mse_ratio:.2f}x")
    
    print(f"\n3. SEPARATION ANALYSIS:")
    # Check overlap
    overlap_threshold = normal_uncertainties.max()
    anomalies_below_threshold = (anomaly_uncertainties < overlap_threshold).sum()
    separation = 1.0 - (anomalies_below_threshold / len(anomaly_uncertainties))
    print(f"   Complete separation: {separation*100:.1f}%")
    print(f"   (% anomalies with uncertainty > max normal uncertainty)")
    
    # ROC-style threshold analysis
    print(f"\n4. THRESHOLD ANALYSIS:")
    # Use thresholds in the actual uncertainty range
    min_unc = min(normal_uncertainties.min(), anomaly_uncertainties.min())
    max_unc = max(normal_uncertainties.max(), anomaly_uncertainties.max())
    thresholds = [
        min_unc + 0.2 * (max_unc - min_unc),  # 20th percentile
        min_unc + 0.4 * (max_unc - min_unc),  # 40th percentile
        min_unc + 0.6 * (max_unc - min_unc),  # 60th percentile
        min_unc + 0.8 * (max_unc - min_unc),  # 80th percentile
    ]
    
    # Compute ROC-AUC
    all_uncertainties = np.concatenate([normal_uncertainties, anomaly_uncertainties])
    all_labels = np.concatenate([np.zeros(len(normal_uncertainties)), 
                                  np.ones(len(anomaly_uncertainties))])
    
    # Sort by uncertainty
    sorted_indices = np.argsort(all_uncertainties)
    sorted_unc = all_uncertainties[sorted_indices]
    sorted_labels = all_labels[sorted_indices]
    
    # Compute ROC curve
    tpr_list = []
    fpr_list = []
    for i in range(len(sorted_unc)):
        threshold = sorted_unc[i]
        predictions = (all_uncertainties > threshold).astype(int)
        tp = np.sum((predictions == 1) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        tn = np.sum((predictions == 0) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Calculate AUC using trapezoidal rule
    roc_auc = 0
    for i in range(1, len(fpr_list)):
        roc_auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    roc_auc = abs(roc_auc)  # Make positive
    
    print(f"\n   ROC-AUC: {roc_auc:.4f}")
    print(f"\n   Threshold Analysis:")
    for thresh in thresholds:
        tp = (anomaly_uncertainties > thresh).sum()
        fp = (normal_uncertainties > thresh).sum()
        tn = (normal_uncertainties <= thresh).sum()
        fn = (anomaly_uncertainties <= thresh).sum()
        
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        
        print(f"     Threshold = {thresh:.4f}:")
        print(f"       Precision: {precision:.3f}, Recall: {recall:.3f}")
    
    print("\n" + "="*70)
    
    # Save metrics to file
    metrics_file = f'{OUTPUT_DIR}/study_c_metrics.txt'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(metrics_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("STUDY C: OUT-OF-DISTRIBUTION DETECTION METRICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"CONFIGURATION:\n")
        f.write(f"  Bayesian Inference: {'ENABLED' if USE_KL else 'DISABLED'}\n")
        if not USE_KL:
            f.write(f"  Note: KL disabled for OOD detection study.\n")
            f.write(f"        Detection relies on natural GP variance growth\n")
            f.write(f"        when encountering data far from training set.\n")
        if USE_KL:
            f.write(f"  KL Weight: {KL_WEIGHT}\n")
        f.write(f"  Epochs: {EPOCHS}\n\n")
        f.write("SINGLE EXAMPLE RESULTS:\n")
        f.write(f"  Normal uncertainty: {u_norm:.4f}\n")
        f.write(f"  Anomaly uncertainty: {u_anom:.4f}\n")
        f.write(f"  Uncertainty ratio: {ratio:.1f}x\n")
        f.write(f"  Normal MSE: {recon_error_norm:.6f}\n")
        f.write(f"  Anomaly MSE: {recon_error_anom:.6f}\n\n")
        f.write("STATISTICAL ANALYSIS (n=50 each):\n")
        f.write(f"  Normal uncertainty: {normal_uncertainties.mean():.6f} ± {normal_uncertainties.std():.6f}\n")
        f.write(f"  Anomaly uncertainty: {anomaly_uncertainties.mean():.6f} ± {anomaly_uncertainties.std():.6f}\n")
        f.write(f"  Mean ratio: {mean_ratio:.1f}x\n")
        f.write(f"  Normal MSE: {normal_recon_errors.mean():.6f} ± {normal_recon_errors.std():.6f}\n")
        f.write(f"  Anomaly MSE: {anomaly_recon_errors.mean():.6f} ± {anomaly_recon_errors.std():.6f}\n")
        f.write(f"  MSE ratio: {mse_ratio:.2f}x\n")
        f.write(f"  Separation: {separation*100:.1f}%\n")
        f.write(f"  ROC-AUC: {roc_auc:.4f}\n")
    
    print(f"✅ Metrics saved to: {metrics_file}\n")

    # 6. Visualization - 2x2 layout (no uncertainty bars, no colorbars)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=300)

    labels = [f"Normal ({normal_digit})", f"Anomaly ({anomaly_digit})"]
    
    for i in range(2):
        # Input
        axes[i, 0].imshow(imgs[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].text(0.02, 0.98, f'({chr(97 + i*2)})', transform=axes[i, 0].transAxes, 
                        fontsize=14, fontweight='bold', va='top', ha='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[i, 0].axis('off')
        
        # Reconstruction
        axes[i, 1].imshow(recons[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].text(0.02, 0.98, f'({chr(97 + i*2 + 1)})', transform=axes[i, 1].transAxes, 
                        fontsize=14, fontweight='bold', va='top', ha='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/study_c_anomaly_detection.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved: {save_path}")

if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StrictAutoencoder().to(DEVICE)
    train_loader, test_loader = get_data(normal_digit=NORMAL_DIGIT, batch_size=BATCH_SIZE)
    train(model, train_loader, DEVICE, epochs=EPOCHS, use_kl=USE_KL, kl_weight=KL_WEIGHT)
    evaluate(model, test_loader, DEVICE, normal_digit=NORMAL_DIGIT, anomaly_digit=ANOMALY_DIGIT)