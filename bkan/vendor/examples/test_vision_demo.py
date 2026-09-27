import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time

# Import the new U-Net from your library
from svgp_kan import SVGPUNet

def test_vision_pipeline():
    print("=== SVGP-KAN Vision Test (U-Net) ===")
    
    # 1. Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize U-Net
    # 1 Input Channel (Grayscale) -> 2 Output Classes (Background, Object)
    model = SVGPUNet(in_channels=1, num_classes=2, base=16).to(device)
    print("Model initialized successfully.")
    
    # 2. Generate Synthetic Data
    # Batch of 4 images, 1 Channel, 64x64
    B, C, H, W = 4, 1, 64, 64
    x = torch.randn(B, C, H, W).to(device)
    # Dummy masks (Integer class labels 0 or 1)
    y = torch.randint(0, 2, (B, H, W)).to(device)
    
    print(f"Input Shape: {x.shape}")
    
    # 3. Forward Pass Check
    start = time.time()
    logits, unc_map = model(x)
    dt = time.time() - start
    
    print(f"Forward Pass Time: {dt*1000:.2f} ms")
    print(f"Logits Shape: {logits.shape} (Expected: [{B}, 2, {H}, {W}])")
    print(f"Uncertainty Shape: {unc_map.shape} (Expected: [{B}, 1, {H}, {W}])")
    
    # 4. Backward Pass Check (Training Logic)
    # This proves the gradients flow through the GP bottleneck correctly
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Task Loss (Segmentation)
    loss_seg = F.cross_entropy(logits, y)
    
    # Sparsity Loss (From the Bottleneck GP)
    # We access the bottleneck layer to get ARD terms
    gp_layer = model.bottleneck_gp
    # We use .sum() here because we want to prune entire feature channels in the bottleneck
    loss_sparsity = torch.exp(gp_layer.log_variance).sum() * 0.01
    
    total_loss = loss_seg + loss_sparsity
    
    print(f"Loss: {total_loss.item():.4f}")
    
    try:
        total_loss.backward()
        optimizer.step()
        print("✅ Backward Pass Successful (Gradients Flowing)")
    except Exception as e:
        print(f"❌ Backward Pass Failed: {e}")
        return

    # 5. Visual Check (Uncertainty)
    print("\nVisualizing Output...")
    
    # Detach for plotting
    img = x[0, 0].cpu().detach().numpy()
    pred = torch.argmax(logits, dim=1)[0].cpu().detach().numpy()
    unc = unc_map[0, 0].cpu().detach().numpy()
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Input Image (Noise)")
    
    ax[1].imshow(pred, cmap='tab10')
    ax[1].set_title("Prediction (Random Init)")
    
    # Uncertainty map should show variation (not all zeros)
    im = ax[2].imshow(unc, cmap='plasma')
    ax[2].set_title("Bottleneck Uncertainty")
    plt.colorbar(im, ax=ax[2])
    
    plt.tight_layout()
    plt.show()
    print("Test Complete.")

if __name__ == "__main__":
    test_vision_pipeline()

# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np
# import time

# # Import your classes
# from unet import SVGPUNet

# def test_vision_pipeline():
#     print("=== SVGP-KAN Vision Test ===")
    
#     # 1. Setup
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Device: {device}")
    
#     # Create Model (1 Input Channel -> 2 Classes)
#     model = SVGPUNet(in_channels=1, num_classes=2, base=16).to(device)
#     print("Model initialized successfully.")
    
#     # 2. Generate Synthetic Data
#     # Batch of 4 images, 1 Channel, 64x64
#     B, C, H, W = 4, 1, 64, 64
#     x = torch.randn(B, C, H, W).to(device)
#     # Dummy masks (0 or 1)
#     y = torch.randint(0, 2, (B, H, W)).to(device)
    
#     print(f"Input Shape: {x.shape}")
    
#     # 3. Forward Pass Check
#     start = time.time()
#     logits, unc_map = model(x)
#     dt = time.time() - start
    
#     print(f"Forward Pass Time: {dt*1000:.2f} ms")
#     print(f"Logits Shape: {logits.shape} (Expected: [{B}, 2, {H}, {W}])")
#     print(f"Uncertainty Shape: {unc_map.shape} (Expected: [{B}, 1, {H}, {W}])")
    
#     # 4. Backward Pass Check (Training Logic)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
#     # Loss 1: Segmentation (Cross Entropy)
#     loss_seg = F.cross_entropy(logits, y)
    
#     # Loss 2: Sparsity / KL (From the GP Layer)
#     # We access the bottleneck layer to get ARD (Automatic Relevance Determination)
#     gp_layer = model.bottleneck_gp
#     # Simple L1 on the log_variance (pushes irrelevant features to zero)
#     loss_sparsity = torch.exp(gp_layer.log_variance).mean() * 0.1
    
#     total_loss = loss_seg + loss_sparsity
    
#     print(f"Loss: {total_loss.item():.4f}")
    
#     try:
#         total_loss.backward()
#         optimizer.step()
#         print("✅ Backward Pass Successful (Gradients Flowing)")
#     except Exception as e:
#         print(f"❌ Backward Pass Failed: {e}")
#         return

#     # 5. Visual Check (Uncertainty)
#     print("\nVisualizing Output...")
    
#     # Detach for plotting
#     img = x[0, 0].cpu().detach().numpy()
#     pred = torch.argmax(logits, dim=1)[0].cpu().detach().numpy()
#     unc = unc_map[0, 0].cpu().detach().numpy()
    
#     fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
#     ax[0].imshow(img, cmap='gray')
#     ax[0].set_title("Input Image (Noise)")
    
#     ax[1].imshow(pred, cmap='tab10')
#     ax[1].set_title("Prediction (Random Init)")
    
#     # Uncertainty map should show variation (not all zeros)
#     im = ax[2].imshow(unc, cmap='plasma')
#     ax[2].set_title("Bottleneck Uncertainty")
#     plt.colorbar(im, ax=ax[2])
    
#     plt.tight_layout()
#     plt.show()
#     print("Test Complete.")

# if __name__ == "__main__":
#     test_vision_pipeline()