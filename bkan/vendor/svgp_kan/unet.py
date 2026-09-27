import torch
import torch.nn as nn
from .layers import GPKANLayer

class SVGPUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base=32):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck (Uses GPKANLayer which now supports 4D)
        self.bottleneck_gp = GPKANLayer(
            in_features=base*2,
            out_features=base*2,
            num_inducing=32,
            kernel_type='rbf'
        )
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(base*2, base*2, kernel_size=2, stride=2) 
        self.dec2 = self.conv_block(base*2 + base*2, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(base + base, base)
        self.final = nn.Conv2d(base, num_classes, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        mu, var = self.bottleneck_gp(p2)
        
        if self.training:
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
            
        d2 = self.up2(z)
        d2 = torch.cat([d2, e2], dim=1) 
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1), var.sum(dim=1, keepdim=True)
    
    def compute_kl(self, jitter=1e-4):
        """Compute KL divergence for the bottleneck GP layer."""
        return self.bottleneck_gp.compute_kl(jitter=jitter)


class SVGPUNet_Fluid(SVGPUNet):
    """
        
    This version implements  Bayesian inference by:
    1. Sampling z ~ N(mu, var) during training (reparameterization trick)
    2. Using mean z = mu during evaluation
    
    During training, the behavior uses sampled z instead of just mu.
    """
    def __init__(self, in_channels=1, base=16):
        # Initialize parent with dummy classes
        super().__init__(in_channels=in_channels, num_classes=2, base=base)
        
        # Output 2 channels (Mean, Log_Variance)
        self.final = nn.Conv2d(base, 2, kernel_size=1)
        
        # Optional: Initialize log_var to be low (stable start)
        torch.nn.init.constant_(self.final.bias[1], -5.0)

    def forward(self, x, return_gp_var=False):
        """
        Forward pass for fluid dynamics prediction.

        Args:
            x: Input tensor [B, C, H, W]
            return_gp_var: If True, also return the GP bottleneck epistemic
                           variance as a third element. The returned z_var_spatial
                           is the mean GP variance across bottleneck channels,
                           spatially upsampled to match the input resolution
                           [B, 1, H, W]. This is the principled epistemic
                           uncertainty signal from the SVGP-KAN layer, distinct
                           from the learned aleatoric pred_var output.
                           Default False preserves the original return signature
                           so training code is unaffected.
        """
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # GP Bottleneck (Latent Uncertainty)
        # z_var is the GP posterior variance: high near inducing points (data
        # support), reverts to prior sigma_f^2 for out-of-distribution inputs.
        z_mu, z_var = self.bottleneck_gp(p2)

        # Sampling during training (reparameterization trick)
        if self.training:
            std = torch.sqrt(torch.clamp(z_var, min=1e-6))
            eps = torch.randn_like(std)
            z = z_mu + eps * std
        else:
            z = z_mu  # Use mean during evaluation

        # Decoder uses the sampled (or mean) latent representation
        d2 = self.up2(z)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final Output: [Batch, 2, H, W]
        out = self.final(d1)

        pred_mean = out[:, 0:1, :, :]      # Channel 0: Fluid State
        pred_log_var = out[:, 1:2, :, :]   # Channel 1: Aleatoric uncertainty
        pred_var = torch.exp(pred_log_var)

        if return_gp_var:
            # Mean GP variance across bottleneck channels -> [B, 1, H_enc, W_enc]
            z_var_spatial = z_var.mean(dim=1, keepdim=True)
            # Upsample to input resolution for visualization and comparison
            z_var_spatial = torch.nn.functional.interpolate(
                z_var_spatial, size=x.shape[-2:], mode='bilinear', align_corners=False
            )
            return pred_mean, pred_var, z_var_spatial

        return pred_mean, pred_var
