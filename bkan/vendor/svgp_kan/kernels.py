"""
kernels.py - Moment matching for SVGP-KAN with orthogonal variance

This version includes:
- Original mean computation (unchanged)
- Projected variance (mean-field approximation)
- Orthogonal variance (Nyström approximation error) with CORRECTED normalization
- Optional strict_mean_field mode for enforcing approximation assumptions

The orthogonal variance formula is derived from:
    σ²_⊥ = ψ₀ - tr(K_ZZ⁻¹ Ψ₂)

Under mean-field approximation (K_ZZ ≈ σ_f² I, Ψ₂ ≈ ψ₁ψ₁ᵀ):
    σ²_⊥ ≈ σ_f² - ||ψ₁||² / σ_f² = σ_f² (1 - ||ψ₁||² / σ_f⁴)

MODES:
- strict_mean_field=False (DEFAULT): Uses clamping as heuristic. Backward compatible.
  The normalized trace is clamped to [0,1], which works well in practice even when
  the mean-field assumptions are not strictly satisfied.
  
- strict_mean_field=True: Enforces that inducing points are well-separated (>2ℓ apart)
  at initialization. This ensures the mean-field approximation starts valid, though
  lengthscales may adapt during training. Clamping is still used as a safety net.

BUGFIX (v0.4.2): Corrected normalization from σ_f² to σ_f⁴.
UPDATE (v0.4.4): Toned down warning messages - clamping works well in practice.
"""

import torch
import math
import warnings


def rbf_moment_matching(x_mean, x_var, z, q_mu, q_var, lengthscale, variance,
                        return_diagnostics=False):
    """
    Analytic moment matching for RBF Kernel with orthogonal variance correction.
    
    Propagates Gaussian distributions N(x_mean, x_var) through univariate GPs
    defined by inducing points z and variational parameters q_mu, q_var.
    
    Args:
        x_mean: [Batch, In] input means
        x_var: [Batch, In] input variances
        z: [Out, In, M] inducing point locations
        q_mu: [Out, In, M] variational means
        q_var: [Out, In, M] variational variances
        lengthscale: [Out, In] kernel lengthscales
        variance: [Out, In] kernel signal variances (σ_f²)
        return_diagnostics: if True, also return diagnostic dict
        
    Returns:
        edge_means: [Batch, Out, In] output means
        edge_vars: [Batch, Out, In] output variances
        diagnostics: (optional) dict with approximation quality metrics
    """
    # Broadcast inputs to match edge dimensions [Batch, Out, In, Inducing]
    x_mu_exp = x_mean.unsqueeze(1).unsqueeze(-1)
    x_var_exp = x_var.unsqueeze(1).unsqueeze(-1)
    
    # Kernel parameters broadcast
    ell_sq = lengthscale.unsqueeze(0).unsqueeze(-1).pow(2)
    var_kern = variance.unsqueeze(0).unsqueeze(-1)  # This is σ_f²

    # --- 1. Compute Psi_1 (Expected Kernel Matrix) ---
    # ψ₁,j = σ_f² √(ℓ²/(ℓ²+σ_x²)) exp(-(Z_j - μ_x)² / (2(ℓ²+σ_x²)))
    denom = ell_sq + x_var_exp
    diff = x_mu_exp - z.unsqueeze(0)
    exponent = -0.5 * (diff.pow(2) / denom)
    scale = var_kern * torch.sqrt(ell_sq / denom)
    psi_1 = scale * torch.exp(exponent)

    # --- 2. Compute Mean ---
    # μ_f = Σ_j ψ₁,j · m_j  (under mean-field with normalized variational params)
    edge_means = (psi_1 * q_mu.unsqueeze(0)).sum(dim=-1)
    
    # --- 3. Compute Variance Components ---
    
    # Component A: Projected variance (standard mean-field)
    # σ²_proj = Σ_j ψ₁,j² (s_j + m_j²) - μ_f²
    var_projected = (psi_1.pow(2) * (q_var.unsqueeze(0) + q_mu.unsqueeze(0).pow(2))).sum(dim=-1)
    var_projected = var_projected - edge_means.pow(2)
    var_projected = torch.clamp(var_projected, min=0)
    
    # Component B: Orthogonal variance (Nyström approximation error)
    # 
    # Mathematically correct formula:
    #   trace = ||ψ₁||² / σ_f⁴
    #   σ²_⊥ = σ_f² (1 - trace)
    #
    # When trace > 1, the mean-field approximation is breaking down.
    # We clamp to [0,1] as a heuristic that still gives reasonable behavior.
    
    k_xx = var_kern.squeeze(-1)  # [1, Out, In], this is σ_f²
    
    # CORRECTED normalization: divide by σ_f⁴
    psi1_sq_sum = psi_1.pow(2).sum(dim=-1)  # ||ψ₁||²
    k_xx_sq = k_xx.pow(2)  # σ_f⁴
    
    trace_approx_raw = psi1_sq_sum / (k_xx_sq + 1e-8)
    
    # Clamp to [0, 1] - values outside indicate approximation breakdown
    # This is the heuristic that maintains backward compatibility
    trace_approx = torch.clamp(trace_approx_raw, min=0.0, max=1.0)
    var_ortho = k_xx * (1.0 - trace_approx)
    
    # Total variance = projected + orthogonal
    edge_vars = var_projected + var_ortho
    
    if return_diagnostics:
        diagnostics = {
            'trace_raw_mean': trace_approx_raw.mean().item(),
            'trace_raw_max': trace_approx_raw.max().item(),
            'trace_raw_min': trace_approx_raw.min().item(),
            'n_clamped_high': (trace_approx_raw > 1.0).sum().item(),
            'n_clamped_low': (trace_approx_raw < 0.0).sum().item(),
            'max_clamp_excess': (trace_approx_raw - 1.0).clamp(min=0).max().item(),
            'var_ortho_mean': var_ortho.mean().item(),
            'var_projected_mean': var_projected.mean().item(),
            'psi1_max': psi_1.max().item(),
            'k_xx_mean': k_xx.mean().item(),
        }
        return edge_means, torch.clamp(edge_vars, min=1e-6), diagnostics
    
    return edge_means, torch.clamp(edge_vars, min=1e-6)


def cosine_moment_matching(x_mean, x_var, z, q_mu, q_var, period, variance,
                           return_diagnostics=False):
    """
    Analytic moment matching for Cosine Kernel with orthogonal variance correction.
    
    Uses the same corrected normalization as rbf_moment_matching.
    """
    x_mu_exp = x_mean.unsqueeze(1).unsqueeze(-1)
    x_var_exp = x_var.unsqueeze(1).unsqueeze(-1)
    var_kern = variance.unsqueeze(0).unsqueeze(-1)
    
    omega = (2 * math.pi) / period.unsqueeze(0).unsqueeze(-1)

    damping = torch.exp(-0.5 * omega.pow(2) * x_var_exp)
    dist = x_mu_exp - z.unsqueeze(0)
    cosine_term = torch.cos(omega * dist)
    
    psi_1 = var_kern * damping * cosine_term
    
    # Mean (unchanged)
    edge_means = (psi_1 * q_mu.unsqueeze(0)).sum(dim=-1)
    
    # Projected variance
    var_projected = (psi_1.pow(2) * (q_var.unsqueeze(0) + q_mu.unsqueeze(0).pow(2))).sum(dim=-1)
    var_projected = var_projected - edge_means.pow(2)
    var_projected = torch.clamp(var_projected, min=0)
    
    # Orthogonal variance with CORRECTED normalization
    k_xx = var_kern.squeeze(-1)
    k_xx_sq = k_xx.pow(2)
    psi1_sq_sum = psi_1.pow(2).sum(dim=-1)
    
    trace_approx_raw = psi1_sq_sum / (k_xx_sq + 1e-8)
    trace_approx = torch.clamp(trace_approx_raw, min=0.0, max=1.0)
    var_ortho = k_xx * (1.0 - trace_approx)
    
    edge_vars = var_projected + var_ortho
    
    if return_diagnostics:
        diagnostics = {
            'trace_raw_mean': trace_approx_raw.mean().item(),
            'trace_raw_max': trace_approx_raw.max().item(),
            'n_clamped_high': (trace_approx_raw > 1.0).sum().item(),
            'max_clamp_excess': (trace_approx_raw - 1.0).clamp(min=0).max().item(),
        }
        return edge_means, torch.clamp(edge_vars, min=1e-6), diagnostics
    
    return edge_means, torch.clamp(edge_vars, min=1e-6)


# =============================================================================
# Mean-field approximation utilities
# =============================================================================

def compute_recommended_num_inducing(z_min, z_max, lengthscale, min_separation_factor=2.0):
    """
    Compute the maximum number of inducing points that satisfies mean-field assumptions.
    
    For the mean-field approximation to be valid, inducing points should be separated
    by at least `min_separation_factor * lengthscale`.
    
    Args:
        z_min: Minimum inducing point location
        z_max: Maximum inducing point location
        lengthscale: Kernel lengthscale
        min_separation_factor: Minimum separation in units of lengthscale (default: 2.0)
        
    Returns:
        max_inducing: Maximum number of inducing points
    """
    domain_size = z_max - z_min
    min_spacing = min_separation_factor * lengthscale
    
    if min_spacing <= 0:
        return 1
    
    # Number of points = floor(domain_size / min_spacing) + 1
    max_inducing = int(domain_size / min_spacing) + 1
    return max(2, max_inducing)  # At least 2 inducing points


def check_mean_field_validity(z, lengthscale, min_separation_factor=2.0):
    """
    Check if inducing point configuration satisfies mean-field assumptions.
    
    Args:
        z: [Out, In, M] or [M] inducing point locations
        lengthscale: [Out, In] or scalar kernel lengthscales
        min_separation_factor: Minimum separation in units of lengthscale
        
    Returns:
        is_valid: bool, True if assumptions are satisfied
        min_ratio: float, minimum separation ratio (want > min_separation_factor)
        details: dict with diagnostic information
    """
    # Handle different input shapes
    if z.ndim == 1:
        z = z.unsqueeze(0).unsqueeze(0)  # [1, 1, M]
    if not isinstance(lengthscale, torch.Tensor):
        lengthscale = torch.tensor(lengthscale)
    if lengthscale.ndim == 0:
        lengthscale = lengthscale.unsqueeze(0).unsqueeze(0)
    
    out_features, in_features, M = z.shape
    
    if M < 2:
        return True, float('inf'), {'message': 'Only 1 inducing point, no separation check needed'}
    
    min_ratio = float('inf')
    violations = []
    
    for i in range(out_features):
        for j in range(in_features):
            z_edge = z[i, j]  # [M]
            ell = lengthscale[i, j].item() if lengthscale.ndim > 0 else lengthscale.item()
            
            # Compute pairwise distances
            z_sorted, _ = torch.sort(z_edge)
            spacings = z_sorted[1:] - z_sorted[:-1]
            min_spacing = spacings.min().item()
            
            ratio = min_spacing / (ell + 1e-8)
            min_ratio = min(min_ratio, ratio)
            
            if ratio < min_separation_factor:
                violations.append({
                    'edge': (i, j),
                    'min_spacing': min_spacing,
                    'lengthscale': ell,
                    'ratio': ratio,
                    'required': min_separation_factor
                })
    
    is_valid = len(violations) == 0
    
    details = {
        'is_valid': is_valid,
        'min_ratio': min_ratio,
        'required_ratio': min_separation_factor,
        'n_violations': len(violations),
        'violations': violations[:5] if violations else [],  # First 5 violations
    }
    
    return is_valid, min_ratio, details


def adjust_inducing_points_for_mean_field(num_inducing, z_min, z_max, lengthscale,
                                          min_separation_factor=2.0, verbose=True):
    """
    Adjust number of inducing points to satisfy mean-field assumptions.
    
    Args:
        num_inducing: Requested number of inducing points
        z_min: Minimum inducing point location
        z_max: Maximum inducing point location  
        lengthscale: Initial kernel lengthscale
        min_separation_factor: Minimum separation in units of lengthscale
        verbose: Print info if adjustment is made
        
    Returns:
        adjusted_num: Adjusted number of inducing points
        was_adjusted: bool, True if adjustment was made
    """
    max_inducing = compute_recommended_num_inducing(
        z_min, z_max, lengthscale, min_separation_factor
    )
    
    if num_inducing <= max_inducing:
        return num_inducing, False
    
    if verbose:
        # Toned-down informational message (not a scary warning)
        print(
            f"\n  [strict_mean_field] Adjusting inducing points: {num_inducing} → {max_inducing}\n"
            f"  (Based on initial lengthscale={lengthscale:.3f}, domain=[{z_min:.1f}, {z_max:.1f}])"
        )
    
    return max_inducing, True


def check_mean_field_assumptions(z, lengthscale, variance, x_var=None, verbose=True):
    """
    Check mean-field approximation status.
    
    This is informational - the clamping heuristic handles violations gracefully.
    
    Assumptions checked:
    1. Inducing points are well-separated: |Z_i - Z_j| > 2ℓ for i ≠ j
    2. Input variance is small: σ_x² < ℓ²
    3. Signal variance is reasonable: σ_f² ∈ [0.01, 10] typical
    
    Args:
        z: [Out, In, M] inducing point locations
        lengthscale: [Out, In] kernel lengthscales
        variance: [Out, In] kernel signal variances
        x_var: [Batch, In] input variances (optional)
        verbose: print diagnostic messages
        
    Returns:
        dict with diagnostic information
    """
    out_features, in_features, M = z.shape
    
    diagnostics = {
        'separation_ok': True,
        'input_var_ok': True,
        'variance_ok': True,
        'details': []
    }
    
    # Check inducing point separation
    is_valid, min_ratio, sep_details = check_mean_field_validity(z, lengthscale)
    diagnostics['separation_ok'] = is_valid
    diagnostics['min_separation_ratio'] = min_ratio
    
    if not is_valid:
        diagnostics['details'].append(
            f"Separation ratio: {min_ratio:.2f} (ideal > 2.0) - using clamped approximation"
        )
    
    # Check signal variance
    var_min = variance.min().item()
    var_max = variance.max().item()
    diagnostics['variance_range'] = (var_min, var_max)
    
    if var_max > 10 or var_min < 0.01:
        diagnostics['variance_ok'] = False
        diagnostics['details'].append(
            f"Signal variance outside typical range: [{var_min:.3f}, {var_max:.3f}]"
        )
    
    # Check input variance if provided
    if x_var is not None:
        ell_sq = lengthscale.pow(2)
        x_var_mean = x_var.mean(dim=0)  # [In]
        
        for j in range(in_features):
            ell_sq_avg = ell_sq[:, j].mean().item()
            x_var_j = x_var_mean[j].item()
            
            if x_var_j > ell_sq_avg:
                diagnostics['input_var_ok'] = False
                diagnostics['details'].append(
                    f"Input {j}: σ_x² = {x_var_j:.4f} > ℓ² = {ell_sq_avg:.4f}"
                )
    
    diagnostics['all_ok'] = (
        diagnostics['separation_ok'] and 
        diagnostics['input_var_ok'] and 
        diagnostics['variance_ok']
    )
    
    if verbose:
        if diagnostics['all_ok']:
            print(f"Mean-field approximation: ✓ Ideal conditions met")
        else:
            print(f"Mean-field approximation: Using clamped mode (works well in practice)")
            if diagnostics['details']:
                for detail in diagnostics['details']:
                    print(f"  - {detail}")
        print(f"  Min separation ratio (want > 2): {diagnostics.get('min_separation_ratio', 'N/A'):.2f}")
    
    return diagnostics
