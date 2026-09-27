"""
KL Divergence computation for Sparse Variational Gaussian Processes.

Implements: KL[q(u)||p(u)] where:
- q(u) = N(m, S) is the variational posterior (diagonal covariance)
- p(u) = N(0, K_ZZ) is the GP prior over inducing points

Formula: KL = 0.5 * (tr(K_ZZ^{-1} S) + m^T K_ZZ^{-1} m - M + log(|K_ZZ|/|S|))
"""

import torch
import torch.nn as nn


def rbf_kernel_matrix(z, lengthscale, variance, jitter=1e-4):
    """
    Compute RBF kernel matrix K_ZZ for inducing points.
    
    K(z_i, z_j) = variance * exp(-0.5 * (z_i - z_j)^2 / lengthscale^2)
    
    Args:
        z: [M] inducing points
        lengthscale: scalar lengthscale
        variance: scalar signal variance
        jitter: numerical stability term added to diagonal
    
    Returns:
        K_ZZ: [M, M] kernel matrix
    """
    M = z.shape[0]
    
    # Compute pairwise squared distances
    dist_sq = (z.unsqueeze(0) - z.unsqueeze(1)).pow(2)
    
    # RBF kernel
    K_ZZ = variance * torch.exp(-0.5 * dist_sq / (lengthscale.pow(2) + 1e-8))
    
    # Add jitter for numerical stability
    K_ZZ = K_ZZ + torch.eye(M, device=K_ZZ.device, dtype=K_ZZ.dtype) * jitter
    
    return K_ZZ


def compute_kl_univariate(q_mu, q_var, z, lengthscale, variance, jitter=1e-4):
    """
    Compute KL divergence for a single univariate GP edge.
    
    KL[q(u)||p(u)] for one edge function.
    
    Args:
        q_mu: [M] variational mean
        q_var: [M] variational variance (diagonal)
        z: [M] inducing points
        lengthscale: scalar kernel lengthscale
        variance: scalar kernel signal variance
        jitter: numerical stability parameter
    
    Returns:
        kl: scalar KL divergence
    """
    M = q_mu.shape[0]
    
    # Compute prior covariance K_ZZ
    K_ZZ = rbf_kernel_matrix(z, lengthscale, variance, jitter=jitter)
    
    try:
        # Cholesky decomposition for numerical stability
        # K_ZZ = L L^T
        L = torch.linalg.cholesky(K_ZZ)
        
        # --- Term 1: tr(K_ZZ^{-1} S) ---
        # S is diagonal, so tr(K_ZZ^{-1} S) = sum_i K_ZZ^{-1}[i,i] * S[i,i]
        # Compute K_ZZ^{-1} diagonal efficiently
        K_inv_diag = torch.cholesky_solve(
            torch.eye(M, device=K_ZZ.device, dtype=K_ZZ.dtype), 
            L
        ).diag()
        trace_term = (K_inv_diag * q_var).sum()
        
        # --- Term 2: m^T K_ZZ^{-1} m ---
        # Solve K_ZZ * x = m for x, then compute m^T x
        K_inv_m = torch.cholesky_solve(q_mu.unsqueeze(1), L).squeeze()
        quad_term = (q_mu * K_inv_m).sum()
        
        # --- Term 3: log determinant terms ---
        # log |K_ZZ| = 2 * sum(log(diag(L)))
        log_det_K = 2.0 * L.diag().log().sum()
        
        # log |S| where S is diagonal
        log_det_S = q_var.log().sum()
        
        log_det_term = log_det_K - log_det_S
        
        # --- Final KL ---
        kl = 0.5 * (trace_term + quad_term - M + log_det_term)
        
        # Clamp to prevent numerical issues
        kl = torch.clamp(kl, min=0.0)
        
    except RuntimeError as e:
        # If Cholesky fails (shouldn't happen with jitter, but just in case)
        print(f"Warning: Cholesky decomposition failed in KL computation: {e}")
        # Return a large penalty to discourage this configuration
        kl = torch.tensor(1000.0, device=q_mu.device, dtype=q_mu.dtype)
    
    return kl


def compute_kl_divergence(q_mu, q_log_var, z, lengthscale, variance, jitter=1e-4):
    """
    Compute total KL divergence for all edges in a GPKANLayer.
    
    Sum over all (out_features, in_features) edges.
    
    Args:
        q_mu: [out_features, in_features, M] variational means
        q_log_var: [out_features, in_features, M] variational log variances
        z: [out_features, in_features, M] inducing points
        lengthscale: [out_features, in_features] kernel lengthscales
        variance: [out_features, in_features] kernel signal variances
        jitter: numerical stability parameter
    
    Returns:
        total_kl: scalar, sum of KL divergences across all edges
    """
    out_features, in_features, M = q_mu.shape
    
    # Convert log variance to variance
    q_var = torch.exp(q_log_var).clamp(min=1e-6)
    
    total_kl = 0.0
    
    # Compute KL for each edge independently
    # This parallelizes well but we do it sequentially for clarity
    for i in range(out_features):
        for j in range(in_features):
            kl_edge = compute_kl_univariate(
                q_mu=q_mu[i, j],
                q_var=q_var[i, j],
                z=z[i, j],
                lengthscale=lengthscale[i, j],
                variance=variance[i, j],
                jitter=jitter
            )
            total_kl = total_kl + kl_edge
    
    # Average over number of edges for stability
    # This makes the KL scale less sensitive to network width
    total_kl = total_kl / (out_features * in_features)
    
    return total_kl
