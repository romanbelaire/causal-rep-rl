"""
Effective feature rank metrics for representation collapse diagnostics.

Participation ratio and nuclear-norm ratio follow Lyle et al. (ICLR 2022) /
Moalla et al. (2025) capacity measures on latent batches Z ∈ R^{N×d}.
"""

import torch

# Spectral rank: singular values above 1% of max. PCA rank: 0.1% of max (more lenient).
SPECTRAL_RANK_DELTA = 0.01
PCA_RANK_DELTA = 0.01  # applied to singular values of centered Z


def compute_feature_rank_metrics(z: torch.Tensor) -> dict[str, float]:
    """
    Compute effective feature rank from a batch of latent representations.

    Args:
        z: Latent representations [N, d]

    Returns:
        participation_ratio: (tr Σ)² / tr(Σ²) on sample covariance Σ
        nuclear_norm_ratio: ||Z||_* / ||Z||_F on centered Z
        log_effective_rank_pr: log(participation_ratio)
        log_effective_rank_nuclear: log(nuclear_norm_ratio²)  (squared ratio ≈ effective rank)
        feature_rank_spectral: count of singular values above SPECTRAL_RANK_DELTA × max
        feature_rank_pca: count of singular values above PCA_RANK_DELTA × max
        log_effective_feature_rank_pca: log(feature_rank_pca)
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)

    # CPU: logging-only; avoids flaky cuSOLVER on shared V100s (CUSOLVER_STATUS_INTERNAL_ERROR).
    z = z.detach().float().cpu()
    z_c = z - z.mean(dim=0, keepdim=True)
    frobenius_norm = torch.linalg.norm(z_c, "fro")
    if frobenius_norm.item() < 1e-12:
        return {
            "feature_rank_participation_ratio": 0.0,
            "feature_rank_nuclear_norm_ratio": 0.0,
            "log_effective_feature_rank_pr": float("-inf"),
            "log_effective_feature_rank_nuclear": float("-inf"),
            "feature_rank_spectral": 0.0,
            "feature_rank_pca": 0.0,
            "log_effective_feature_rank_pca": float("-inf"),
        }

    # Participation ratio via SVD of Z_c (cov eigenvalues λ_i = σ_i² / N).
    # Avoids eigvalsh, which fails on collapsed / ill-conditioned cov.
    _, singular_values, _ = torch.linalg.svd(z_c, full_matrices=False)
    singular_values = singular_values.clamp(min=0.0)
    s2 = (singular_values ** 2).sum()
    s4 = (singular_values ** 4).sum().clamp(min=1e-30)
    participation_ratio = (s2 ** 2) / s4

    singular_values = singular_values.clamp(min=1e-12)
    nuclear_norm = singular_values.sum()
    nuclear_norm_ratio = nuclear_norm / frobenius_norm.clamp(min=1e-12)

    spectral_threshold = SPECTRAL_RANK_DELTA * singular_values.max()
    spectral_rank = (singular_values > spectral_threshold).sum().float()

    pca_threshold = PCA_RANK_DELTA * singular_values.max()
    pca_rank = (singular_values > pca_threshold).sum().float()

    return {
        "feature_rank_participation_ratio": participation_ratio.item(),
        "feature_rank_nuclear_norm_ratio": nuclear_norm_ratio.item(),
        "log_effective_feature_rank_pr": torch.log(participation_ratio).item(),
        "log_effective_feature_rank_nuclear": torch.log(nuclear_norm_ratio ** 2).item(),
        "feature_rank_spectral": spectral_rank.item(),
        "feature_rank_pca": pca_rank.item(),
        "log_effective_feature_rank_pca": torch.log(pca_rank).item(),
    }
