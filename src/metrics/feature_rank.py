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

    z_c = z - z.mean(dim=0, keepdim=True)
    n = z_c.shape[0]

    cov = (z_c.T @ z_c) / n
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=1e-12)

    s1 = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    participation_ratio = (s1 ** 2) / s2

    _, singular_values, _ = torch.linalg.svd(z_c, full_matrices=False)
    singular_values = singular_values.clamp(min=1e-12)
    nuclear_norm = singular_values.sum()
    frobenius_norm = torch.linalg.norm(z_c, "fro").clamp(min=1e-12)
    nuclear_norm_ratio = nuclear_norm / frobenius_norm

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
