"""Actor-feature geometry diagnostics for ALE Phase 1."""

from __future__ import annotations

import torch

from src.metrics.collapse_probes import dormant_unit_fraction, near_zero_pair_rate
from src.metrics.feature_rank import compute_feature_rank_metrics


def preactivation_norm(phi: torch.Tensor) -> float:
    return float(phi.detach().float().norm(dim=-1).mean().item())


def stable_rank(phi: torch.Tensor) -> float:
    """||A||_F^2 / ||A||_2^2 on centered features."""
    z = phi.detach().float().cpu()
    z = z - z.mean(dim=0, keepdim=True)
    fro2 = float((z * z).sum().item())
    if fro2 < 1e-20:
        return 0.0
    # spectral norm via top singular value
    s = torch.linalg.svdvals(z)
    top = float(s[0].item()) ** 2
    if top < 1e-20:
        return 0.0
    return fro2 / top


def feature_cosine_stats(phi: torch.Tensor) -> dict[str, float]:
    z = phi.detach().float()
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    # sample upper triangle of Gram if N large
    n = z.shape[0]
    if n > 256:
        idx = torch.randperm(n, device=z.device)[:256]
        z = z[idx]
        n = z.shape[0]
    g = z @ z.T
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=z.device), diagonal=1)
    vals = g[mask]
    return {
        "actor_cosine_mean": float(vals.mean().item()),
        "actor_cosine_abs_mean": float(vals.abs().mean().item()),
    }


def duplicate_feature_rate(phi: torch.Tensor, atol: float = 1e-5, max_n: int = 128) -> float:
    """Fraction of rows that match at least one earlier row (exact/near duplicates)."""
    z = phi.detach().float().cpu()
    if z.shape[0] > max_n:
        z = z[:max_n]
    n = z.shape[0]
    if n < 2:
        return 0.0
    dup = 0
    for i in range(1, n):
        d = (z[:i] - z[i]).abs().amax(dim=-1)
        if bool((d < atol).any().item()):
            dup += 1
    return float(dup) / float(n - 1)


def actor_geometry_metrics(
    phi: torch.Tensor,
    pair_i: torch.Tensor | None = None,
    pair_j: torch.Tensor | None = None,
    s_ref: float | None = None,
) -> dict[str, float]:
    phi = phi.detach()
    out: dict[str, float] = {}
    out["actor_preact_norm"] = preactivation_norm(phi)
    out["actor_dormant_frac"] = dormant_unit_fraction(phi)
    out["actor_stable_rank"] = stable_rank(phi)
    ranks = compute_feature_rank_metrics(phi)
    out["actor_feature_rank_pca"] = float(ranks["feature_rank_pca"])
    out["actor_feature_rank_pr"] = float(ranks["feature_rank_participation_ratio"])
    out.update({f"actor_{k}" if not k.startswith("actor_") else k: v for k, v in feature_cosine_stats(phi).items()})
    out["actor_duplicate_rate"] = duplicate_feature_rate(phi)
    if pair_i is not None and pair_j is not None and s_ref is not None and s_ref > 0:
        out["actor_C_0p01"] = near_zero_pair_rate(phi, pair_i, pair_j, s_ref, 0.01)
    # Consistency flag: dormant≈1 but batch variance not zero
    var_mean = float(phi.float().var(dim=0, unbiased=False).mean().item())
    if out["actor_dormant_frac"] > 0.99 and var_mean > 1e-4:
        out["actor_consistency_flag_dormant_vs_var"] = 1.0
    else:
        out["actor_consistency_flag_dormant_vs_var"] = 0.0
    out["actor_feature_var_mean"] = var_mean
    return out
