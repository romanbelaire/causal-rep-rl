"""Held-out aliasing diagnostics. Independent of the training hinge pair sample."""

import torch

from src.losses.separation import ALPHA_SEP_CANDIDATES, SEP_SR_EPS, pair_derangement

SOFT_DZ_EPS = 1e-6


def _pair_spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation of two 1-d tensors. Undefined if either is constant."""
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise RuntimeError(
            f"pair spearman expects matching 1-d tensors, got {tuple(x.shape)} vs {tuple(y.shape)}"
        )
    if x.shape[0] < 3:
        raise RuntimeError(f"pair spearman needs >= 3 pairs, got {x.shape[0]}")
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float((rx.norm() * ry.norm()).item())
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum().item() / denom)


def _nan_scores(out: dict[str, float]) -> None:
    out["alias_rate"] = float("nan")
    out["alias_soft"] = float("nan")
    out["alias_hinge_mean"] = float("nan")
    out["alias_value_spearman"] = float("nan")
    out["alias_hinge_active_frac"] = float("nan")
    out["alias_skipped"] = 1.0
    for a in ALPHA_SEP_CANDIDATES:
        key = str(a).replace(".", "p")
        out[f"alias_hinge_active_frac_a{key}"] = float("nan")


def aliasing_diagnostics(
    z: torch.Tensor,
    returns: torch.Tensor,
    alpha_sep: float,
    s_r_eps: float = SEP_SR_EPS,
) -> dict[str, float]:
    """Pairwise aliasing on a fresh derangement.

    Hard rate (secondary): fraction of pairs with |ΔR̂| at or above the median
    and ||Δz||/s_Z < α. That 0/1 event sits inside the hinge-active set.

    Soft score (α-free): median of (|ΔR̂|/s_R) / (||Δz||/s_Z + ε) on the same
    above-median |ΔR̂| pairs. Large means return gaps sit at small latent gaps.

    Value Spearman (α-free): rank correlation of ||Δz|| with |ΔR̂| over all pairs.
    Positive means local geometry tracks return differences.
    """
    z = z.detach()
    returns = returns.detach()
    if z.ndim != 2:
        raise RuntimeError(f"z must be [N, d], got {tuple(z.shape)}")
    if returns.ndim != 1 or returns.shape[0] != z.shape[0]:
        raise RuntimeError(
            f"returns must be [N] matching z, got {tuple(returns.shape)} vs {tuple(z.shape)}"
        )
    pair_i, pair_j = pair_derangement(z.shape[0], z.device)
    dist = (z[pair_i] - z[pair_j]).pow(2).sum(dim=1).sqrt()
    d_r = (returns[pair_i] - returns[pair_j]).abs()
    s_z = float(dist.median().item())
    s_r = float(d_r.median().item())
    out = {
        "alias_s_z": s_z,
        "alias_s_r": s_r,
        "alias_n_pairs": float(dist.shape[0]),
    }
    if s_r < s_r_eps:
        out["alias_dz_norm_p25"] = float("nan")
        _nan_scores(out)
        return out
    d_r_n = d_r / s_r
    above_med = d_r >= s_r
    if s_z <= 0.0:
        # Median pairwise distance 0: latent is degenerate while returns still vary.
        out["alias_dz_norm_p25"] = 0.0
        out["alias_rate"] = 1.0
        out["alias_soft"] = float((d_r_n[above_med] / SOFT_DZ_EPS).median().item())
        out["alias_hinge_mean"] = float((alpha_sep * d_r_n).mean().item())
        out["alias_value_spearman"] = 0.0
        out["alias_hinge_active_frac"] = 1.0
        out["alias_skipped"] = 0.0
        for a in ALPHA_SEP_CANDIDATES:
            key = str(a).replace(".", "p")
            out[f"alias_hinge_active_frac_a{key}"] = 1.0
        return out
    d_z_n = dist / s_z
    out["alias_dz_norm_p25"] = float(torch.quantile(d_z_n, 0.25).item())
    aliased = above_med & (d_z_n < alpha_sep)
    ratio = d_r_n / (d_z_n + SOFT_DZ_EPS)
    slack = (alpha_sep * d_r_n - d_z_n).clamp_min(0.0)
    out["alias_rate"] = float(aliased.float().mean().item())
    out["alias_soft"] = float(ratio[above_med].median().item())
    out["alias_hinge_mean"] = float(slack.mean().item())
    out["alias_value_spearman"] = _pair_spearman(dist, d_r)
    out["alias_hinge_active_frac"] = float((slack > 0).float().mean().item())
    out["alias_skipped"] = 0.0
    for a in ALPHA_SEP_CANDIDATES:
        key = str(a).replace(".", "p")
        out[f"alias_hinge_active_frac_a{key}"] = float(
            ((a * d_r_n - d_z_n) > 0).float().mean().item()
        )
    return out
