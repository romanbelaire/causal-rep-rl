"""Replay priorities and the two distinct corrections (IS vs behavior ratio)."""

import torch


def priority_scores(
    ig: torch.Tensor,
    gap: torch.Tensor,
    pair: torch.Tensor,
    geom: torch.Tensor,
    floor: float,
    lambda_ig: float,
    lambda_gap: float,
    lambda_pair: float,
    lambda_geom: float,
) -> torch.Tensor:
    if floor <= 0:
        raise RuntimeError(f"priority floor must be > 0, got {floor}")
    p = (
        floor
        + lambda_ig * ig
        + lambda_gap * gap
        + lambda_pair * pair
        + lambda_geom * geom
    )
    if (p <= 0).any():
        raise RuntimeError("priority scores must be strictly positive")
    return p


def mix_coverage(priority: torch.Tensor, coverage_mix: float) -> torch.Tensor:
    """q = (1-beta_cov) q_priority + beta_cov uniform. Every item keeps mass."""
    if coverage_mix <= 0 or coverage_mix >= 1:
        raise RuntimeError(f"coverage_mix must be in (0,1), got {coverage_mix}")
    q_pri = priority / priority.sum()
    q_cover = torch.full_like(q_pri, 1.0 / q_pri.numel())
    mixed = (1.0 - coverage_mix) * q_pri + coverage_mix * q_cover
    if (mixed <= 0).any():
        raise RuntimeError("coverage mixture assigned a zero probability")
    return mixed


def replay_is_weights(sample_p: torch.Tensor, n: int, exponent: float) -> torch.Tensor:
    """Replay-sampling correction (N p_i)^{-alpha_is}. Not a behavior-policy ratio."""
    if exponent < 0:
        raise RuntimeError(f"importance_exponent must be >= 0, got {exponent}")
    w = (n * sample_p).pow(-exponent)
    return w / w.mean()


def behavior_policy_ratio(
    target_log_prob: torch.Tensor, behavior_log_prob: torch.Tensor
) -> torch.Tensor:
    """pi_target(a|z) / beta(a|z). Separate from replay IS. Not used for v1 one-step TD."""
    return torch.exp(target_log_prob - behavior_log_prob)
