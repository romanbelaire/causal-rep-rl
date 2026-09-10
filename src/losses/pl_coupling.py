"""PL coupling hinge loss: value-suboptimality PL ratio vs mu_0."""

import torch
import torch.nn as nn

from src.metrics.pl_ratio import (
    effective_f_floor,
    median_pairwise_distance,
    value_range_scale,
)
from src.utils.batched_grad import batched_value_head_grad_z


def _value_head_output(critic: nn.Module, z: torch.Tensor) -> torch.Tensor:
    return critic.value_head(z).squeeze(-1)


def _grad_v_wrt_z(critic: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Per-sample dV/dZ rows [N, d]; Z retains connection to encoder."""
    return batched_value_head_grad_z(critic, z)


def compute_pl_coupling_loss(
    critic: nn.Module,
    z: torch.Tensor,
    v_ref: float,
    mu_0: float = 0.1,
    f_floor: float = 1e-3,
    scale_invariant: bool = False,
    value_normalize: bool = False,
    pair_n: int = 4096,
    f_mode: str = "exclude",
    f_min: float = 1e-3,
) -> tuple[torch.Tensor, dict]:
    """
    L_PL = mean(max(0, mu_0 - pl_ratio)).

    f_mode:
      "exclude" (legacy): f_raw = v_ref - V; drop states with f_raw <= tau where
        tau = max(1e-12, f_floor * |v_ref|). Gradients only through Grad V.
      "clamp" (anti-aliased PPO): f = (v_ref - V).clamp_min(f_min) over all states.

    If scale_invariant, the hinge uses mu_PL * L^2 with L = stopgrad(median
    pairwise distance). If value_normalize, divide by stopgrad(V range) as well.
    """
    if f_mode not in ("exclude", "clamp"):
        raise ValueError(f"f_mode must be 'exclude' or 'clamp' (got {f_mode!r})")

    v = _value_head_output(critic, z)
    grad_z = _grad_v_wrt_z(critic, z)
    grad_sq = grad_z.pow(2).sum(dim=1)

    with torch.no_grad():
        L = median_pairwise_distance(z.detach(), n_pairs=pair_n)
        V_scale = value_range_scale(v.detach())

    if f_mode == "clamp":
        f = (v_ref - v).detach().clamp_min(f_min)
        pl_ratio = grad_sq / (2.0 * f)
        valid_fraction = 1.0
        f_floor_eff = float(f_min)
        floor_rate = 0.0
    else:
        f_floor_eff = effective_f_floor(v_ref, f_floor_rel=f_floor)
        f_raw = (v_ref - v).detach()
        valid = f_raw > f_floor_eff
        valid_fraction = valid.float().mean()
        floor_rate = (1.0 - valid_fraction).item()
        if not valid.any():
            loss = grad_sq.sum() * 0.0
            nan = float("nan")
            return loss, {
                "train_pl_loss": 0.0,
                "train_pl_ratio_mean": nan,
                "train_pl_ratio_q05": nan,
                "train_f_mean": nan,
                "train_v_ref": float(v_ref),
                "train_f_floor_eff": f_floor_eff,
                "train_f_floor_rate": 1.0,
                "train_pl_valid_fraction": 0.0,
                "train_pl_scale_invariant": float(scale_invariant),
                "train_pl_value_normalize": float(value_normalize),
                "train_pl_f_mode": 0.0,
            }
        f = f_raw[valid]
        pl_ratio = grad_sq[valid] / (2.0 * f)

    if scale_invariant:
        pl_ratio = pl_ratio * (L.detach() ** 2)
        if value_normalize:
            pl_ratio = pl_ratio / V_scale.detach()
    hinge = torch.relu(mu_0 - pl_ratio)
    loss = hinge.mean()

    stats = {
        "train_pl_loss": loss.item(),
        "train_pl_ratio_mean": pl_ratio.detach().mean().item(),
        "train_pl_ratio_q05": torch.quantile(pl_ratio.detach(), 0.05).item(),
        "train_f_mean": f.mean().item() if torch.is_tensor(f) else float(f),
        "train_v_ref": float(v_ref),
        "train_f_floor_eff": f_floor_eff,
        "train_f_floor_rate": floor_rate,
        "train_pl_valid_fraction": float(valid_fraction)
        if not torch.is_tensor(valid_fraction)
        else valid_fraction.item(),
        "train_pl_L": float(L.item()),
        "train_pl_value_scale": float(V_scale.item()),
        "train_pl_scale_invariant": float(scale_invariant),
        "train_pl_value_normalize": float(value_normalize),
        "train_pl_f_mode": 1.0 if f_mode == "clamp" else 0.0,
    }
    return loss, stats
