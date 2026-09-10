"""Finite-reference relational distortion D_Z,infinity^B."""

import torch


def relational_distortion(
    z_new: torch.Tensor,
    z_old: torch.Tensor,
    min_old_distance: float,
) -> tuple[torch.Tensor, dict]:
    if min_old_distance <= 0:
        raise RuntimeError(f"min_old_distance must be > 0, got {min_old_distance}")
    n = z_old.shape[0]
    if n < 2:
        raise RuntimeError("relational metric needs at least 2 reference states")
    d_old = torch.cdist(z_old, z_old, p=2)
    d_new = torch.cdist(z_new, z_new, p=2)
    eye = torch.eye(n, device=z_old.device, dtype=torch.bool)
    off = ~eye
    valid = off & (d_old >= min_old_distance)
    n_off = int(off.sum().item())
    n_valid = int(valid.sum().item())
    excluded_frac = 1.0 - (n_valid / n_off)
    if n_valid == 0:
        nan = torch.tensor(float("nan"), device=z_old.device)
        stats = {
            "D_Z_inf_B": nan,
            "dz_rel_excluded_frac": torch.tensor(excluded_frac, device=z_old.device),
            "dz_rel_p50": nan,
            "dz_rel_p90": nan,
            "dz_rel_n_valid_pairs": torch.tensor(0.0, device=z_old.device),
            "dz_rel_all_excluded": torch.tensor(1.0, device=z_old.device),
        }
        return nan, stats
    ratio = d_new[valid] / d_old[valid]
    metric = (ratio - 1.0).abs().max()
    abs_dev = (ratio - 1.0).abs()
    stats = {
        "D_Z_inf_B": metric.detach(),
        "dz_rel_excluded_frac": torch.tensor(excluded_frac, device=z_old.device),
        "dz_rel_p50": torch.quantile(abs_dev, 0.50).detach(),
        "dz_rel_p90": torch.quantile(abs_dev, 0.90).detach(),
        "dz_rel_n_valid_pairs": torch.tensor(float(n_valid), device=z_old.device),
        "dz_rel_all_excluded": torch.tensor(0.0, device=z_old.device),
    }
    return metric, stats
