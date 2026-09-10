"""Frozen reference-batch validity: hashes, old-distance quantiles, latent scale."""

import hashlib

import torch

from src.metrics.feature_rank import compute_feature_rank_metrics


def _row_hash(row: torch.Tensor) -> str:
    return hashlib.sha1(row.contiguous().cpu().numpy().tobytes()).hexdigest()


def _quantile_dict(x: torch.Tensor, prefix: str) -> dict[str, float]:
    x = x.detach().float().reshape(-1)
    if x.numel() == 0:
        raise RuntimeError(f"{prefix}: empty tensor for quantiles")
    qs = {
        "min": 0.0,
        "p01": 0.01,
        "p05": 0.05,
        "p50": 0.50,
        "p95": 0.95,
        "max": 1.0,
    }
    out = {}
    for name, q in qs.items():
        out[f"{prefix}_{name}"] = float(torch.quantile(x, q).item())
    return out


def reference_validity_stats(
    obs: torch.Tensor,
    z: torch.Tensor,
    min_old_distance: float,
    freeze_epoch: int,
    latent_name: str,
) -> dict[str, float | int | str]:
    """Diagnostics for a frozen raw-obs reference set and its current latents."""
    if obs.shape[0] != z.shape[0]:
        raise RuntimeError(
            f"reference obs n={obs.shape[0]} != latent n={z.shape[0]}"
        )
    n = int(obs.shape[0])
    if n < 2:
        raise RuntimeError(f"reference validity needs at least 2 states, got {n}")
    hashes = [_row_hash(obs[i]) for i in range(n)]
    unique = len(set(hashes))
    d = torch.cdist(z.detach().float(), z.detach().float(), p=2)
    eye = torch.eye(n, dtype=torch.bool, device=d.device)
    off = d[~eye]
    usable = (~eye) & (d >= min_old_distance)
    n_off = int((~eye).sum().item())
    n_usable = int(usable.sum().item())
    norms = z.detach().float().norm(dim=1)
    z_c = z.detach().float() - z.detach().float().mean(dim=0, keepdim=True)
    _, singular, _ = torch.linalg.svd(z_c.cpu(), full_matrices=False)
    eig = (singular.clamp(min=0.0) ** 2) / n
    coord_std = z.detach().float().std(dim=0, unbiased=False)
    rank = compute_feature_rank_metrics(z)
    out: dict[str, float | int | str] = {
        "ref_n": n,
        "ref_unique_count": unique,
        "ref_duplicate_frac": 1.0 - unique / n,
        "ref_freeze_epoch": int(freeze_epoch),
        "ref_latent_name": latent_name,
        "dz_rel_n_off_pairs": float(n_off),
        "dz_rel_n_usable_pairs": float(n_usable),
        "dz_rel_usable_frac": n_usable / n_off,
        "dz_rel_all_excluded": float(n_usable == 0),
        "cov_eig_min": float(eig.min().item()),
        "cov_eig_max": float(eig.max().item()),
        "coord_std_min": float(coord_std.min().item()),
        "coord_std_max": float(coord_std.max().item()),
    }
    out.update(_quantile_dict(off, "d_old"))
    out.update(_quantile_dict(norms, "latent_norm"))
    out.update({k: float(v) for k, v in rank.items()})
    return out


def covariance_floor_stats(z: torch.Tensor, eig_floor: float) -> dict[str, float]:
    """Secondary global-collapse diagnostic. Not a substitute for L_sep."""
    n = z.shape[0]
    z_c = z.detach().float() - z.detach().float().mean(dim=0, keepdim=True)
    _, singular, _ = torch.linalg.svd(z_c.cpu(), full_matrices=False)
    eig = (singular.clamp(min=0.0) ** 2) / n
    min_eig = float(eig.min().item())
    return {
        "cov_eig_min": min_eig,
        "cov_eig_floor": float(eig_floor),
        "cov_eig_below_floor": float(min_eig < eig_floor),
    }
