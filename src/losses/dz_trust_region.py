"""Reference-scaled log-geometry latent trust region D_Z.

D_Z^log = mean( (log(d_bar_new + delta) - log(d_bar_old + delta))^2 )
with d_bar = ||zi - zj|| / (sqrt(latent_dim) * s_ref), delta = 0.01 absolute,
and s_ref frozen at initialization.

Fix 2 (exclude floor-active pairs from the mean) is intentionally not implemented;
use only if log-ratio shows gradient pathology at tiny distances.
"""

import math

import torch


S_REF_DEGENERATE = 1e-8
DEFAULT_DZ_DELTA = 0.01


def pairwise_distances(z: torch.Tensor) -> torch.Tensor:
    """Full pairwise Euclidean distance matrix [N, N]."""
    return torch.cdist(z, z, p=2)


def sample_pair_index_mask(
    n: int,
    n_pairs: int,
    lambda_loc: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample pair indices (i, j) under mixture:
      nu = lambda_loc * local consecutive + (1-lambda_loc) * uniform global.

    Local pairs: consecutive transitions (i, i+1) for i in 0..n-2 (ref buffer order).
    Global pairs: independent uniform i != j.
    """
    n_local = int(round(lambda_loc * n_pairs))
    n_global = n_pairs - n_local
    if n < 2:
        raise ValueError("Need at least 2 reference states for D_Z pairs")

    if n_local > 0:
        starts = torch.randint(0, n - 1, (n_local,), device=device)
        i_loc = starts
        j_loc = starts + 1
    else:
        i_loc = torch.empty(0, dtype=torch.long, device=device)
        j_loc = torch.empty(0, dtype=torch.long, device=device)

    if n_global > 0:
        i_g = torch.randint(0, n, (n_global,), device=device)
        j_g = torch.randint(0, n, (n_global,), device=device)
        same = i_g == j_g
        if same.any():
            j_g = j_g.clone()
            j_g[same] = (j_g[same] + 1) % n
    else:
        i_g = torch.empty(0, dtype=torch.long, device=device)
        j_g = torch.empty(0, dtype=torch.long, device=device)

    i = torch.cat([i_loc, i_g])
    j = torch.cat([j_loc, j_g])
    return i, j


def compute_s_ref(
    z: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> float:
    """
    s_ref = median_{(i,j)} ||z_i - z_j||_2 / sqrt(d).

    Detached scalar; raise if degenerate (< 1e-8).
    """
    if z.ndim != 2:
        raise RuntimeError(f"compute_s_ref expects [N, d], got shape {tuple(z.shape)}")
    d_lat = z.shape[1]
    dist = torch.norm(z[pair_i] - z[pair_j], dim=-1)
    scaled = dist / math.sqrt(float(d_lat))
    s_ref = float(scaled.median().item())
    if s_ref < S_REF_DEGENERATE:
        raise RuntimeError(
            f"degenerate_at_initialization: s_ref={s_ref} < {S_REF_DEGENERATE}"
        )
    return s_ref


def mico_pair_targets(
    rewards: torch.Tensor,
    z_next_old: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """t_ij = |r_i - r_j| + gamma * ||Z_tgt(s'_i) - Z_tgt(s'_j)|| (Euclidean)."""
    d_next = torch.norm(z_next_old[pair_i] - z_next_old[pair_j], dim=-1)
    return (rewards[pair_i] - rewards[pair_j]).abs() + gamma * d_next


def collapse_unjustified_fraction(
    d_bar_old: torch.Tensor,
    thresh: float,
    targets: torch.Tensor,
    target_thresh: float,
) -> torch.Tensor:
    """Fraction of all pairs with d_bar_old < thresh and behaviorally distinct."""
    floor_active = d_bar_old < thresh
    unjustified = floor_active & (targets > target_thresh)
    return unjustified.float().mean()


def compute_dz(
    z_new: torch.Tensor,
    z_old: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    s_ref: float,
    delta: float = DEFAULT_DZ_DELTA,
    rewards: torch.Tensor | None = None,
    z_next_old: torch.Tensor | None = None,
    gamma: float = 0.99,
    collapse_target_thresh: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    D_Z^log = mean( (log(d_bar_new + delta) - log(d_bar_old + delta))^2 ).

    d_bar = ||zi - zj|| / (sqrt(d) * s_ref). All sampled pairs enter the mean.
    s_ref must be the frozen initialization scale (not recomputed here).

    Stats values are detached GPU tensors (no .item() sync in the hot path).
    """
    if s_ref < S_REF_DEGENERATE:
        raise RuntimeError(f"s_ref must be >= {S_REF_DEGENERATE}, got {s_ref}")
    if delta <= 0:
        raise RuntimeError(f"delta must be > 0, got {delta}")
    if z_new.shape != z_old.shape:
        raise RuntimeError(
            f"z_new/z_old shape mismatch {tuple(z_new.shape)} vs {tuple(z_old.shape)}"
        )

    d_lat = z_new.shape[1]
    scale = math.sqrt(float(d_lat)) * s_ref
    d_old = torch.norm(z_old[pair_i] - z_old[pair_j], dim=-1)
    d_new = torch.norm(z_new[pair_i] - z_new[pair_j], dim=-1)
    d_bar_old = d_old / scale
    d_bar_new = d_new / scale

    log_delta = torch.log(d_bar_new + delta) - torch.log(d_bar_old + delta)
    dz = (log_delta ** 2).mean()

    with torch.no_grad():
        abs_log = log_delta.abs()
        c_001 = (d_bar_old < 0.01).float().mean()
        expand = (d_bar_new > d_bar_old).float().mean()
        contract = (d_bar_new < d_bar_old).float().mean()
        stats = {
            "D_Z": dz.detach(),
            "dz_s_ref": torch.tensor(s_ref, device=z_new.device, dtype=z_new.dtype),
            "dz_delta_eps": torch.tensor(delta, device=z_new.device, dtype=z_new.dtype),
            "dz_log_delta_p50": torch.quantile(abs_log, 0.50),
            "dz_log_delta_p90": torch.quantile(abs_log, 0.90),
            "dz_log_delta_p95": torch.quantile(abs_log, 0.95),
            "dz_log_delta_p99": torch.quantile(abs_log, 0.99),
            "dz_log_delta_mean": abs_log.mean(),
            "dz_signed_log_median": torch.quantile(log_delta, 0.50),
            "dz_d_bar_old_median": torch.quantile(d_bar_old, 0.50),
            "dz_d_bar_new_median": torch.quantile(d_bar_new, 0.50),
            "dz_C_0p01": c_001,
            "dz_C_0p001": (d_bar_old < 1e-3).float().mean(),
            "dz_C_0p1": (d_bar_old < 0.1).float().mean(),
            "dz_expand_frac": expand,
            "dz_contract_frac": contract,
            # Aliases for continuity with older logs / collapse naming.
            "dz_collapse_frac": c_001,
            "dz_floor_activation_rate": c_001,
        }
        if rewards is not None and z_next_old is not None:
            targets = mico_pair_targets(rewards, z_next_old, pair_i, pair_j, gamma)
            stats["dz_collapse_unjustified_frac"] = collapse_unjustified_fraction(
                d_bar_old, 0.01, targets, collapse_target_thresh
            )
            stats["dz_collapse_target_mean"] = targets.mean()
            near = d_bar_old < 0.01
            if near.any():
                stats["dz_collapse_target_floor_mean"] = targets[near].mean()
            else:
                stats["dz_collapse_target_floor_mean"] = torch.zeros(
                    (), device=z_new.device, dtype=z_new.dtype
                )
    return dz, stats


def adaptive_lambda_dz(
    lambda_dz: float,
    dz_value: float,
    eta: float,
    lo: float = 1e-4,
    hi: float = 1e4,
) -> float:
    """PPO-style adaptive multiplier for D_Z penalty."""
    eta_sq = eta * eta
    if dz_value > 1.5 * eta_sq:
        lambda_dz *= 2.0
    elif dz_value < eta_sq / 1.5:
        lambda_dz /= 2.0
    return float(max(lo, min(hi, lambda_dz)))


def adaptive_eta_dz(
    dz_ema_prev: float,
    eta_min: float,
    c: float = 1.5,
) -> float:
    """
    Radius from pre-update EMA of realized D_Z, floored at calibrated settled η.

    η_t = max(η_min, c * D̄_Z^(t-1)). Never tighter than η_min. η enters λ-adaptation
    as η² vs D_Z (same convention as fixed-η pilots).
    """
    if c <= 1.0:
        raise ValueError(f"eta adapt c must be > 1 (got {c})")
    if eta_min <= 0.0:
        raise ValueError(f"eta_min must be > 0 (got {eta_min})")
    if dz_ema_prev < 0.0:
        raise ValueError(f"D_Z EMA must be >= 0 (got {dz_ema_prev})")
    return float(max(eta_min, c * dz_ema_prev))


def update_dz_ema(dz_ema: float | None, dz_value: float, tau: float) -> float:
    """EMA of realized D_Z. tau small ⇒ slow timescale relative to PPO updates."""
    if tau <= 0.0 or tau > 1.0:
        raise ValueError(f"dz_ema_tau must be in (0, 1] (got {tau})")
    if dz_value < 0.0:
        raise ValueError(f"D_Z must be >= 0 (got {dz_value})")
    if dz_ema is None:
        return float(dz_value)
    return float((1.0 - tau) * dz_ema + tau * dz_value)


def displacement_penalty(z_new: torch.Tensor, z_old: torch.Tensor) -> torch.Tensor:
    """E||Z' - Z||^2  (absolute displacement; gauge-non-invariant foil)."""
    return (z_new - z_old).pow(2).sum(dim=-1).mean()


def mean_nearest_buffer_distance(z_on_policy: torch.Tensor, z_buffer: torch.Tensor) -> float:
    """Mean Euclidean distance from each on-policy latent to nearest ref-buffer latent."""
    d = torch.cdist(z_on_policy, z_buffer, p=2)
    return d.min(dim=1).values.mean().item()


def pairwise_distance_histogram(
    z: torch.Tensor,
    n_pairs: int = 8192,
    n_bins: int = 64,
) -> dict[str, object]:
    """Sample pairwise Euclidean distances; return histogram + quantiles (CPU numpy-ready)."""
    n = z.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 latents for pairwise histogram")
    device = z.device
    i = torch.randint(0, n, (n_pairs,), device=device)
    j = torch.randint(0, n, (n_pairs,), device=device)
    same = i == j
    if same.any():
        j = j.clone()
        j[same] = (j[same] + 1) % n
    d = torch.norm(z[i] - z[j], dim=-1)
    d_cpu = d.detach().float().cpu()
    hist = torch.histc(d_cpu, bins=n_bins, min=0.0, max=float(d_cpu.max().clamp_min(1e-12)))
    edges = torch.linspace(0.0, float(d_cpu.max().clamp_min(1e-12)), n_bins + 1)
    return {
        "counts": hist.numpy(),
        "bin_edges": edges.numpy(),
        "n_pairs": n_pairs,
        "p05": float(torch.quantile(d_cpu, 0.05)),
        "p50": float(torch.quantile(d_cpu, 0.50)),
        "p95": float(torch.quantile(d_cpu, 0.95)),
        "mean": float(d_cpu.mean()),
        "max": float(d_cpu.max()),
    }
