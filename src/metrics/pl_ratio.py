"""mu_PL diagnostic: value-suboptimality PL ratio (not Bellman residual)."""

import torch
import torch.nn as nn

from src.utils.batched_grad import batched_value_head_grad_z


def _quantile_stats(x: torch.Tensor, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_q05": torch.quantile(x, 0.05).item(),
        f"{prefix}_q25": torch.quantile(x, 0.25).item(),
        f"{prefix}_q50": torch.quantile(x, 0.50).item(),
        f"{prefix}_median": x.median().item(),
        f"{prefix}_mean": x.mean().item(),
    }


def median_pairwise_distance(
    z: torch.Tensor,
    n_pairs: int = 4096,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Median Euclidean pairwise distance on a random subset of pairs (scalar tensor)."""
    n = z.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 latents for median pairwise distance")
    device = z.device
    if generator is None:
        i = torch.randint(0, n, (n_pairs,), device=device)
        j = torch.randint(0, n, (n_pairs,), device=device)
    else:
        i = torch.randint(0, n, (n_pairs,), generator=generator, device=device)
        j = torch.randint(0, n, (n_pairs,), generator=generator, device=device)
    same = i == j
    if same.any():
        j = j.clone()
        j[same] = (j[same] + 1) % n
    d = torch.norm(z[i] - z[j], dim=-1)
    return d.median()


def value_range_scale(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Characteristic value scale: max(V) - min(V), floored at eps."""
    return (v.max() - v.min()).clamp_min(eps)


def update_v_ref(
    v_ref: float | None,
    values: torch.Tensor,
    tau_ref: float = 0.01,
    quantile: float = 0.99,
) -> float:
    """EMA-smoothed q-quantile of batch values as reference level V̂_q."""
    v_ref_batch = torch.quantile(values.detach().float(), quantile).item()
    if v_ref is None:
        return v_ref_batch
    return (1.0 - tau_ref) * v_ref + tau_ref * v_ref_batch


def value_gap_histogram(
    f_raw: torch.Tensor,
    n_bins: int = 64,
) -> dict[str, object]:
    """
    Histogram of raw value gaps f = V̂_q - V (no clamp).

    Persisted for post-hoc τ sensitivity on μ_PL without re-running.
    Negative gaps (V above the reference) are included; bin edges span min..max.
    """
    f_cpu = f_raw.detach().float().reshape(-1).cpu()
    if f_cpu.numel() == 0:
        raise RuntimeError("value_gap_histogram: empty f_raw")
    f_min = float(f_cpu.min())
    f_max = float(f_cpu.max())
    if abs(f_max - f_min) < 1e-12:
        f_max = f_min + 1e-12
    hist = torch.histc(f_cpu, bins=n_bins, min=f_min, max=f_max)
    edges = torch.linspace(f_min, f_max, n_bins + 1)
    return {
        "counts": hist.numpy(),
        "bin_edges": edges.numpy(),
        "n": int(f_cpu.numel()),
        "p05": float(torch.quantile(f_cpu, 0.05)),
        "p50": float(torch.quantile(f_cpu, 0.50)),
        "p95": float(torch.quantile(f_cpu, 0.95)),
        "mean": float(f_cpu.mean()),
        "min": f_min,
        "max": f_max,
        "frac_positive": float((f_cpu > 0).float().mean()),
    }


def effective_f_floor(
    v_ref: float,
    f_floor_rel: float = 1e-3,
    f_floor_abs_min: float = 1e-12,
) -> float:
    """
    Scale-relative numerical floor for the value gap.

    f_floor_eff = max(abs_min, f_floor_rel * |v_ref|).

    Absolute 1e-3 (legacy) floors the entire batch whenever the value scale is
    ≲ 1e-3 or V is peaked within 1e-3 of V̂_q — which is the formation-phase
    failure of P0.1. Scaling with |v_ref| keeps the same relative threshold
    (0.1% of the reference level) at every scale.
    """
    if f_floor_rel <= 0.0:
        raise ValueError(f"f_floor_rel must be > 0 (got {f_floor_rel})")
    if f_floor_abs_min <= 0.0:
        raise ValueError(f"f_floor_abs_min must be > 0 (got {f_floor_abs_min})")
    return float(max(f_floor_abs_min, f_floor_rel * abs(v_ref)))


def _nan_mu_pl_stats(prefix: str) -> dict[str, float]:
    nan = float("nan")
    return {
        f"{prefix}_q05": nan,
        f"{prefix}_q05_conditional": nan,
        f"{prefix}_q25": nan,
        f"{prefix}_q50": nan,
        f"{prefix}_mean": nan,
        f"{prefix}_median": nan,
        f"{prefix}_inf": nan,
    }


def compute_mu_pl_bootstrap(
    critic: nn.Module,
    z: torch.Tensor,
    v_ref: float,
    f_floor: float = 1e-3,
    max_samples: int | None = None,
    f_floor_abs_min: float = 1e-12,
    pair_n: int = 4096,
) -> dict[str, float]:
    """
    mu_PL from value-suboptimality gap, not one-step Bellman residual.

    f_raw = v_ref - V(Z(s))
    f_floor_eff = max(f_floor_abs_min, f_floor * |v_ref|)   # f_floor is relative
    Floored samples (f_raw <= f_floor_eff) are excluded from mu_PL / grad_sq / f
    percentiles. f_floor_rate is the mask rate.

    Also logs scale-invariant and value-normalized variants (T1.1 / T1.2):
      L = median pairwise ||Z_i - Z_j|| on the batch
      V_scale = max(V) - min(V) on the batch
      mu_pl_tilde = mu_PL * L^2
      mu_pl_tilde_v = mu_PL * L^2 / V_scale

    If every sample is floored, percentiles are NaN and f_floor_rate=1 — do not
    abort training (diagnostic only).
    """
    n = z.shape[0]
    if max_samples is not None and max_samples < n:
        idx = torch.randperm(n, device=z.device)[:max_samples]
        z = z[idx]
        n = z.shape[0]

    z_grad = z.detach().requires_grad_(True)
    grad_z = batched_value_head_grad_z(critic, z_grad)
    grad_sq = grad_z.pow(2).sum(dim=1)

    f_floor_eff = effective_f_floor(v_ref, f_floor_rel=f_floor, f_floor_abs_min=f_floor_abs_min)

    with torch.no_grad():
        v = critic.value_head(z.detach()).squeeze(-1)
        f_raw = v_ref - v
        at_floor = f_raw <= f_floor_eff
        floor_rate = at_floor.float().mean().item()
        valid = ~at_floor
        n_valid = int(valid.sum().item())
        n_total = float(f_raw.numel())
        i = torch.randint(0, n, (pair_n,), device=z.device)
        j = torch.randint(0, n, (pair_n,), device=z.device)
        same = i == j
        if same.any():
            j = j.clone()
            j[same] = (j[same] + 1) % n
        d_sample = torch.norm(z.detach()[i] - z.detach()[j], dim=-1)
        pair_p05 = torch.quantile(d_sample, 0.05)
        pair_p50 = d_sample.median()
        L = pair_p50
        V_scale = value_range_scale(v)

        base = {
            "v_ref": float(v_ref),
            "f_floor_rate": floor_rate,
            "f_floor_eff": f_floor_eff,
            "mu_pl_valid_fraction": float(n_valid) / n_total,
            "mu_pl_n_valid": float(n_valid),
            "mu_pl_n_total": n_total,
            "latent_pair_L": float(L.item()),
            "latent_pair_p05": float(pair_p05.item()),
            "latent_pair_p50": float(pair_p50.item()),
            "value_scale": float(V_scale.item()),
        }
        if n_valid == 0:
            out = {
                **base,
                **_nan_mu_pl_stats("mu_pl"),
                **_nan_mu_pl_stats("mu_pl_tilde"),
                **_nan_mu_pl_stats("mu_pl_tilde_v"),
                "grad_sq_q05": float("nan"),
                "grad_sq_q25": float("nan"),
                "grad_sq_q50": float("nan"),
                "grad_sq_mean": float("nan"),
                "grad_sq_median": float("nan"),
                "f_q05": float("nan"),
                "f_q25": float("nan"),
                "f_q50": float("nan"),
                "f_mean": float("nan"),
                "f_median": float("nan"),
            }
            return out

        f_valid = f_raw[valid]
        grad_sq_valid = grad_sq[valid]
        pl_ratio = grad_sq_valid / (2.0 * f_valid)
        pl_tilde = pl_ratio * (L ** 2)
        pl_tilde_v = pl_tilde / V_scale

        if (pl_ratio < 0).any():
            raise RuntimeError("mu_PL must be non-negative; got negative pl_ratio")

        out = {
            **base,
            "mu_pl_q05": torch.quantile(pl_ratio, 0.05).item(),
            "mu_pl_q05_conditional": torch.quantile(pl_ratio, 0.05).item(),
            "mu_pl_mean": pl_ratio.mean().item(),
            "mu_pl_median": pl_ratio.median().item(),
            "mu_pl_inf": pl_ratio.min().item(),
            "mu_pl_tilde_q05": torch.quantile(pl_tilde, 0.05).item(),
            "mu_pl_tilde_q05_conditional": torch.quantile(pl_tilde, 0.05).item(),
            "mu_pl_tilde_mean": pl_tilde.mean().item(),
            "mu_pl_tilde_median": pl_tilde.median().item(),
            "mu_pl_tilde_inf": pl_tilde.min().item(),
            "mu_pl_tilde_v_q05": torch.quantile(pl_tilde_v, 0.05).item(),
            "mu_pl_tilde_v_q05_conditional": torch.quantile(pl_tilde_v, 0.05).item(),
            "mu_pl_tilde_v_mean": pl_tilde_v.mean().item(),
            "mu_pl_tilde_v_median": pl_tilde_v.median().item(),
            "mu_pl_tilde_v_inf": pl_tilde_v.min().item(),
        }
        out.update(_quantile_stats(grad_sq_valid, "grad_sq"))
        out.update(_quantile_stats(f_valid, "f"))
        out.update(_quantile_stats(pl_ratio, "mu_pl"))
        out.update(_quantile_stats(pl_tilde, "mu_pl_tilde"))
        out.update(_quantile_stats(pl_tilde_v, "mu_pl_tilde_v"))
    return out


def target_landscape_pl_stats(
    critic_target: nn.Module,
    z_target: torch.Tensor,
    v_ref_target: float,
    f_floor: float = 1e-3,
) -> dict[str, float]:
    """PL on a frozen encoder+critic. Prefix target_ so it cannot overwrite online PL."""
    raw = compute_mu_pl_bootstrap(
        critic_target, z_target, v_ref_target, f_floor=f_floor
    )
    return {f"target_{k}": v for k, v in raw.items()}
