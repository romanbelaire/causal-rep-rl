"""Phase 1 collapse diagnostics: functional value probe and dormant units."""

from __future__ import annotations

import torch


def dormant_unit_fraction(z: torch.Tensor, var_eps: float = 1e-6) -> float:
    """Fraction of latent dims with activation variance below var_eps on the batch."""
    if z.ndim != 2:
        raise RuntimeError(f"dormant_unit_fraction expects [N, d], got {tuple(z.shape)}")
    if z.shape[0] < 2:
        raise RuntimeError("dormant_unit_fraction needs at least 2 rows")
    var = z.detach().float().var(dim=0, unbiased=False)
    return float((var < var_eps).float().mean().item())


def functional_value_probe_mse(
    z: torch.Tensor,
    targets: torch.Tensor,
    ridge: float = 1e-3,
    train_frac: float = 0.5,
) -> float:
    """
    Freeze encoder features z; fit a linear probe to value targets with ridge LS.

    Uses the first train_frac of rows for the closed-form fit and reports MSE on
    the held-out remainder. Fixed budget = one ridge solve (no iterative search).
    """
    if z.ndim != 2:
        raise RuntimeError(f"functional_value_probe_mse expects [N, d], got {tuple(z.shape)}")
    if targets.ndim != 1:
        raise RuntimeError(
            f"functional_value_probe_mse targets must be [N], got {tuple(targets.shape)}"
        )
    if z.shape[0] != targets.shape[0]:
        raise RuntimeError(
            f"z/targets length mismatch {z.shape[0]} vs {targets.shape[0]}"
        )
    n = z.shape[0]
    n_train = int(n * train_frac)
    if n_train < 2 or n - n_train < 2:
        raise RuntimeError(
            f"need enough rows for train/val split (n={n}, train_frac={train_frac})"
        )

    z = z.detach().float()
    y = targets.detach().float()
    # Bias column.
    ones = torch.ones(n, 1, device=z.device, dtype=z.dtype)
    x = torch.cat([z, ones], dim=1)

    x_tr, y_tr = x[:n_train], y[:n_train]
    x_te, y_te = x[n_train:], y[n_train:]
    d = x_tr.shape[1]
    xtx = x_tr.T @ x_tr + ridge * torch.eye(d, device=z.device, dtype=z.dtype)
    xty = x_tr.T @ y_tr
    w = torch.linalg.solve(xtx, xty)
    pred = x_te @ w
    return float(((pred - y_te) ** 2).mean().item())


def near_zero_pair_rate(
    z: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    s_ref: float,
    thresh: float = 0.01,
) -> float:
    """C_thresh = Pr[ ||zi-zj|| / (sqrt(d) s_ref) < thresh ]."""
    if s_ref <= 0:
        raise RuntimeError(f"s_ref must be > 0, got {s_ref}")
    d_lat = z.shape[1]
    scale = (float(d_lat) ** 0.5) * s_ref
    dist = torch.norm(z[pair_i] - z[pair_j], dim=-1)
    d_bar = dist / scale
    return float((d_bar < thresh).float().mean().item())
