"""Actor random-cumulant probe (Moalla-style functional capacity).

Freeze actor encoder features; fit linear probes to fixed random linear targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def make_random_cumulant_targets(
    n_obs: int,
    n_targets: int,
    feature_dim: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Fixed random linear maps of a reference feature basis → [N, K] targets.

    Targets are drawn once and stored in diag.pt; they do not depend on the live encoder.
    We store the weight matrix separately; this helper builds targets from random features
    when seeding a diagnostic set.
    """
    # Targets ~ N(0,1); independent of live phi (Moalla random cumulants).
    return torch.randn(n_obs, n_targets, generator=generator)


def fit_linear_probe_mse(
    z: torch.Tensor,
    y: torch.Tensor,
    ridge: float = 1e-3,
    train_frac: float = 0.5,
) -> tuple[float, float, float, float]:
    """Held-out MSE, constant MSE, NMSE, R^2 for one target column."""
    if z.ndim != 2 or y.ndim != 1:
        raise RuntimeError(f"bad shapes z={tuple(z.shape)} y={tuple(y.shape)}")
    n = z.shape[0]
    n_train = int(n * train_frac)
    if n_train < 2 or n - n_train < 2:
        raise RuntimeError(f"insufficient rows for split n={n}")

    z = z.detach().float()
    y = y.detach().float()
    ones = torch.ones(n, 1, device=z.device, dtype=z.dtype)
    x = torch.cat([z, ones], dim=1)
    x_tr, y_tr = x[:n_train], y[:n_train]
    x_te, y_te = x[n_train:], y[n_train:]
    d = x_tr.shape[1]
    xtx = x_tr.T @ x_tr + ridge * torch.eye(d, device=z.device, dtype=z.dtype)
    w = torch.linalg.solve(xtx, x_tr.T @ y_tr)
    pred = x_te @ w
    mse = float(((pred - y_te) ** 2).mean().item())
    y_mean = y_tr.mean()
    mse_const = float(((y_te - y_mean) ** 2).mean().item())
    if mse_const < 1e-12:
        nmse = 0.0 if mse < 1e-12 else float("inf")
    else:
        nmse = mse / mse_const
    r2 = 1.0 - nmse if nmse != float("inf") else float("-inf")
    return mse, mse_const, nmse, r2


def random_cumulant_probe(
    phi: torch.Tensor,
    targets: torch.Tensor,
    ridge: float = 1e-3,
    train_frac: float = 0.5,
) -> dict[str, float]:
    """
    Average probe metrics over K random cumulant columns.

    `phi` must be detached (encoder frozen). Asserts no requires_grad leafs in phi.
    """
    if phi.requires_grad:
        raise RuntimeError("encoder features must be detached before cumulant probe")
    if phi.ndim != 2 or targets.ndim != 2:
        raise RuntimeError(f"phi/targets shapes {tuple(phi.shape)} {tuple(targets.shape)}")
    if phi.shape[0] != targets.shape[0]:
        raise RuntimeError("phi/targets row mismatch")

    mses, nmse_s, r2s = [], [], []
    for k in range(targets.shape[1]):
        mse, _mse_c, nmse, r2 = fit_linear_probe_mse(
            phi, targets[:, k], ridge=ridge, train_frac=train_frac
        )
        mses.append(mse)
        nmse_s.append(nmse)
        r2s.append(r2)
    return {
        "cumulant_mse": float(sum(mses) / len(mses)),
        "cumulant_nmse": float(sum(nmse_s) / len(nmse_s)),
        "cumulant_r2": float(sum(r2s) / len(r2s)),
        "cumulant_n_targets": float(targets.shape[1]),
    }


def assert_no_encoder_grads(params: list[nn.Parameter]) -> None:
    for p in params:
        if p.grad is not None and float(p.grad.abs().sum().item()) != 0.0:
            raise RuntimeError("encoder received gradients during cumulant probe fit")
