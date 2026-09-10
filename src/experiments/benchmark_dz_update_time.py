#!/usr/bin/env python3
"""P0.4 CPU microbenchmark: D_Z overhead relative to a CTRO-like encoder+value step.

Times three modes on a synthetic minibatch (CPU only, deterministic exit):
1. CTRO-like: encode + value MSE + MICo-free auxiliary (value-head PL-style hinge stub)
2. Fixed-λ D_Z added
3. Full dual (D_Z + adaptive λ update)

Avoids full MICo/PL pair loops so the measurement isolates D_Z cost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.losses.dz_trust_region import (
    DEFAULT_DZ_DELTA,
    adaptive_lambda_dz,
    compute_dz,
    compute_s_ref,
    sample_pair_index_mask,
)


class _Toy(nn.Module):
    def __init__(self, obs_dim: int = 64, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, latent_dim)
        self.value_head = nn.Linear(latent_dim, 1)

    def encode(self, obs: torch.Tensor):
        z = self.encoder(obs)
        return z, torch.zeros_like(z)


def _time_iters(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--n-pairs", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/dmcontrol_pixels/phase0_dz_update_timing.json"),
    )
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")
    B = args.batch_size
    D = args.latent_dim

    net = _Toy(obs_dim=64, latent_dim=D).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    obs = torch.randn(B, 64, device=device)
    targets = torch.randn(B, device=device)
    rewards = torch.randn(B, device=device)
    z_old = torch.randn(B, D, device=device)
    z_next_old = torch.randn(B, D, device=device)
    pair_i, pair_j = sample_pair_index_mask(B, args.n_pairs, 0.7, device)
    s_ref = compute_s_ref(z_old, pair_i, pair_j)
    lambda_dz = 1.0
    eta = 0.05

    def base_loss(z: torch.Tensor) -> torch.Tensor:
        v = net.value_head(z).squeeze(-1)
        # Stand-in for CTRO task + aux load on the encoder/value path.
        return ((v - targets) ** 2).mean() + 0.1 * z.pow(2).mean()

    def ctro_no_dz():
        opt.zero_grad(set_to_none=True)
        z, _ = net.encode(obs)
        base_loss(z).backward()
        opt.step()

    def dz_fixed_lambda():
        opt.zero_grad(set_to_none=True)
        z, _ = net.encode(obs)
        dz, _ = compute_dz(
            z,
            z_old,
            pair_i,
            pair_j,
            s_ref=s_ref,
            delta=DEFAULT_DZ_DELTA,
            rewards=rewards,
            z_next_old=z_next_old,
            gamma=0.99,
        )
        loss = base_loss(z) + 1.0 * (dz - eta * eta)
        loss.backward()
        opt.step()

    def dz_full_dual():
        nonlocal lambda_dz
        opt.zero_grad(set_to_none=True)
        z, _ = net.encode(obs)
        dz, _ = compute_dz(
            z,
            z_old,
            pair_i,
            pair_j,
            s_ref=s_ref,
            delta=DEFAULT_DZ_DELTA,
            rewards=rewards,
            z_next_old=z_next_old,
            gamma=0.99,
        )
        loss = base_loss(z) + lambda_dz * (dz - eta * eta)
        loss.backward()
        opt.step()
        with torch.no_grad():
            lambda_dz = adaptive_lambda_dz(lambda_dz, float(dz.detach()), eta)

    t_ctro = _time_iters(ctro_no_dz, args.warmup, args.iters)
    t_fixed = _time_iters(dz_fixed_lambda, args.warmup, args.iters)
    t_dual = _time_iters(dz_full_dual, args.warmup, args.iters)

    payload = {
        "device": "cpu",
        "batch_size": B,
        "latent_dim": D,
        "n_pairs": args.n_pairs,
        "warmup": args.warmup,
        "iters": args.iters,
        "note": (
            "Synthetic encoder+value step; isolates D_Z forward/backward cost "
            "relative to a CTRO-like base loss (not full MICo/PL)."
        ),
        "seconds_per_update": {
            "ctro_no_dz": t_ctro,
            "dz_fixed_lambda": t_fixed,
            "dz_full_dual": t_dual,
        },
        "overhead_vs_ctro": {
            "dz_fixed_lambda": (t_fixed - t_ctro) / t_ctro,
            "dz_full_dual": (t_dual - t_ctro) / t_ctro,
        },
        "target_max_overhead": 0.30,
        "dual_overhead_ok": ((t_dual - t_ctro) / t_ctro) < 0.30,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
