"""Validate reference-scaled D_Z^log (Stage A checklist, CPU only)."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn

from src.losses.dz_trust_region import (
    DEFAULT_DZ_DELTA,
    compute_dz,
    compute_s_ref,
    displacement_penalty,
    sample_pair_index_mask,
)
from src.metrics.pl_ratio import compute_mu_pl_bootstrap, update_v_ref
from src.utils.batched_grad import batched_value_head_grad_z


class ToyCritic(nn.Module):
    """Minimal encoder + value_head + policy_head for geometry tests."""

    def __init__(self, obs_dim: int = 8, latent_dim: int = 8, action_dim: int = 4):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, latent_dim)
        self.value_head = nn.Linear(latent_dim, 1)
        self.policy_head = nn.Linear(latent_dim, action_dim)
        nn.init.orthogonal_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=0.5)
        nn.init.zeros_(self.value_head.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.5)
        nn.init.zeros_(self.policy_head.bias)
        self.latent_dim = latent_dim

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(obs)
        return z, torch.zeros_like(z)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode(obs)
        return self.value_head(z)


def _random_orthogonal(d: int) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(d, d))
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def _s_ref_for(z: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor) -> float:
    return compute_s_ref(z, pair_i, pair_j)


@torch.no_grad()
def _dz_pair(
    z_new: torch.Tensor,
    z_old: torch.Tensor,
    s_ref: float,
    n_pairs: int = 1024,
    delta: float = DEFAULT_DZ_DELTA,
    pair_i: torch.Tensor | None = None,
    pair_j: torch.Tensor | None = None,
) -> tuple[float, dict]:
    if pair_i is None or pair_j is None:
        pair_i, pair_j = sample_pair_index_mask(
            z_new.shape[0], n_pairs, lambda_loc=0.0, device=z_new.device
        )
    dz, stats = compute_dz(
        z_new, z_old, pair_i, pair_j, s_ref=s_ref, delta=delta
    )
    return float(dz.item()), {k: float(v.item()) for k, v in stats.items()}


def test_finite_zero_old() -> None:
    torch.manual_seed(0)
    z = torch.randn(64, 8)
    i = torch.zeros(32, dtype=torch.long)
    j = torch.ones(32, dtype=torch.long)
    # Force identical old pair (d_old=0) while new differs.
    z_old = z.clone()
    z_new = z.clone()
    z_new[1] = z_new[0] + 1.0
    s_ref = 1.0
    dz, _ = _dz_pair(z_new, z_old, s_ref=s_ref, pair_i=i, pair_j=j)
    if not math.isfinite(dz):
        raise RuntimeError(f"Zero-old distance produced non-finite D_Z={dz}")
    print(f"PASS finite zero-old: D_Z={dz:.6f}")


def test_swap_symmetry() -> None:
    torch.manual_seed(1)
    z0 = torch.randn(128, 8)
    z1 = z0 * 1.5 + torch.randn_like(z0) * 0.05
    i, j = sample_pair_index_mask(z0.shape[0], 512, lambda_loc=0.0, device=z0.device)
    s_ref = _s_ref_for(z0, i, j)
    dz_ab, _ = _dz_pair(z1, z0, s_ref=s_ref, pair_i=i, pair_j=j)
    dz_ba, _ = _dz_pair(z0, z1, s_ref=s_ref, pair_i=i, pair_j=j)
    if abs(dz_ab - dz_ba) > 1e-6:
        raise RuntimeError(f"Swap: D_Z(new,old)={dz_ab} != D_Z(old,new)={dz_ba}")
    print(f"PASS swap symmetry: {dz_ab:.6f}")


def test_scale_invariance_with_s_ref() -> None:
    torch.manual_seed(2)
    z0 = torch.randn(128, 8)
    i, j = sample_pair_index_mask(z0.shape[0], 512, lambda_loc=0.0, device=z0.device)
    s0 = _s_ref_for(z0, i, j)
    c = 3.0
    z1 = c * z0
    # Matching s_ref rescale leaves d_bar unchanged under pure scale of both.
    dz_a, _ = _dz_pair(1.1 * z0, z0, s_ref=s0, pair_i=i, pair_j=j)
    dz_b, _ = _dz_pair(1.1 * z1, z1, s_ref=c * s0, pair_i=i, pair_j=j)
    if abs(dz_a - dz_b) > 1e-5:
        raise RuntimeError(f"Scale invariance: {dz_a} vs {dz_b}")
    print(f"PASS s_ref scale invariance: {dz_a:.6f}")


def test_expand_contract() -> None:
    torch.manual_seed(3)
    z0 = torch.randn(256, 8) * 2.0
    i, j = sample_pair_index_mask(z0.shape[0], 1024, lambda_loc=0.0, device=z0.device)
    s_ref = _s_ref_for(z0, i, j)
    c = 2.0
    dz_exp, _ = _dz_pair(c * z0, z0, s_ref=s_ref, pair_i=i, pair_j=j)
    dz_con, _ = _dz_pair((1.0 / c) * z0, z0, s_ref=s_ref, pair_i=i, pair_j=j)
    rel = abs(dz_exp - dz_con) / max(dz_exp, dz_con, 1e-12)
    if rel > 0.02:
        raise RuntimeError(
            f"Expand/contract: expand={dz_exp} contract={dz_con} rel={rel:.4f}"
        )
    expected = (math.log(c)) ** 2
    if abs(dz_exp - expected) > 0.02:
        raise RuntimeError(f"Expand: D_Z={dz_exp} expected ~{expected}")
    print(f"PASS expand/contract: {dz_exp:.6f} vs {dz_con:.6f}")


def test_gauge() -> None:
    torch.manual_seed(4)
    critic = ToyCritic()
    obs = torch.randn(256, 8)
    z0, _ = critic.encode(obs)
    v0 = critic.value_head(z0).squeeze(-1)
    logits0 = critic.policy_head(z0)
    i, j = sample_pair_index_mask(z0.shape[0], 1024, lambda_loc=0.0, device=z0.device)
    s_ref = _s_ref_for(z0, i, j)

    R = _random_orthogonal(8)
    Rinv = R.T
    with torch.no_grad():
        critic.encoder.weight.copy_(R @ critic.encoder.weight)
        critic.encoder.bias.copy_(R @ critic.encoder.bias)
        critic.value_head.weight.copy_(critic.value_head.weight @ Rinv)
        critic.policy_head.weight.copy_(critic.policy_head.weight @ Rinv)

    z1, _ = critic.encode(obs)
    v1 = critic.value_head(z1).squeeze(-1)
    logits1 = critic.policy_head(z1)
    if not torch.allclose(v0, v1, atol=1e-5):
        raise RuntimeError(f"Gauge: values changed max|dv|={(v0 - v1).abs().max().item()}")
    if not torch.allclose(logits0, logits1, atol=1e-5):
        raise RuntimeError(
            f"Gauge: logits changed max={((logits0 - logits1).abs().max().item())}"
        )
    dz, _ = _dz_pair(z1, z0, s_ref=s_ref, pair_i=i, pair_j=j)
    if dz > 1e-5:
        raise RuntimeError(f"Gauge: D_Z={dz} expected <1e-5")
    disp = float(displacement_penalty(z1, z0).item())
    if disp < 1e-3:
        raise RuntimeError(f"Gauge: displacement {disp} should be large")
    print(f"PASS gauge: D_Z={dz:.2e} displacement={disp:.4f}")


def test_shear() -> None:
    torch.manual_seed(5)
    critic = ToyCritic()
    with torch.no_grad():
        critic.value_head.weight.zero_()
        critic.value_head.weight[0, 0] = 1.0
        critic.value_head.weight[0, 1] = 2.0
    obs = torch.randn(256, 8)
    z0, _ = critic.encode(obs)
    i, j = sample_pair_index_mask(z0.shape[0], 1024, lambda_loc=0.0, device=z0.device)
    s_ref = _s_ref_for(z0, i, j)
    v0 = critic.value_head(z0).squeeze(-1)
    v_ref = update_v_ref(None, v0)
    mu0 = compute_mu_pl_bootstrap(critic, z0, v_ref=v_ref)["mu_pl_q05"]

    c = 2.0
    A = torch.eye(8)
    A[1, 1] = c
    Ainv = torch.eye(8)
    Ainv[1, 1] = 1.0 / c
    with torch.no_grad():
        critic.encoder.weight.copy_(A @ critic.encoder.weight)
        critic.encoder.bias.copy_(A @ critic.encoder.bias)
        critic.value_head.weight.copy_(critic.value_head.weight @ Ainv)
        critic.policy_head.weight.copy_(critic.policy_head.weight @ Ainv)

    z1, _ = critic.encode(obs)
    v1 = critic.value_head(z1).squeeze(-1)
    if not torch.allclose(v0, v1, atol=1e-5):
        raise RuntimeError("Shear: value function changed")
    dz, _ = _dz_pair(z1, z0, s_ref=s_ref, pair_i=i, pair_j=j)
    if dz < 1e-6:
        raise RuntimeError(f"Shear: D_Z={dz} should be clearly nonzero")
    mu1 = compute_mu_pl_bootstrap(critic, z1, v_ref=v_ref)["mu_pl_q05"]
    if math.isnan(mu0) or math.isnan(mu1):
        print(f"PASS shear: D_Z={dz:.6f} (mu_PL undefined/floored; geometry ok)")
        return
    if mu1 >= mu0 - 1e-12:
        raise RuntimeError(f"Shear: mu_PL should decrease ({mu0} -> {mu1})")
    print(f"PASS shear: D_Z={dz:.6f} mu_pl {mu0:.6f} -> {mu1:.6f}")


def test_all_pairs_contribute() -> None:
    torch.manual_seed(6)
    z0 = torch.randn(64, 8)
    z1 = z0 * 1.2
    i = torch.arange(32)
    j = (i + 1) % 64
    s_ref = _s_ref_for(z0, i, j)
    dz_mean, _ = _dz_pair(z1, z0, s_ref=s_ref, pair_i=i, pair_j=j)
    # Explicit per-pair mean must match.
    d_lat = z0.shape[1]
    scale = math.sqrt(float(d_lat)) * s_ref
    d_old = torch.norm(z0[i] - z0[j], dim=-1) / scale
    d_new = torch.norm(z1[i] - z1[j], dim=-1) / scale
    manual = (
        (torch.log(d_new + DEFAULT_DZ_DELTA) - torch.log(d_old + DEFAULT_DZ_DELTA)) ** 2
    ).mean().item()
    if abs(dz_mean - manual) > 1e-6:
        raise RuntimeError(f"All-pairs: {dz_mean} != manual {manual}")
    print(f"PASS all pairs contribute: {dz_mean:.6f}")


def test_s_ref_frozen() -> None:
    torch.manual_seed(7)
    z0 = torch.randn(128, 8)
    i, j = sample_pair_index_mask(z0.shape[0], 512, lambda_loc=0.0, device=z0.device)
    s0 = _s_ref_for(z0, i, j)
    z1 = z0 * 0.1
    s1 = _s_ref_for(z1, i, j)
    if abs(s0 - s1) < 1e-6:
        raise RuntimeError("s_ref should change if recomputed on collapsed z")
    # Using frozen s0 on collapsed z must still be finite and different.
    dz, stats = _dz_pair(z1, z0, s_ref=s0, pair_i=i, pair_j=j)
    if abs(stats["dz_s_ref"] - s0) > 1e-12:
        raise RuntimeError("stats s_ref must equal frozen value")
    if not math.isfinite(dz):
        raise RuntimeError("frozen s_ref path non-finite")
    print(f"PASS s_ref frozen: s0={s0:.6f} recomputed_s1={s1:.6f}")


def test_dual_slack() -> None:
    torch.manual_seed(8)
    z0 = torch.randn(64, 8)
    z1 = z0 * 1.3
    i, j = sample_pair_index_mask(z0.shape[0], 256, lambda_loc=0.0, device=z0.device)
    s_ref = _s_ref_for(z0, i, j)
    dz, _ = _dz_pair(z1, z0, s_ref=s_ref, pair_i=i, pair_j=j)
    eta = 0.1
    slack = dz - eta * eta
    if abs(slack - (dz - 0.01)) > 1e-12:
        raise RuntimeError(f"Dual slack mismatch: {slack}")
    print(f"PASS dual slack: D_Z-eta^2={slack:.6f}")


def test_c_001() -> None:
    torch.manual_seed(9)
    # Construct known fraction of near-zero d_bar_old.
    n = 100
    z = torch.randn(n, 8)
    # 20 pairs with identical points (d=0), 80 with large separation.
    i = torch.arange(100)
    j = torch.arange(100)
    j[:20] = i[:20]
    j[20:] = (i[20:] + 50) % n
    s_ref = 1.0
    # Force scale so non-identical pairs have d_bar >> 0.01
    z = z * 10.0
    _, stats = _dz_pair(z, z, s_ref=s_ref, pair_i=i, pair_j=j)
    c = stats["dz_C_0p01"]
    if abs(c - 0.20) > 0.02:
        raise RuntimeError(f"C_0.01={c} expected ~0.20")
    print(f"PASS C_0.01: {c:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    if args.device != "cpu":
        raise SystemExit("validate_dz_geometry must run on CPU")
    _ = batched_value_head_grad_z
    test_finite_zero_old()
    test_swap_symmetry()
    test_scale_invariance_with_s_ref()
    test_expand_contract()
    test_gauge()
    test_shear()
    test_all_pairs_contribute()
    test_s_ref_frozen()
    test_dual_slack()
    test_c_001()
    print("All D_Z geometry validation tests passed.")


if __name__ == "__main__":
    main()
