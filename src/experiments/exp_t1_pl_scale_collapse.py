"""T1: PL scale-collapse theorem. CPU, deterministic."""

import torch
import torch.nn as nn

from src.losses.separation import separation_loss
from src.metrics.feature_rank import compute_feature_rank_metrics
from src.metrics.pl_ratio import compute_mu_pl_bootstrap, target_landscape_pl_stats


class _ScaleCritic(nn.Module):
    """V(z) = ||z||^2 / (2 c^2) so V(c s) = ||s||^2 / 2."""

    def __init__(self, c: float, dim: int):
        super().__init__()
        self.c = float(c)
        self.encoder = nn.Identity()
        self.fc_mu = nn.Identity()
        self.value_head = _QuadraticHead(c)

    def encode(self, s: torch.Tensor):
        z = self.c * s
        return z, torch.zeros_like(z)


class _QuadraticHead(nn.Module):
    def __init__(self, c: float):
        super().__init__()
        self.register_buffer("inv_c2", torch.tensor(1.0 / (c * c)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return 0.5 * z.pow(2).sum(dim=-1, keepdim=True) * self.inv_c2


def _pl_at_c(s: torch.Tensor, c: float) -> dict[str, float]:
    critic = _ScaleCritic(c, s.shape[1])
    z, _ = critic.encode(s)
    with torch.no_grad():
        v = critic.value_head(z).squeeze(-1)
        v_ref = float(v.max().item())
    return compute_mu_pl_bootstrap(critic, z, v_ref, f_floor=1e-3)


def main() -> None:
    torch.manual_seed(0)
    s = torch.linspace(0.2, 1.0, 16).unsqueeze(1).repeat(1, 2)
    c_vals = [1.0, 0.5, 0.25]
    pls = []
    dists = []
    for c in c_vals:
        stats = _pl_at_c(s, c)
        z = c * s
        d = torch.cdist(z, z, p=2)
        off = d[~torch.eye(z.shape[0], dtype=torch.bool)]
        pls.append(stats["mu_pl_mean"])
        dists.append(float(off.median().item()))
        rank = compute_feature_rank_metrics(z)
        print(
            f"T1 c={c} mu_pl_mean={stats['mu_pl_mean']:.4f} "
            f"d_med={dists[-1]:.4f} pr={rank['feature_rank_participation_ratio']:.4f}"
        )
    ratio_pl = pls[-1] / pls[0]
    expected = (c_vals[0] / c_vals[-1]) ** 2
    if abs(ratio_pl / expected - 1.0) > 0.25:
        raise RuntimeError(
            f"T1: pullback PL did not scale as 1/c^2: ratio={ratio_pl:.3f} expected={expected:.3f}"
        )
    dist_ratio = dists[-1] / dists[0]
    if abs(dist_ratio / c_vals[-1] - 1.0) > 0.15:
        raise RuntimeError(
            f"T1: latent distances did not shrink linearly in c: {dist_ratio:.3f} vs {c_vals[-1]}"
        )
    critic_anchor = _ScaleCritic(1.0, s.shape[1])
    z_anchor, _ = critic_anchor.encode(s)
    with torch.no_grad():
        v_anchor = critic_anchor.value_head(z_anchor).squeeze(-1)
        v_ref_t = float(v_anchor.max().item())
    anchored = target_landscape_pl_stats(critic_anchor, z_anchor, v_ref_t)
    collapsed = _pl_at_c(s, 0.25)
    if abs(anchored["target_mu_pl_mean"] / pls[0] - 1.0) > 0.25:
        raise RuntimeError("T1: anchored target PL moved under a frozen encoder")
    if collapsed["mu_pl_mean"] <= anchored["target_mu_pl_mean"]:
        raise RuntimeError("T1: online collapsed PL did not exceed anchored PL")
    z_c = 0.25 * s
    pair_i = torch.arange(0, 8)
    pair_j = torch.arange(8, 16)
    d_true = (s[pair_i] - s[pair_j]).norm(dim=1)
    w_neg = torch.ones_like(d_true)
    sep, sep_stats = separation_loss(z_c, pair_i, pair_j, d_true, w_neg, alpha_sep=1.0)
    if float(sep_stats["train_sep_n_pairs"]) <= 0 or float(sep.item()) <= 0:
        raise RuntimeError("T1: L_sep failed to detect the collapsed negative pairs")
    print("T1 pl_scale_collapse ok")


if __name__ == "__main__":
    main()
