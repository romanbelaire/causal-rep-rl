"""Separation losses.

`separation_loss` is the on-policy GAE hinge (PPO minibatches).
`trusted_negative_separation_loss` is the Active-CTRO UCB/LCB term.
"""

import torch

SEP_SR_EPS = 1e-8
ALPHA_SEP_CANDIDATES = (0.25, 0.5, 1.0)


def pair_derangement(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """B pairs from a random Hamiltonian cycle. No self-pairs. O(B)."""
    if batch_size < 2:
        raise RuntimeError(f"need batch_size >= 2 for pairs, got {batch_size}")
    order = torch.randperm(batch_size, device=device)
    return order, torch.roll(order, 1)


def trusted_negative_separation_loss(
    z: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    d_hat_lower: torch.Tensor,
    w_neg: torch.Tensor,
    alpha_sep: float,
) -> tuple[torch.Tensor, dict]:
    """L_sep = mean_{negatives} [alpha_sep * D_hat_lower - ||z_i - z_j||]_+^2.

    Undecided and positive pairs (w_neg == 0) do not enter the mean.
    """
    if alpha_sep <= 0:
        raise RuntimeError(f"alpha_sep must be > 0, got {alpha_sep}")
    live = (z[pair_i] - z[pair_j]).pow(2).sum(dim=1).sqrt()
    gap = (alpha_sep * d_hat_lower.detach() - live).clamp_min(0.0)
    sq = gap.pow(2)
    mass = w_neg.sum()
    n_neg = int((w_neg > 0).sum().item())
    if n_neg == 0:
        loss = sq.sum() * 0.0
        stats = {
            "train_sep_loss": loss.detach(),
            "train_sep_n_pairs": torch.tensor(0.0, device=z.device),
            "train_sep_margin_mean": torch.tensor(float("nan"), device=z.device),
        }
        return loss, stats
    w = w_neg / mass
    loss = (sq * w).sum()
    stats = {
        "train_sep_loss": loss.detach(),
        "train_sep_n_pairs": torch.tensor(float(n_neg), device=z.device),
        "train_sep_margin_mean": (live[w_neg > 0] - alpha_sep * d_hat_lower[w_neg > 0]).mean().detach(),
    }
    return loss, stats


def separation_loss(
    z: torch.Tensor,
    returns: torch.Tensor,
    alpha_sep: float,
    shuffle_returns: bool = False,
    pair_mode: str = "perm",
    s_r_eps: float = SEP_SR_EPS,
) -> tuple[torch.Tensor, dict]:
    """Median-normalized squared hinge on |ΔR̂| vs latent distance.

    L = mean [α · (|ΔR̂| / s_R) − (||Δz|| / s_Z)]_+²
    s_Z and s_R are detached batch medians over the paired sample.
    Skip (zero, no graph) when s_R < eps — sparse-reward batches, not a crash.
    """
    if pair_mode not in ("perm", "top_decile"):
        raise RuntimeError(f"sep_pair_mode must be perm or top_decile, got {pair_mode!r}")
    if alpha_sep <= 0:
        raise RuntimeError(f"alpha_sep must be > 0, got {alpha_sep}")
    if z.ndim != 2:
        raise RuntimeError(f"z must be [B, d], got {tuple(z.shape)}")
    if returns.ndim != 1 or returns.shape[0] != z.shape[0]:
        raise RuntimeError(
            f"returns must be [B] matching z, got {tuple(returns.shape)} vs {tuple(z.shape)}"
        )
    b = z.shape[0]
    pair_i, pair_j = pair_derangement(b, z.device)
    r = returns
    if shuffle_returns:
        r = r[torch.randperm(b, device=z.device)]
    dist = (z[pair_i] - z[pair_j]).pow(2).sum(dim=1).sqrt()
    d_r = (r[pair_i] - r[pair_j]).abs()
    if pair_mode == "top_decile":
        k = max(1, b // 10)
        _, keep = torch.topk(d_r, k)
        dist = dist[keep]
        d_r = d_r[keep]
    s_z = dist.median().detach()
    s_r = d_r.median().detach()
    n_pairs = dist.shape[0]
    empty = {
        "train_sep_loss": torch.tensor(0.0, device=z.device),
        "train_sep_n_pairs": torch.tensor(float(n_pairs), device=z.device),
        "train_sep_skipped": torch.tensor(1.0, device=z.device),
        "train_sep_s_z": s_z,
        "train_sep_s_r": s_r,
        "train_sep_hinge_active_frac": torch.tensor(float("nan"), device=z.device),
        "train_sep_margin_mean": torch.tensor(float("nan"), device=z.device),
    }
    if float(s_z.item()) <= 0.0:
        raise RuntimeError(f"s_Z must be > 0, got {float(s_z.item())}")
    if float(s_r.item()) < s_r_eps:
        return torch.zeros((), device=z.device), empty
    d_z_n = dist / s_z
    d_r_n = d_r / s_r
    gap = (alpha_sep * d_r_n - d_z_n).clamp_min(0.0)
    loss = gap.pow(2).mean()
    active = (alpha_sep * d_r_n - d_z_n) > 0
    stats = {
        "train_sep_loss": loss.detach(),
        "train_sep_n_pairs": torch.tensor(float(n_pairs), device=z.device),
        "train_sep_skipped": torch.tensor(0.0, device=z.device),
        "train_sep_s_z": s_z,
        "train_sep_s_r": s_r,
        "train_sep_hinge_active_frac": active.float().mean().detach(),
        "train_sep_margin_mean": (d_z_n - alpha_sep * d_r_n).mean().detach(),
    }
    return loss, stats
