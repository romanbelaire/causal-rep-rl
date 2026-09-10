"""Trusted negative-pair separation loss L_sep. Not a global co-Lipschitz claim."""

import torch


def separation_loss(
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
