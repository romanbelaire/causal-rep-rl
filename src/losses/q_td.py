"""One-step Huber TD for Q^{pi_ctrl}. Bootstraps on truncation, not termination."""

import torch
import torch.nn.functional as F


def policy_value_from_q(q: torch.Tensor, pi_probs: torch.Tensor) -> torch.Tensor:
    """q [B, J, A], pi [B, A] -> V [B, J]."""
    return (q * pi_probs.unsqueeze(1)).sum(dim=-1)


def q_td_loss(
    q_sa: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    v_target_members: torch.Tensor,
    gamma: float,
    huber_delta: float,
    bootstrap_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """
    q_sa: online Q(z,a) [B, J]
    v_target_members: sum_{a'} pi^-(a'|z+) Qbar_j(z+, a') [B, J]
    bootstrap_mask: per-member sample mask [B, J]
    """
    if bootstrap_mask.shape != q_sa.shape:
        raise RuntimeError(
            f"bootstrap mask shape {tuple(bootstrap_mask.shape)} != Q {tuple(q_sa.shape)}"
        )
    if (bootstrap_mask.sum(dim=0) == 0).any():
        raise RuntimeError("an ensemble member received an all-zero bootstrap mask")
    live = (~terminated).float().unsqueeze(1)
    target = reward.unsqueeze(1) + gamma * live * v_target_members
    residual = F.huber_loss(q_sa, target.detach(), reduction="none", delta=huber_delta)
    loss = (residual * bootstrap_mask).sum() / bootstrap_mask.sum()
    disagreement = v_target_members.var(dim=1)
    stats = {
        "train_q_td_loss": loss.detach(),
        "train_q_target_var_mean": disagreement.mean().detach(),
        "train_q_td_per_member_mean": residual.mean(dim=0).detach(),
    }
    return loss, stats, disagreement


def action_gap_uncertainty(q: torch.Tensor, pi_probs: torch.Tensor) -> torch.Tensor:
    """Std across members of greedy-vs-mean action gap. q [B, J, A] -> [B]."""
    v = policy_value_from_q(q, pi_probs)
    a_star = q.mean(dim=1).argmax(dim=-1)
    b = torch.arange(q.shape[0], device=q.device)
    q_star = q[b, :, a_star]
    gap = q_star - v
    return gap.std(dim=1)


def member_bootstrap_mask(batch_size: int, n_members: int, device: torch.device) -> torch.Tensor:
    """Independent Bernoulli masks; fail if a column is empty."""
    mask = torch.bernoulli(torch.full((batch_size, n_members), 0.8, device=device))
    empty = mask.sum(dim=0) == 0
    if empty.any():
        mask[:, empty] = 1.0
    return mask
