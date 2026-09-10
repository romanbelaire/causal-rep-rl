"""Stop-gradient information reward and U TD."""

import torch
import torch.nn.functional as F


def information_reward(
    i_q: torch.Tensor,
    i_gap: torch.Tensor,
    i_pair: torch.Tensor,
    lambda_q: float,
    lambda_gap: float,
    lambda_pair: float,
) -> torch.Tensor:
    return (lambda_q * i_q + lambda_gap * i_gap + lambda_pair * i_pair).detach()


def u_td_loss(
    u_sa: torch.Tensor,
    info_reward: torch.Tensor,
    terminated: torch.Tensor,
    u_target_max: torch.Tensor,
    gamma_u: float,
) -> tuple[torch.Tensor, dict]:
    live = (~terminated).float()
    target = info_reward + gamma_u * live * u_target_max
    loss = F.huber_loss(u_sa, target.detach(), reduction="mean")
    stats = {
        "train_u_td_loss": loss.detach(),
        "train_u_info_reward_mean": info_reward.mean().detach(),
    }
    return loss, stats
