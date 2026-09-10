"""Control / query / behavior-mixture stream diagnostics."""

import torch

from src.metrics.action_coverage import action_coverage_stats


def stream_policy_stats(
    prefix: str,
    rewards: torch.Tensor,
    actions: torch.Tensor,
    n_actions: int,
    behavior_log_prob: torch.Tensor,
    control_log_prob: torch.Tensor,
    episode_returns: list[float],
    eta: float,
) -> dict[str, float]:
    cov = action_coverage_stats(actions, n_actions, behavior_log_prob, eta)
    ratio = torch.exp(control_log_prob - behavior_log_prob)
    ret = float(sum(episode_returns) / len(episode_returns)) if episode_returns else 0.0
    stats = {
        f"{prefix}_return": ret,
        f"{prefix}_n_steps": float(actions.shape[0]),
        f"{prefix}_reward_mean": float(rewards.mean().item()),
        f"{prefix}_nonzero_reward_frac": float((rewards != 0).float().mean().item()),
        f"{prefix}_behavior_logp_mean": float(behavior_log_prob.mean().item()),
        f"{prefix}_control_logp_mean": float(control_log_prob.mean().item()),
        f"{prefix}_ctrl_over_beh_ratio_mean": float(ratio.mean().item()),
        f"{prefix}_action_entropy": cov["coverage_action_entropy"],
        f"{prefix}_min_action_prob": cov["coverage_min_action_prob"],
    }
    for a in range(n_actions):
        stats[f"{prefix}_count_a{a}"] = cov[f"coverage_count_a{a}"]
    return stats


def sparse_reward_by_source(rewards: torch.Tensor, source: torch.Tensor) -> dict[str, float]:
    """source: 0=ppo, 1=query."""
    out = {
        "replay_n": float(rewards.shape[0]),
        "replay_nonzero_reward_frac": float((rewards != 0).float().mean().item()),
    }
    for code, name in ((0, "ppo"), (1, "query")):
        mask = source == code
        n = int(mask.sum().item())
        out[f"replay_{name}_n"] = float(n)
        if n == 0:
            out[f"replay_{name}_nonzero_reward_frac"] = float("nan")
            out[f"replay_{name}_frac"] = 0.0
        else:
            out[f"replay_{name}_nonzero_reward_frac"] = float((rewards[mask] != 0).float().mean().item())
            out[f"replay_{name}_frac"] = n / rewards.shape[0]
    return out
