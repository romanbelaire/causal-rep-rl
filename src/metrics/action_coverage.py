"""Action coverage diagnostics for discrete query behavior."""

import torch


def action_coverage_stats(
    actions: torch.Tensor,
    n_actions: int,
    behavior_log_prob: torch.Tensor,
    eta: float,
) -> dict[str, float]:
    counts = torch.bincount(actions.long(), minlength=n_actions).float()
    probs = counts / counts.sum()
    entropy = -(probs * (probs + 1e-12).log()).sum()
    floor = eta / n_actions
    mix_probs = torch.exp(behavior_log_prob)
    stats = {
        "coverage_action_entropy": float(entropy.item()),
        "coverage_uniform_floor": float(floor),
        "coverage_min_action_prob": float(probs.min().item()),
        "coverage_mean_beta": float(mix_probs.mean().item()),
        "coverage_min_beta": float(mix_probs.min().item()),
    }
    for a in range(n_actions):
        stats[f"coverage_count_a{a}"] = float(counts[a].item())
    return stats
