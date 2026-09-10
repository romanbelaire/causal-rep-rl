"""E3: query uniform floor vs pure uniform (CPU sample-efficiency proxy)."""

import torch

from src.agents.behavior_mixture import sample_behavior_actions
from src.environments.toys import CoverageToy


def _discriminating_hits(n_steps: int, eta: float, eps: float, seed: int) -> int:
    torch.manual_seed(seed)
    env = CoverageToy(seed=seed)
    hits = 0
    for _ in range(n_steps):
        obs, _ = env.reset()
        logits = torch.zeros(env.action_dim)
        query = torch.zeros(env.action_dim)
        action, _, _ = sample_behavior_actions(
            logits.unsqueeze(0), query.unsqueeze(0), eps, eta
        )
        _, r, _, _, info = env.step(int(action.item()))
        if info["alias"] == 1 and int(action.item()) == 1 and r > 5:
            hits += 1
    return hits


def _uniform_hits(n_steps: int, seed: int) -> int:
    torch.manual_seed(seed)
    env = CoverageToy(seed=seed)
    hits = 0
    for _ in range(n_steps):
        env.reset()
        action = int(torch.randint(0, env.action_dim, (1,)).item())
        _, r, _, _, info = env.step(action)
        if info["alias"] == 1 and action == 1 and r > 5:
            hits += 1
    return hits


def main():
    steps = 5000
    floor_hits = _discriminating_hits(steps, eta=0.05, eps=0.15, seed=0)
    uniform_hits = _uniform_hits(steps, seed=0)
    print(
        f"E3 query_coverage floor_hits={floor_hits} uniform_hits={uniform_hits} steps={steps}"
    )
    if floor_hits < uniform_hits:
        raise RuntimeError("E3: coverage floor did not improve discriminating-action hits")


if __name__ == "__main__":
    main()
