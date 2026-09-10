"""E4: nuisance merge — same-action pairs with identical rewards (CPU)."""

import torch

from src.environments.toys import NuisanceToy
from src.losses.action_conditioned_mico import action_conditioned_mico_loss
from src.replay.pair_index import pair_confidence_weights


def main():
    torch.manual_seed(2)
    env = NuisanceToy(seed=2)
    obs_rows = []
    for variant in (0, 1):
        env._variant = variant
        o, _ = env.reset()
        obs_rows.append(o)
    obs_t = torch.stack(obs_rows)
    actions_t = torch.tensor([0, 0])
    rewards_t = torch.tensor([1.0, 1.0])
    term_t = torch.tensor([True, True])
    z = obs_t
    pair_i = torch.tensor([0])
    pair_j = torch.tensor([1])
    diversity = (z[0] - z[1]).norm().unsqueeze(0)
    dhat = torch.tensor([0.0])
    sigma = torch.tensor([0.0])
    w_pos, w_neg, und = pair_confidence_weights(
        dhat,
        sigma,
        tau_pos=0.5,
        tau_neg=0.5,
        confidence_z=1.0,
        diversity=diversity,
        warmup=False,
    )
    loss, stats = action_conditioned_mico_loss(
        z,
        z,
        rewards_t,
        term_t,
        actions_t,
        pair_i,
        pair_j,
        0.99,
        "euclidean",
        0.1,
        1.0,
        pair_weight=w_pos,
    )
    pos_frac = float((w_pos > 0).float().mean())
    print(
        f"E4 nuisance ac_loss={loss.item():.4f} pos_frac={pos_frac:.3f} "
        f"diversity={float(diversity.item()):.3f} undecided={float(und.mean()):.3f}"
    )
    if pos_frac <= 0:
        raise RuntimeError("E4: expected positive pairs for nuisance-equivalent states")


if __name__ == "__main__":
    main()
