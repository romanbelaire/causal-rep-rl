"""CPU validation of replay / MICo / relational invariants (deterministic exit)."""

import torch

from src.architectures.critics.vae_critic import VAECritic
from src.losses.action_conditioned_mico import action_conditioned_mico_loss
from src.metrics.relational_trust_region import relational_distortion
from src.replay.priorities import behavior_policy_ratio, mix_coverage, priority_scores, replay_is_weights
from src.replay.transition_replay import TransitionReplay


def main() -> None:
    torch.manual_seed(0)
    replay = TransitionReplay(capacity=16)
    obs = torch.randn(8, 4)
    next_obs = torch.randn(8, 4)
    action = torch.randint(0, 3, (8,))
    reward = torch.randn(8)
    terminated = torch.zeros(8, dtype=torch.bool)
    terminated[3] = True
    truncated = torch.zeros(8, dtype=torch.bool)
    truncated[5] = True
    replay.add_batch(
        obs=obs,
        next_obs=next_obs,
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        episode_id=torch.arange(8),
        step_id=torch.zeros(8, dtype=torch.long),
        behavior_log_prob=torch.randn(8),
        control_log_prob=torch.randn(8),
        source="query",
        policy_version=1,
        encoder_version=1,
    )
    gathered = replay.gather(torch.tensor([3, 5]))
    if not gathered["terminated"][0]:
        raise RuntimeError("replay dropped terminated flag at stored index 3")
    if not gathered["truncated"][1]:
        raise RuntimeError("replay dropped truncated flag at stored index 5")

    pri = priority_scores(
        torch.zeros(8),
        torch.zeros(8),
        torch.zeros(8),
        torch.zeros(8),
        floor=0.001,
        lambda_ig=1.0,
        lambda_gap=0.5,
        lambda_pair=1.0,
        lambda_geom=0.5,
    )
    q = mix_coverage(pri, 0.1)
    w = replay_is_weights(q, 8, 0.4)
    _ratio = behavior_policy_ratio(torch.zeros(4), torch.zeros(4))

    critic = VAECritic(4, latent_dim=4, encoder_hidden=[8], decoder_hidden=[8], value_hidden=[8])
    phi = torch.randn(6, 4)
    phi_t = torch.randn(6, 4)
    actions = torch.tensor([0, 0, 1, 1, 2, 2])
    pair_i = torch.tensor([0, 2, 4])
    pair_j = torch.tensor([1, 3, 5])
    term = torch.zeros(6, dtype=torch.bool)
    rew = torch.randn(6)
    try:
        action_conditioned_mico_loss(
            phi,
            phi_t,
            rew,
            term,
            actions,
            pair_i,
            torch.tensor([1, 3, 4]),
            0.99,
            "euclidean",
            0.1,
            1.0,
        )
        raise RuntimeError("AC-MICo should refuse unequal-action pairs")
    except RuntimeError:
        pass

    action_conditioned_mico_loss(
        phi,
        phi_t,
        rew,
        term,
        actions,
        pair_i,
        pair_j,
        0.99,
        "euclidean",
        0.1,
        1.0,
    )

    z = torch.randn(8, 4)
    relational_distortion(z, z, min_old_distance=1e-3)
    print("validate_invariants ok")


if __name__ == "__main__":
    main()
