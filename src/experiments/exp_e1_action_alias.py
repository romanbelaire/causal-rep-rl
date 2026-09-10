"""E1: action-alias toy — AC-MICo/Q vs random-pair MICo (CPU)."""

import torch
import torch.nn.functional as F

from src.architectures.critics.q_ensemble import QEnsemble
from src.environments.toys import ActionAliasToy
from src.losses.action_conditioned_mico import action_conditioned_mico_loss
from src.losses.mico import compute_mico_loss
from src.losses.q_td import policy_value_from_q, q_td_loss, member_bootstrap_mask
from src.replay.pair_index import select_same_action_pairs
from src.utils.bisimulation_utils import VAEEncoderTarget


class _TinyEncoder(torch.nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Linear(obs_dim, latent_dim)
        self.fc_mu = torch.nn.Linear(latent_dim, latent_dim)
        self.value_head = torch.nn.Linear(latent_dim, 1)

    def encode(self, obs):
        h = self.encoder(obs)
        return self.fc_mu(h), torch.zeros_like(h)

    def forward(self, obs):
        mu, _ = self.encode(obs)
        return self.value_head(mu)


def _collect(env, n=128):
    obs, actions, rewards, term, next_obs = [], [], [], [], []
    for _ in range(n):
        o, _ = env.reset()
        a = int(torch.randint(0, env.action_dim, (1,)).item())
        no, r, te, tr, _ = env.step(a)
        obs.append(o)
        actions.append(a)
        rewards.append(r)
        term.append(te)
        next_obs.append(no)
    return (
        torch.stack(obs),
        torch.tensor(actions),
        torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(term, dtype=torch.bool),
        torch.stack(next_obs),
    )


def main():
    torch.manual_seed(0)
    env = ActionAliasToy(seed=0)
    critic = _TinyEncoder(env.obs_dim, 8)
    target = VAEEncoderTarget(critic)
    q = QEnsemble(8, env.action_dim, n_members=3, hidden_sizes=[16])
    obs, actions, rewards, term, next_obs = _collect(env, 256)
    z = critic.encode(obs)[0]
    q_out = q(z)
    pi = F.one_hot(actions, env.action_dim).float()
    v_bar = policy_value_from_q(q_out, pi)
    mask = member_bootstrap_mask(actions.shape[0], 3, obs.device)
    q_loss, _, _ = q_td_loss(
        q_out[torch.arange(actions.shape[0]), :, actions],
        rewards,
        term,
        v_bar,
        0.99,
        1.0,
        mask,
    )
    pair_i, pair_j, _ = select_same_action_pairs(actions, z, n_candidates=4)
    ac_loss, _ = action_conditioned_mico_loss(
        z,
        target.encode_mu(next_obs),
        rewards,
        term,
        actions,
        pair_i,
        pair_j,
        0.99,
        "euclidean",
        0.1,
        1.0,
    )
    rand_loss, _ = compute_mico_loss(
        critic, target, obs, next_obs, rewards, 0.99, repr_net=None
    )
    alias0 = (actions == 0) & (rewards > 5)
    alias1 = (actions == 1) & (rewards < 1)
    separation = float((alias0 | alias1).float().mean())
    print(
        f"E1 action_alias q_loss={q_loss.item():.4f} "
        f"ac_mico={ac_loss.item():.4f} rand_mico={rand_loss.item():.4f} "
        f"alias_rate={separation:.3f}"
    )
    if separation <= 0:
        raise RuntimeError("E1 toy failed to produce alias episodes")


if __name__ == "__main__":
    main()
