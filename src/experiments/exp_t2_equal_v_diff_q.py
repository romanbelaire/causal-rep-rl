"""T2: equal V^pi, different unused-action Q. CPU, deterministic."""

import torch

from src.environments.toys import EqualValueAliasToy
from src.losses.action_conditioned_mico import action_conditioned_mico_loss
from src.losses.separation import separation_loss
from src.replay.pair_index import pair_confidence_weights, pair_lower_bound, select_same_action_pairs


def _collect_by_action(env: EqualValueAliasToy, action: int, n: int, alias: int):
    obs, rewards = [], []
    for _ in range(n):
        env._alias = alias
        o = env._obs()
        _no, r, _te, _tr, _ = env.step(action)
        obs.append(o)
        rewards.append(r)
    return torch.stack(obs), torch.tensor(rewards, dtype=torch.float32)


def main() -> None:
    torch.manual_seed(0)
    env = EqualValueAliasToy(seed=0)
    o0, r0 = _collect_by_action(env, action=0, n=16, alias=0)
    o1, r1 = _collect_by_action(env, action=0, n=16, alias=1)
    v_pi_0 = float(r0.mean())
    v_pi_1 = float(r1.mean())
    if abs(v_pi_0 - v_pi_1) > 1e-6:
        raise RuntimeError(f"T2: action-0 values differ: {v_pi_0} vs {v_pi_1}")
    o0_a1, r0_a1 = _collect_by_action(env, action=1, n=16, alias=0)
    o1_a1, r1_a1 = _collect_by_action(env, action=1, n=16, alias=1)
    q_gap = float((r0_a1.mean() - r1_a1.mean()).abs())
    if q_gap < 5.0:
        raise RuntimeError(f"T2: unused-action Q gap too small: {q_gap}")
    obs = torch.cat([o0_a1, o1_a1], dim=0)
    rewards = torch.cat([r0_a1, r1_a1], dim=0)
    actions = torch.cat([torch.ones(16, dtype=torch.long), torch.ones(16, dtype=torch.long)])
    term = torch.ones(32, dtype=torch.bool)
    z = obs.clone().requires_grad_(True)
    pair_i, pair_j, diversity = select_same_action_pairs(actions, z.detach(), n_candidates=8)
    dhat = (rewards[pair_i] - rewards[pair_j]).abs()
    sigma = torch.zeros_like(dhat)
    w_pos, w_neg, _und = pair_confidence_weights(
        dhat, sigma, tau_pos=0.1, tau_neg=0.3, confidence_z=1.0, diversity=diversity, warmup=False
    )
    d_hat_lower = pair_lower_bound(dhat, sigma, 1.0)
    if float((w_neg > 0).float().mean()) <= 0:
        raise RuntimeError("T2: expected confident negative pairs on action 1")
    ac_loss, _ = action_conditioned_mico_loss(
        z, z.detach(), rewards, term, actions, pair_i, pair_j, 0.99, "euclidean", 0.1, 1.0, pair_weight=w_pos
    )
    sep, sep_stats = separation_loss(z, pair_i, pair_j, d_hat_lower, w_neg, alpha_sep=1.0)
    z_merged = torch.zeros_like(z)
    merge_d = (z_merged[pair_i] - z_merged[pair_j]).norm(dim=1).mean()
    live_d = (z[pair_i][w_neg > 0] - z[pair_j][w_neg > 0]).norm(dim=1).mean()
    print(
        f"T2 equal_v q_gap={q_gap:.3f} neg_frac={float((w_neg>0).float().mean()):.3f} "
        f"sep={float(sep.item()):.4f} live_d={float(live_d):.3f} merge_d={float(merge_d):.3f} "
        f"ac={float(ac_loss.item()):.4f} n_sep={float(sep_stats['train_sep_n_pairs'])}"
    )
    if float(live_d) <= float(merge_d):
        raise RuntimeError("T2: value-only merge is not farther-collapsing than the true obs embedding")
    if float(sep.item()) <= 0:
        raise RuntimeError("T2: L_sep did not fire on the Q-distinct pair")
    print("T2 equal_v_diff_q ok")


if __name__ == "__main__":
    main()
