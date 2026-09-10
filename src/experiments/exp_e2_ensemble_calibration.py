"""E2: ensemble disagreement vs scalar V (CPU calibration smoke)."""

import torch
import torch.nn.functional as F

from src.architectures.critics.q_ensemble import QEnsemble
from src.environments.toys import ActionAliasToy
from src.losses.q_td import member_bootstrap_mask, policy_value_from_q, q_td_loss


def main():
    torch.manual_seed(1)
    env = ActionAliasToy(seed=1)
    z = torch.randn(64, 8)
    actions = torch.randint(0, 2, (64,))
    rewards = torch.randn(64)
    term = torch.zeros(64, dtype=torch.bool)
    q = QEnsemble(8, 2, n_members=5, hidden_sizes=[16])
    q_out = q(z)
    pi = F.one_hot(actions, 2).float()
    v_bar = policy_value_from_q(q_out, pi)
    mask = member_bootstrap_mask(64, 5, z.device)
    _, stats, disagreement = q_td_loss(
        q_out[torch.arange(64), :, actions],
        rewards,
        term,
        v_bar,
        0.99,
        1.0,
        mask,
    )
    v_scalar = q_out.mean(dim=1)[torch.arange(64), actions]
    corr = torch.corrcoef(
        torch.stack([disagreement, (v_scalar - rewards).abs()])
    )[0, 1]
    print(
        f"E2 ensemble_calibration target_var={stats['train_q_target_var_mean']:.4f} "
        f"disagreement_td_corr={float(corr):.4f}"
    )


if __name__ == "__main__":
    main()
