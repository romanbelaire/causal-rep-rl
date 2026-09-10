"""Query/control mixture with an explicit uniform coverage floor."""

import torch
import torch.nn.functional as F


def sample_behavior_actions(
    ctrl_logits: torch.Tensor,
    query_logits: torch.Tensor,
    epsilon: float,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """beta = (1-eps-eta) pi_ctrl + eps pi_query + eta uniform.

    Returns action, log beta, log pi_ctrl.
    """
    n_actions = ctrl_logits.shape[-1]
    if eta <= 0:
        raise RuntimeError(f"uniform_floor_eta must be > 0 for query collection, got {eta}")
    if epsilon < 0 or eta < 0 or (epsilon + eta) >= 1:
        raise RuntimeError(f"invalid mixture weights eps={epsilon} eta={eta}")
    ctrl_probs = F.softmax(ctrl_logits, dim=-1)
    query_probs = F.softmax(query_logits, dim=-1)
    uniform = torch.full_like(ctrl_probs, 1.0 / n_actions)
    mix = (1.0 - epsilon - eta) * ctrl_probs + epsilon * query_probs + eta * uniform
    floor = eta / n_actions
    if (mix < floor - 1e-6).any():
        raise RuntimeError("behavior mixture violated the uniform coverage floor")
    dist = torch.distributions.Categorical(probs=mix)
    action = dist.sample()
    log_beta = dist.log_prob(action)
    log_ctrl = torch.distributions.Categorical(logits=ctrl_logits).log_prob(action)
    return action, log_beta, log_ctrl
