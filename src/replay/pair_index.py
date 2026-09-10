"""Same-action pair mining with confidence weights."""

import torch


def select_same_action_pairs(
    actions: torch.Tensor,
    z: torch.Tensor,
    n_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Anchor each row to a same-action partner, preferring latent diversity.

    Returns (anchor_index, partner_index, latent_distance).
    """
    if actions.dim() != 1:
        raise RuntimeError(f"actions must be rank-1, got shape {tuple(actions.shape)}")
    n = actions.shape[0]
    if n < 2:
        raise RuntimeError("need at least 2 transitions to form pairs")
    anchor = []
    partner = []
    dist = []
    for i in range(n):
        same = (actions == actions[i]) & (torch.arange(n, device=actions.device) != i)
        cand = torch.where(same)[0]
        if cand.numel() == 0:
            continue
        if cand.numel() > n_candidates:
            perm = torch.randperm(cand.numel(), device=actions.device)[:n_candidates]
            cand = cand[perm]
        delta = z[cand] - z[i].unsqueeze(0)
        d = delta.pow(2).sum(dim=1).sqrt()
        j = cand[d.argmax()]
        anchor.append(i)
        partner.append(int(j.item()))
        dist.append(d.max())
    if len(anchor) == 0:
        raise RuntimeError("AC-MICo batch has no same-action pairs")
    i_t = torch.tensor(anchor, device=actions.device, dtype=torch.long)
    j_t = torch.tensor(partner, device=actions.device, dtype=torch.long)
    if (actions[i_t] != actions[j_t]).any():
        raise RuntimeError("pair index produced unequal-action pairs")
    d_t = torch.stack(dist)
    return i_t, j_t, d_t


def pair_confidence_weights(
    discrepancy: torch.Tensor,
    sigma: torch.Tensor,
    tau_pos: float,
    tau_neg: float,
    confidence_z: float,
    diversity: torch.Tensor,
    warmup: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (w_pos, w_neg, undecided). Nonfinite sigma cannot be positive."""
    if warmup:
        zeros = torch.zeros_like(discrepancy)
        undecided = torch.ones_like(discrepancy)
        return zeros, zeros, undecided
    finite = torch.isfinite(sigma) & torch.isfinite(discrepancy)
    ucb = discrepancy + confidence_z * sigma
    lcb = discrepancy - confidence_z * sigma
    w_pos = ((ucb <= tau_pos) & finite).float() * diversity
    if (~finite & (w_pos > 0)).any():
        raise RuntimeError("nonfinite pair uncertainty produced a positive pair")
    if ((~torch.isfinite(sigma)) & (w_pos > 0)).any():
        raise RuntimeError("uninitialized pair uncertainty produced a positive pair")
    w_neg = ((lcb >= tau_neg) & finite).float()
    undecided = (~(w_pos > 0) & ~(w_neg > 0)).float()
    return w_pos, w_neg, undecided


def pair_lower_bound(
    discrepancy: torch.Tensor,
    sigma: torch.Tensor,
    confidence_z: float,
) -> torch.Tensor:
    """Conservative D_hat_lower = D_hat_pair - c * sigma. Not a metric."""
    return discrepancy - confidence_z * sigma
