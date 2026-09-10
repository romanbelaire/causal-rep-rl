"""Reward-test signatures: empirical D_hat_sig (toys) and estimated Q-vector surrogate."""

import torch


def intervene_reward_strings(
    env,
    state_setter,
    tests: list[tuple[int, ...]],
    n_repeats: int,
) -> torch.Tensor:
    """Reset-and-intervene evaluator. Returns counts [n_hist, n_tests, n_bins].

    `state_setter(env, h)` places the toy in history `h`. Reward strings are
    hashed into a fixed bin via the integer reward sequence.
    """
    n_hist = env.n_histories
    n_tests = len(tests)
    # bins keyed by a small integer code of the reward string
    codes: dict[tuple, int] = {}
    counts = torch.zeros(n_hist, n_tests, 64)
    for h in range(n_hist):
        for u_idx, test in enumerate(tests):
            for _ in range(n_repeats):
                state_setter(env, h)
                rewards = []
                for a in test:
                    _obs, r, terminated, truncated, _info = env.step(a)
                    rewards.append(int(round(float(r))))
                    if terminated or truncated:
                        break
                key = tuple(rewards)
                if key not in codes:
                    if len(codes) >= 64:
                        raise RuntimeError("reward-string codebook overflow (64 bins)")
                    codes[key] = len(codes)
                counts[h, u_idx, codes[key]] += 1.0
    return counts


def signature_from_counts(counts: torch.Tensor) -> torch.Tensor:
    """Normalize counts to a probability embedding per (history, test)."""
    mass = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return counts / mass


def d_hat_sig(
    sig_h: torch.Tensor,
    sig_hp: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Empirical pseudometric: sum_u w_u ||sig[h,u] - sig[h',u]||_2.

    sig_* : [n_tests, n_bins]
    """
    if sig_h.shape != sig_hp.shape:
        raise RuntimeError(f"signature shape mismatch {tuple(sig_h.shape)} vs {tuple(sig_hp.shape)}")
    if weights.shape[0] != sig_h.shape[0]:
        raise RuntimeError("weight count must match number of tests")
    delta = (sig_h - sig_hp).norm(dim=-1)
    return (weights * delta).sum()


def pairwise_d_hat_sig(signatures: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """signatures [n_hist, n_tests, n_bins] -> [n_hist, n_hist] matrix."""
    n = signatures.shape[0]
    mat = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            mat[i, j] = d_hat_sig(signatures[i], signatures[j], weights)
    return mat


def check_pseudometric(mat: torch.Tensor, atol: float = 1e-6) -> dict[str, float]:
    """Return violation magnitudes. Zeros mean the matrix is a pseudometric."""
    n = mat.shape[0]
    if mat.shape != (n, n):
        raise RuntimeError(f"expected square matrix, got {tuple(mat.shape)}")
    nonneg = float(mat.min().clamp(max=0.0).abs().item())
    sym = float((mat - mat.T).abs().max().item())
    diag = float(mat.diag().abs().max().item())
    tri = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                gap = float((mat[i, j] - mat[i, k] - mat[k, j] - atol).item())
                if gap > tri:
                    tri = gap
    return {
        "negativity": nonneg,
        "asymmetry": sym,
        "diag_mass": diag,
        "triangle_violation": max(tri, 0.0),
    }


def estimated_q_signature(q_mean: torch.Tensor) -> torch.Tensor:
    """Online estimated signature: mean ensemble Q-vector as length-1 tests.

    q_mean: [batch, n_actions]
    Each action is a test; the embedding is the scalar Q, stored as a 1-bin vector.
    """
    return q_mean.unsqueeze(-1)


def estimated_d_hat_sig_mean(q_mean: torch.Tensor, n_pairs: int = 256) -> dict[str, float]:
    """Mean estimated D_hat_sig on random pairs. Diagnostic only, not a loss."""
    n = q_mean.shape[0]
    if n < 2:
        raise RuntimeError("estimated D_hat_sig needs at least 2 states")
    n_pairs = min(n_pairs, n * (n - 1))
    i = torch.randint(0, n, (n_pairs,), device=q_mean.device)
    j = torch.randint(0, n, (n_pairs,), device=q_mean.device)
    same = i == j
    j = j.clone()
    j[same] = (j[same] + 1) % n
    w = torch.ones(q_mean.shape[1], device=q_mean.device)
    sig = estimated_q_signature(q_mean)
    vals = []
    for k in range(n_pairs):
        vals.append(d_hat_sig(sig[i[k]], sig[j[k]], w))
    stacked = torch.stack(vals)
    return {
        "D_hat_sig_estimated_mean": float(stacked.mean().item()),
        "D_hat_sig_estimated_p50": float(stacked.median().item()),
        "D_hat_sig_source": 1.0,
    }
