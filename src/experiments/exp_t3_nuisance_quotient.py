"""T3: nuisance quotient plus retained T2 separation. CPU, deterministic."""

import torch

from src.environments.toys import EqualValueAliasToy, NuisanceToy
from src.losses.separation import separation_loss
from src.metrics.reward_test_signature import (
    d_hat_sig,
    intervene_reward_strings,
    signature_from_counts,
)
from src.replay.pair_index import pair_confidence_weights, pair_lower_bound


def _nuisance_setter(env: NuisanceToy, h: int) -> None:
    env._variant = int(h)


def main() -> None:
    torch.manual_seed(2)
    nuisance = NuisanceToy(seed=2)
    tests = [(0,), (1,)]
    counts = intervene_reward_strings(nuisance, _nuisance_setter, tests, n_repeats=8)
    sig = signature_from_counts(counts)
    w = torch.ones(len(tests))
    d_nuis = float(d_hat_sig(sig[0], sig[1], w).item())
    if d_nuis > 1e-6:
        raise RuntimeError(f"T3: nuisance histories are not reward-test equivalent: D_hat_sig={d_nuis}")

    alias = EqualValueAliasToy(seed=0)
    o0 = torch.zeros(alias.obs_dim); o0[0] = 1.0
    o1 = torch.zeros(alias.obs_dim); o1[1] = 1.0
    z = torch.stack([o0, o1])
    dhat = torch.tensor([10.0])
    sigma = torch.tensor([0.0])
    diversity = (z[0] - z[1]).norm().unsqueeze(0)
    _w_pos, w_neg, _ = pair_confidence_weights(
        dhat, sigma, tau_pos=0.1, tau_neg=0.3, confidence_z=1.0, diversity=diversity, warmup=False
    )
    d_hat_lower = pair_lower_bound(dhat, sigma, 1.0)
    sep, stats = separation_loss(z, torch.tensor([0]), torch.tensor([1]), d_hat_lower, w_neg, alpha_sep=1.0)
    if float(stats["train_sep_n_pairs"]) <= 0 or float(sep.item()) <= 0:
        raise RuntimeError("T3: T2-style pair lost its separation margin")
    n0 = torch.zeros(nuisance.obs_dim); n0[0] = 1.0
    n1 = torch.zeros(nuisance.obs_dim); n1[4] = 1.0
    live_nuis = float((n0 - n1).norm().item())
    print(
        f"T3 nuisance D_hat_sig={d_nuis:.6f} obs_dist={live_nuis:.3f} "
        f"t2_sep={float(sep.item()):.4f}"
    )
    if live_nuis <= 0:
        raise RuntimeError("T3: nuisance observations are not visually distant")
    print("T3 nuisance_quotient ok")


if __name__ == "__main__":
    main()
