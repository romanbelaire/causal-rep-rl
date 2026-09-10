"""T4: D_hat_sig is a pseudometric; raw D_hat_pair need not be. CPU, deterministic."""

import torch

from src.environments.toys import FiniteRewardTestToy
from src.metrics.reward_test_signature import (
    check_pseudometric,
    intervene_reward_strings,
    pairwise_d_hat_sig,
    signature_from_counts,
)


def _setter(env: FiniteRewardTestToy, h: int) -> None:
    env.set_history(h)


def main() -> None:
    torch.manual_seed(4)
    env = FiniteRewardTestToy(seed=4)
    tests = [(0,), (1,)]
    counts = intervene_reward_strings(env, _setter, tests, n_repeats=16)
    sig = signature_from_counts(counts)
    weights = torch.ones(len(tests))
    mat = pairwise_d_hat_sig(sig, weights)
    viol = check_pseudometric(mat, atol=1e-6)
    print(
        f"T4 D_hat_sig neg={viol['negativity']:.2e} asym={viol['asymmetry']:.2e} "
        f"diag={viol['diag_mass']:.2e} tri={viol['triangle_violation']:.2e} "
        f"d01={float(mat[0,1]):.4f} d02={float(mat[0,2]):.4f}"
    )
    if viol["negativity"] > 1e-8 or viol["asymmetry"] > 1e-8 or viol["diag_mass"] > 1e-8:
        raise RuntimeError(f"T4: D_hat_sig failed pseudometric axioms: {viol}")
    if viol["triangle_violation"] > 0:
        raise RuntimeError(f"T4: D_hat_sig triangle failed: {viol['triangle_violation']}")
    if float(mat[0, 1]) > 1e-6:
        raise RuntimeError("T4: reward-test equivalent histories were not at D_hat_sig=0")
    if float(mat[0, 2]) <= 1e-6:
        raise RuntimeError("T4: distinct reward-test histories collapsed in D_hat_sig")

    # Directed residual (r_i - r_j) is the raw pairwise score without metric axioms.
    r = torch.tensor([1.0, 1.0, 0.0])
    directed = r.unsqueeze(0) - r.unsqueeze(1)
    directed_viol = check_pseudometric(directed, atol=1e-6)
    if directed_viol["asymmetry"] <= 1e-8:
        raise RuntimeError("T4: expected the raw directed pair score to fail symmetry")
    print(
        f"T4 D_hat_pair_raw asymmetry={directed_viol['asymmetry']:.3f} "
        "(surrogate, not a pseudometric)"
    )
    print("T4 empirical_pseudometric ok")


if __name__ == "__main__":
    main()
