"""Pair equivalence bookkeeping: positive / negative / undecided fractions."""


def pair_equivalence_stats(w_pos, w_neg, undecided) -> dict[str, float]:
    n = w_pos.numel()
    n_pos = float((w_pos > 0).sum().item())
    n_neg = float((w_neg > 0).sum().item())
    n_und = float((undecided > 0).sum().item())
    return {
        "pair_n": float(n),
        "pair_frac_positive": n_pos / n,
        "pair_frac_negative": n_neg / n,
        "pair_frac_undecided": n_und / n,
        "pair_weight_pos_mean": float(w_pos.mean().item()),
        "pair_weight_neg_mean": float(w_neg.mean().item()),
    }
