"""T5: relational preservation of a trusted-pair margin. CPU, deterministic."""

import torch

from src.environments.toys import kl_null_shear_z
from src.losses.separation import separation_loss
from src.metrics.relational_trust_region import relational_distortion


def main() -> None:
    torch.manual_seed(5)
    z = torch.randn(32, 8)
    z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-3)
    pair_i = torch.arange(0, 16)
    pair_j = torch.arange(16, 32)
    d_old = (z[pair_i] - z[pair_j]).norm(dim=1)
    w_neg = torch.ones_like(d_old)
    sep0, _ = separation_loss(z, pair_i, pair_j, d_old, w_neg, alpha_sep=1.0)
    if float(sep0.item()) > 1e-6:
        raise RuntimeError("T5: trusted pairs should sit on the margin before contraction")

    z_bad = kl_null_shear_z(z, c=0.25)
    metric_bad, stats_bad = relational_distortion(z_bad, z, min_old_distance=1e-3)
    if float(stats_bad["dz_rel_all_excluded"]) > 0.5:
        raise RuntimeError("T5: denominators were excluded; cannot test preservation")
    if float(metric_bad) <= 0.2:
        raise RuntimeError(f"T5: contraction was not detected: D_Z_inf_B={float(metric_bad):.4f}")
    sep_bad, _ = separation_loss(z_bad, pair_i, pair_j, d_old, w_neg, alpha_sep=1.0)
    if float(sep_bad.item()) <= float(sep0.item()):
        raise RuntimeError("T5: contracted encoder did not increase L_sep")

    eps = 0.10
    z_ok = z * (1.0 - 0.5 * eps)
    metric_ok, stats_ok = relational_distortion(z_ok, z, min_old_distance=1e-3)
    if float(metric_ok) > eps:
        raise RuntimeError(
            f"T5: a (1-eps/2) update should be accepted, got D_Z_inf_B={float(metric_ok):.4f}"
        )
    live_ok = (z_ok[pair_i] - z_ok[pair_j]).norm(dim=1)
    if (live_ok < (1.0 - eps) * d_old - 1e-5).any():
        raise RuntimeError("T5: accepted (1-eps) update did not preserve the lower margin")
    print(
        f"T5 D_Z_inf_B_bad={float(metric_bad):.4f} D_Z_inf_B_ok={float(metric_ok):.4f} "
        f"excluded={float(stats_ok['dz_rel_excluded_frac']):.3f} sep_bad={float(sep_bad.item()):.4f}"
    )
    print("T5 relational_preservation ok")


if __name__ == "__main__":
    main()
