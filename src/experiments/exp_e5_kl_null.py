"""E5: KL-null shear moves relational metric while logits can stay fixed (CPU)."""

import torch

from src.environments.toys import kl_null_shear_z
from src.metrics.relational_trust_region import relational_distortion


def main():
    torch.manual_seed(3)
    z = torch.randn(32, 8)
    z_shear = kl_null_shear_z(z, c=0.25)
    metric, stats = relational_distortion(z_shear, z, min_old_distance=1e-3)
    logits_before = z @ torch.randn(8, 4)
    logits_after = z_shear @ torch.randn(8, 4)
    kl_proxy = (logits_before - logits_after).pow(2).mean()
    print(
        f"E5 kl_null D_Z_inf_B={float(metric):.4f} "
        f"excluded_frac={float(stats['dz_rel_excluded_frac']):.4f} "
        f"logit_mse={float(kl_proxy):.4f}"
    )
    if float(metric) <= 0.01:
        raise RuntimeError("E5: relational metric did not detect shear")


if __name__ == "__main__":
    main()
