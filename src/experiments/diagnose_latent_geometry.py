"""Diagnose latent covariance/participation-ratio and pairwise distances (CPU).

Gate B1 / Phase 1: if p5 of pairwise distances is near zero, geometry is ill-defined.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.experiments.performance_models import build_performance_stack_from_config
from src.metrics.feature_rank import compute_feature_rank_metrics


DISTANCE_EPS = 1e-8


def _action_info(policy_sd: dict) -> tuple[int, str]:
    if "action_mean.weight" in policy_sd:
        return int(policy_sd["action_mean.weight"].shape[0]), "continuous"
    if "action_head.weight" in policy_sd:
        return int(policy_sd["action_head.weight"].shape[0]), "discrete"
    raise KeyError("Cannot infer action dim from policy state_dict")


def load_stack(run_dir: Path, device: str = "cpu"):
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    weights = run_dir / "weights_final.pt"
    if not weights.exists():
        weights = run_dir / "weights_latest.pt"
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    action_dim, action_space_type = _action_info(ckpt["policy"])

    obs_shape = config.get("obs_shape")
    if obs_shape is None:
        w = ckpt["critic"]["encoder.0.weight"]
        obs_dim = int(w.shape[1])
        obs_shape_t = None
    else:
        obs_dim = 0
        obs_shape_t = tuple(obs_shape)

    stack = build_performance_stack_from_config(
        {
            "architecture": config["architecture"],
            "agent_class": config["agent_class"],
            "stack_type": config.get("stack_type"),
        },
        obs_dim=obs_dim if obs_shape is None else 0,
        action_dim=action_dim,
        action_space_type=action_space_type,
        obs_shape=obs_shape_t,
        device=device,
    )
    stack.critic.load_state_dict(ckpt["critic"])
    stack.policy.load_state_dict(ckpt["policy"])
    stack.critic.eval()
    stack.policy.eval()
    return stack, config, ckpt


def encode_probe(stack, n: int, device: str, config: dict) -> torch.Tensor:
    if not hasattr(stack.critic, "encode"):
        raise RuntimeError("critic lacks encode()")
    torch.manual_seed(0)
    obs_shape = config.get("obs_shape")
    if obs_shape is not None:
        # Pixel stack: CHW float in [0, 1]
        c, h, w = int(obs_shape[0]), int(obs_shape[1]), int(obs_shape[2])
        obs = torch.rand(n, c, h, w, device=device)
    else:
        first = stack.critic.encoder[0]
        in_features = first.in_features
        obs = torch.randn(n, in_features, device=device)
    with torch.no_grad():
        z, _ = stack.critic.encode(obs)
    return z


def main() -> None:
    parser = argparse.ArgumentParser(description="Latent geometry PR / distance diagnostic")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.device != "cpu":
        raise SystemExit("diagnose_latent_geometry must run on CPU")

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "geometry_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    stack, config, _ckpt = load_stack(run_dir, device=args.device)
    z = encode_probe(stack, args.n, args.device, config)

    rank = compute_feature_rank_metrics(z)
    zc = z - z.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(zc, full_matrices=False)
    eig = (s ** 2) / max(z.shape[0] - 1, 1)
    eig_np = eig.numpy()

    dmat = torch.cdist(z, z)
    iu = torch.triu_indices(z.shape[0], z.shape[0], offset=1)
    dists = dmat[iu[0], iu[1]]
    d_np = dists.numpy()
    p5 = float(np.quantile(d_np, 0.05))
    p50 = float(np.median(d_np))
    ratio = p5 / p50 if p50 > DISTANCE_EPS else 0.0
    collapsed = p5 <= DISTANCE_EPS

    report = {
        "run_dir": str(run_dir),
        "n_samples": args.n,
        "latent_dim": int(z.shape[1]),
        "participation_ratio": rank["feature_rank_participation_ratio"],
        "feature_rank_spectral": rank["feature_rank_spectral"],
        "feature_rank_pca": rank["feature_rank_pca"],
        "eigenvalues": eig_np.tolist(),
        "pairwise_distance_p05": p5,
        "pairwise_distance_median": p50,
        "pairwise_p05_over_median": ratio,
        "collapsed": collapsed,
        "gate_b1_ok": not collapsed,
        "gate_phase1_ok": not collapsed,
        "interpretation": (
            "metric collapse (p5~0); fix encoder before E-A"
            if collapsed
            else (
                "dominant low-rank manifold, distances OK"
                if rank["feature_rank_participation_ratio"] < 1.5
                else "healthy latent geometry"
            )
        ),
    }

    with open(out_dir / "geometry_report.json", "w") as f:
        json.dump(report, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(np.arange(1, len(eig_np) + 1), eig_np + 1e-30, marker="o")
    axes[0].set_title("Latent covariance eigenvalues")
    axes[0].set_xlabel("rank")
    axes[0].set_ylabel("lambda")
    axes[1].hist(d_np, bins=50, color="#4c72b0", edgecolor="none")
    axes[1].axvline(p5, color="C1", linestyle="--", label=f"p5={p5:.3g}")
    axes[1].axvline(p50, color="C3", linestyle="--", label=f"med={p50:.3g}")
    axes[1].set_title("Pairwise Euclidean distances")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "geometry.png", dpi=120)
    plt.close(fig)

    print(json.dumps({k: v for k, v in report.items() if k != "eigenvalues"}, indent=2))
    print(f"eigenvalues (first 10): {eig_np[:10]}")
    print(f"Wrote {out_dir / 'geometry_report.json'} and geometry.png")
    if collapsed:
        raise SystemExit(
            "GATE FAIL: pairwise distance p5 near zero — latent collapse; E-A blocked"
        )


if __name__ == "__main__":
    main()
