"""
PR → 1 diagnostic: cross-seed agreement of top encoder covariance eigenvectors.

High pairwise |cosine| → shared low-D manifold (genuine bisimulation structure).
Low agreement → seed-specific collapse that can fool PR while mu_PL stays positive.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.experiments.aggregate_ablation import _best_full_config
from src.experiments.config import BASE_ARCH_CONFIG, DEFAULT_SEEDS
from src.experiments.plot_ctro import load_seed_metrics
from src.experiments.runner import collect_rollout_buffer, create_critic, create_policy
from src.environments.minigrid_wrapper import MinigridWrapper
from src.metrics.feature_rank import compute_feature_rank_metrics

PROBE_BUFFER_SIZE = 256
PROBE_ENV_SEED = 0
PROBE_CACHE = Path("results/probe_obs_unlock_seed0.pt")
OBS_DIM = 7 * 7 * 3
ENCODE_BATCH = 256


def load_critic_checkpoint(weights_path: Path, device: str) -> torch.nn.Module:
    critic = create_critic(OBS_DIM, BASE_ARCH_CONFIG, device)
    checkpoint = torch.load(weights_path, map_location=device)
    critic.load_state_dict(checkpoint["critic"])
    critic.eval()
    return critic


def collect_probe_obs(device: str) -> torch.Tensor:
    if PROBE_CACHE.exists():
        return torch.load(PROBE_CACHE, map_location=device)

    env = MinigridWrapper("MiniGrid-Unlock-v0", seed=PROBE_ENV_SEED, keep_image_format=False)
    latent_dim = BASE_ARCH_CONFIG["critic"]["latent_dim"]
    policy = create_policy(latent_dim, env.action_dim, BASE_ARCH_CONFIG, device)
    critic = create_critic(env.obs_dim, BASE_ARCH_CONFIG, device)
    buffer = collect_rollout_buffer(env, policy, critic, PROBE_BUFFER_SIZE, device)
    obs = buffer["obs"].to(device)
    PROBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obs.cpu(), PROBE_CACHE)
    return obs


def encode_latents(critic: torch.nn.Module, obs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        mu, _ = critic.encode(obs)
    return mu


def top_cov_eigenvector(z: torch.Tensor) -> torch.Tensor:
    z_c = z - z.mean(dim=0, keepdim=True)
    n = z_c.shape[0]
    cov = (z_c.T @ z_c) / n
    eigvals, eigvecs = torch.linalg.eigh(cov)
    v = eigvecs[:, -1]
    if v[0] < 0:
        v = -v
    return v / v.norm()


def pairwise_abs_cosine(vectors: list[torch.Tensor]) -> np.ndarray:
    n = len(vectors)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            sim[i, j] = abs(torch.dot(vectors[i], vectors[j]).item())
    return sim


def mean_off_diagonal(sim: np.ndarray) -> float:
    n = sim.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(sim[mask].mean())


def analyze_cell(
    results_root: Path,
    exp_path: str,
    seeds: list[int],
    probe_obs: torch.Tensor,
    device: str,
) -> dict:
    vectors = []
    pr_values = []
    mu_pl_values = []

    for seed in seeds:
        weights = results_root / exp_path / f"seed_{seed}" / "weights_final.pt"
        critic = load_critic_checkpoint(weights, device)
        z = encode_latents(critic, probe_obs[:ENCODE_BATCH])
        vectors.append(top_cov_eigenvector(z))
        pr_values.append(compute_feature_rank_metrics(z)["feature_rank_participation_ratio"])

        final_row = load_seed_metrics(results_root, exp_path, seed).iloc[-1]
        mu_pl_values.append(float(final_row["mu_pl_q05"]))

    sim = pairwise_abs_cosine(vectors)
    return {
        "pairwise_abs_cosine": sim.tolist(),
        "mean_off_diagonal_cosine": mean_off_diagonal(sim),
        "per_seed_pr": pr_values,
        "per_seed_mu_pl_q05": mu_pl_values,
        "seeds": seeds,
    }


def plot_cosine_heatmaps(
    cell_results: dict[str, dict],
    output_path: Path,
) -> None:
    names = list(cell_results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        sim = np.array(cell_results[name]["pairwise_abs_cosine"])
        seeds = cell_results[name]["seeds"]
        im = ax.imshow(sim, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(seeds)))
        ax.set_yticks(range(len(seeds)))
        ax.set_xticklabels(seeds)
        ax.set_yticklabels(seeds)
        mean_cos = cell_results[name]["mean_off_diagonal_cosine"]
        ax.set_title(f"{name}\nmean |cos|={mean_cos:.3f}")
        for i in range(len(seeds)):
            for j in range(len(seeds)):
                ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="|cosine similarity|")
    fig.suptitle("Top encoder covariance eigenvector — cross-seed agreement", fontsize=13)
    fig.subplots_adjust(top=0.82, wspace=0.35)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_mu_pl_scatter(cell_results: dict[str, dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    markers = {"FULL": "o", "MICO_ONLY": "s", "BASELINE": "^", "PL_ONLY": "D"}
    colors = {"FULL": "#1f77b4", "MICO_ONLY": "#ff7f0e", "BASELINE": "#d62728", "PL_ONLY": "#2ca02c"}

    for name, res in cell_results.items():
        pr = res["per_seed_pr"]
        mu = res["per_seed_mu_pl_q05"]
        seeds = res["seeds"]
        ax.scatter(
            pr,
            mu,
            label=name,
            marker=markers.get(name, "o"),
            color=colors.get(name, "#333"),
            s=80,
        )
        for pr_i, mu_i, seed in zip(pr, mu, seeds):
            ax.annotate(str(seed), (pr_i, mu_i), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_xlabel("Participation ratio (probe batch)")
    ax.set_ylabel("mu_PL q05 (final training log)")
    ax.set_title("PR vs mu_PL at final checkpoint")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="PR manifold vs collapse diagnostic")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="plots/ctro")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_alpha, best_beta, _ = _best_full_config(results_root, args.seeds)
    full_path = f"exp_full/alpha_{best_alpha}_beta_{best_beta}"

    cells = {
        "FULL": full_path,
        "MICO_ONLY": "exp_mico_only",
        "BASELINE": "exp_baseline",
        "PL_ONLY": "exp_pl_only",
    }

    probe_obs = collect_probe_obs(args.device)
    cell_results = {}
    for name, exp_path in cells.items():
        cell_results[name] = analyze_cell(
            results_root, exp_path, args.seeds, probe_obs, args.device
        )

    plot_cosine_heatmaps(cell_results, output_dir / "manifold_eigenvector_cosine.png")
    plot_pr_mu_pl_scatter(cell_results, output_dir / "pr_vs_mu_pl_probe.png")

    report = {
        name: {
            "mean_off_diagonal_cosine": res["mean_off_diagonal_cosine"],
            "per_seed_pr": res["per_seed_pr"],
            "per_seed_mu_pl_q05": res["per_seed_mu_pl_q05"],
        }
        for name, res in cell_results.items()
    }
    report_path = output_dir / "manifold_diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote {output_dir / 'manifold_eigenvector_cosine.png'}")
    print(f"Wrote {output_dir / 'pr_vs_mu_pl_probe.png'}")
    print(f"Wrote {report_path}")
    print("\nCross-seed top eigenvector |cosine| (off-diagonal mean):")
    for name, res in cell_results.items():
        print(f"  {name}: {res['mean_off_diagonal_cosine']:.4f}")


if __name__ == "__main__":
    main()
