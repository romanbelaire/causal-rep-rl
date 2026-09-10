"""
Training runner for Procgen and DMControl performance environments.

Trains one task and writes checkpoints to the layout expected by run_performance_eval:
  results/{suite_prefix}/{exp_name}/seed_{seed}/{task}/weights_final.pt
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.agents.ctro import CTRO
from src.agents.ppo import PPO
from src.environments.vec_env import make_train_vec_env
from src.evaluation.runner import make_env, run_eval_episodes
from src.evaluation.suites import DistributionSpec, EVAL_SUITES
from src.experiments.config import (
    BASE_ALGO_CONFIG,
    DMCONTROL_COLLAPSE_FLOORS,
    PERFORMANCE_SUITE_CONFIG,
)
from src.experiments.performance_models import PerformanceStack, build_performance_stack
from src.experiments.runner import set_seed
from src.losses.dz_trust_region import compute_dz, compute_s_ref, pairwise_distance_histogram
from src.metrics.collapse_probes import (
    dormant_unit_fraction,
    functional_value_probe_mse,
    near_zero_pair_rate,
)
from src.metrics.pl_ratio import value_gap_histogram
from src.utils.best_episode_recorder import BestEpisodeFrameRecorder, make_best_episode_frame_recorder
from src.utils.bisimulation_utils import encode_phi
from src.utils.ctro_metric_evaluator import CTROMetricEvaluator
from src.utils.logging import CSVLogger, truncate_metrics_csv
from src.utils.normalization import PerformanceNormalizer


def _ale_task_short(task: str) -> str:
    # ALE/Phoenix-v5 -> Phoenix
    name = task.split("/")[-1]
    if name.endswith("-v5"):
        name = name[: -len("-v5")]
    return name


def _resolve_dz_diag_path(run_dir: Path, suite_name: str, task: str) -> Path:
    if suite_name == "ale":
        shared = Path("results") / "ale" / "diag" / _ale_task_short(task) / "diag.pt"
        if shared.is_file():
            return shared
        return run_dir / "diag.pt"
    # Prefer suite-level frozen diag shared across seeds.
    candidate = Path("results") / "dmcontrol_pixels" / "diag" / task / "dz_diag.pt"
    if suite_name.startswith("dmcontrol") and candidate.is_file():
        return candidate
    return run_dir / "dz_diag.pt"


def _ensure_dz_diag(
    run_dir: Path,
    suite_name: str,
    task: str,
    obs: torch.Tensor,
    n_pairs: int,
) -> dict:
    """Load frozen diag batch, or freeze the current obs buffer once."""
    path = _resolve_dz_diag_path(run_dir, suite_name, task)
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=False)
    from src.losses.dz_trust_region import sample_pair_index_mask

    obs_cpu = obs.detach().float().cpu()
    pair_i, pair_j = sample_pair_index_mask(
        obs_cpu.shape[0], n_pairs, lambda_loc=0.0, device=torch.device("cpu")
    )
    payload = {
        "obs": obs_cpu,
        "pair_i": pair_i.cpu(),
        "pair_j": pair_j.cpu(),
        "suite": suite_name,
        "task": task,
        "n_obs": int(obs_cpu.shape[0]),
        "n_pairs": int(pair_i.shape[0]),
    }
    if suite_name == "ale":
        # Fixed random linear map for cumulant probe: targets = phi @ W (Moalla-style).
        gen = torch.Generator()
        gen.manual_seed(0)
        # NatureCNN flat_dim for 4x84x84 is 3136; store oversized-safe W rows.
        feature_dim = 3136
        n_targets = 8
        payload["cumulant_W"] = torch.randn(feature_dim, n_targets, generator=gen)
        payload["cumulant_feature_dim"] = feature_dim
        payload["cumulant_n_targets"] = n_targets

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    # Also mirror into shared diag location for later seeds when created from a run.
    if suite_name == "ale":
        shared = Path("results") / "ale" / "diag" / _ale_task_short(task) / "diag.pt"
        if not shared.is_file():
            shared.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, shared)
    else:
        shared = Path("results") / "dmcontrol_pixels" / "diag" / task / "dz_diag.pt"
        if suite_name.startswith("dmcontrol") and not shared.is_file():
            shared.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, shared)
    return payload


@torch.no_grad()
def _eval_actor_ale_metrics(
    policy: nn.Module,
    obs: torch.Tensor,
    diag: dict,
    device: str,
    prefix: str,
) -> dict[str, float]:
    from src.metrics.actor_geometry import actor_geometry_metrics
    from src.metrics.cumulant_probe import random_cumulant_probe
    from src.losses.dz_trust_region import compute_s_ref

    # Cap batch for geometry SVD under tight login memory cgroups.
    max_n = 256
    obs_use = obs[:max_n] if obs.shape[0] > max_n else obs
    phi = policy.actor_preactivation(obs_use.to(device)).detach()
    out: dict[str, float] = {}
    use_diag_pairs = obs_use.shape[0] == diag["obs"].shape[0] and obs.shape[0] == diag["obs"].shape[0]
    if use_diag_pairs:
        pair_i = diag["pair_i"].to(device)
        pair_j = diag["pair_j"].to(device)
    else:
        from src.losses.dz_trust_region import sample_pair_index_mask

        pair_i, pair_j = sample_pair_index_mask(
            phi.shape[0],
            min(512, max(1, phi.shape[0] * (phi.shape[0] - 1) // 2)),
            lambda_loc=0.0,
            device=phi.device,
        )
    try:
        s_ref = compute_s_ref(phi, pair_i, pair_j)
    except RuntimeError as exc:
        if "degenerate_at_initialization" not in str(exc):
            raise
        s_ref = 0.0
        out[f"{prefix}_s_ref_degenerate"] = 1.0
    geo = actor_geometry_metrics(
        phi,
        pair_i=pair_i,
        pair_j=pair_j,
        s_ref=float(s_ref) if s_ref else None,
    )
    for k, v in geo.items():
        out[f"{prefix}_{k}"] = v
    if "cumulant_W" in diag:
        w = diag["cumulant_W"]
        d = phi.shape[1]
        if w.shape[0] < d:
            raise RuntimeError(f"cumulant_W rows {w.shape[0]} < phi dim {d}")
        targets = (phi.cpu().float() @ w[:d].float())
        probe = random_cumulant_probe(phi.cpu(), targets)
        for k, v in probe.items():
            out[f"{prefix}_{k}"] = v
    return out


@torch.no_grad()
def _eval_dz_diag_metrics(
    agent: CTRO | PPO,
    critic: nn.Module,
    diag: dict,
    device: str,
    mico_embed_ball_radius: float | None,
    s_ref_store: dict,
) -> dict[str, float]:
    """C_0.01 and related diagnostics on a frozen observation/pair set.

    Works for PPO and CTRO. If the agent has no frozen s_ref (plain PPO), compute
    s_ref once from the diag batch and reuse it for the run.
    """
    obs = diag["obs"].to(device)
    pair_i = diag["pair_i"].to(device)
    pair_j = diag["pair_j"].to(device)
    z = encode_phi(
        critic,
        obs,
        repr_net=None,
        embed_ball_radius=mico_embed_ball_radius,
    )
    s_ref = getattr(agent, "s_ref", None)
    if s_ref is None:
        if "s_ref" not in s_ref_store:
            try:
                s_ref_store["s_ref"] = compute_s_ref(z.detach(), pair_i, pair_j)
            except RuntimeError as exc:
                if "degenerate_at_initialization" not in str(exc):
                    raise
                # Total latent collapse: cannot form a scale. Record saturated
                # collapse diagnostics and continue training instead of aborting.
                return {
                    "diag_dz_C_0p01": 1.0,
                    "diag_dz_C_0p001": 1.0,
                    "diag_dz_C_0p1": 1.0,
                    "diag_dz_d_bar_old_median": 0.0,
                    "diag_dz_s_ref": 0.0,
                    "diag_dormant_unit_frac": dormant_unit_fraction(z),
                    "diag_C_0p01": 1.0,
                    "diag_s_ref_degenerate": 1.0,
                }
        s_ref = s_ref_store["s_ref"]
    delta = float(getattr(agent, "dz_delta", 0.01))
    # Same encoder for old/new → D_Z≈0; C_0.01 still measures absolute collapse vs s_ref.
    _, stats = compute_dz(
        z,
        z.detach(),
        pair_i,
        pair_j,
        s_ref=float(s_ref),
        delta=delta,
    )
    return {
        "diag_dz_C_0p01": float(stats["dz_C_0p01"].item()),
        "diag_dz_C_0p001": float(stats["dz_C_0p001"].item()),
        "diag_dz_C_0p1": float(stats["dz_C_0p1"].item()),
        "diag_dz_d_bar_old_median": float(stats["dz_d_bar_old_median"].item()),
        "diag_dz_s_ref": float(s_ref),
        "diag_dormant_unit_frac": dormant_unit_fraction(z),
        "diag_C_0p01": near_zero_pair_rate(z, pair_i, pair_j, float(s_ref), 0.01),
    }


def _eval_functional_probe(
    critic: nn.Module,
    obs: torch.Tensor,
    returns: torch.Tensor,
    mico_embed_ball_radius: float | None,
) -> dict[str, float]:
    """Freeze encoder; ridge-fit linear probe to on-policy returns; held-out MSE."""
    with torch.no_grad():
        z = encode_phi(
            critic,
            obs,
            repr_net=None,
            embed_ball_radius=mico_embed_ball_radius,
        )
    mse = functional_value_probe_mse(z, returns)
    return {
        "functional_probe_mse": mse,
        "dormant_unit_frac": dormant_unit_fraction(z),
    }

def _save_f_gap_hist(
    run_dir: Path,
    critic: nn.Module,
    obs: torch.Tensor,
    v_ref: float,
    step: int,
    epoch: int,
    mico_embed_ball_radius: float | None,
    repr_net: nn.Module | None = None,
) -> None:
    """Per-log histogram of raw f=V̂_q−V for post-hoc τ sweeps (Run 4)."""
    with torch.no_grad():
        z = encode_phi(
            critic,
            obs,
            repr_net=repr_net,
            embed_ball_radius=mico_embed_ball_radius,
        )
        v = critic.value_head(z).squeeze(-1)
        f_raw = float(v_ref) - v
        hist = value_gap_histogram(f_raw)
    out = run_dir / "f_gap_hist"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"epoch_{epoch:06d}_step_{step}.npz"
    np.savez(
        path,
        counts=hist["counts"],
        bin_edges=hist["bin_edges"],
        n=hist["n"],
        p05=hist["p05"],
        p50=hist["p50"],
        p95=hist["p95"],
        mean=hist["mean"],
        min=hist["min"],
        max=hist["max"],
        frac_positive=hist["frac_positive"],
        v_ref=float(v_ref),
        epoch=epoch,
        step=step,
    )


def _save_pairwise_distance_hist(
    run_dir: Path,
    critic: nn.Module,
    obs: torch.Tensor,
    step: int,
    epoch: int,
    mico_embed_ball_radius: float | None,
    repr_net: nn.Module | None = None,
) -> None:
    """Checkpoint pairwise latent-distance histogram (degeneracy diagnostic)."""
    with torch.no_grad():
        z = encode_phi(
            critic,
            obs,
            repr_net=repr_net,
            embed_ball_radius=mico_embed_ball_radius,
        )
        hist = pairwise_distance_histogram(z)
    out = run_dir / "pairwise_distance_hist"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"epoch_{epoch:06d}_step_{step}.npz"
    np.savez(
        path,
        counts=hist["counts"],
        bin_edges=hist["bin_edges"],
        n_pairs=hist["n_pairs"],
        p05=hist["p05"],
        p50=hist["p50"],
        p95=hist["p95"],
        mean=hist["mean"],
        max=hist["max"],
        epoch=epoch,
        step=step,
    )
    print(
        f"Pairwise dist hist epoch={epoch} p05={hist['p05']:.4f} "
        f"p50={hist['p50']:.4f} p95={hist['p95']:.4f} -> {path}",
        flush=True,
    )


class RunAborted(Exception):
    """Training stopped early (failed or pruned). Does not write weights_final.pt."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass
class TrainResult:
    run_dir: Path
    status: str
    reason: str
    eval_full_return_mean: float | None
    total_steps: int


def make_train_env(suite_name: str, task: str):
    suite = EVAL_SUITES[suite_name]
    if suite.env_type == "procgen":
        train_dist = DistributionSpec(
            name="train",
            procgen_num_levels=suite.train_num_levels,
            procgen_start_level=0,
        )
        return make_env(suite, task, train_dist)
    train_dist = DistributionSpec(name="train", dmcontrol_seed_offset=0)
    return make_env(suite, task, train_dist)


def _obs_norm_shape(env) -> tuple[int, ...]:
    if env.obs_shape is not None:
        return tuple(env.obs_shape)
    return (env.obs_dim,)


def _select_action_value(
    stack: PerformanceStack,
    normalizer: PerformanceNormalizer,
    obs: torch.Tensor,
    device: str,
    mico_embed_ball_radius: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_tensor = obs.unsqueeze(0).to(device)
    norm_obs = normalizer.observe(obs_tensor).squeeze(0)
    norm_batch = norm_obs.unsqueeze(0)
    with torch.no_grad():
        if stack.policy_on_latent:
            phi = encode_phi(
                stack.critic,
                norm_batch,
                embed_ball_radius=mico_embed_ball_radius,
            )
            action, log_prob = stack.policy.get_action(phi.squeeze(0))
            value = stack.critic(norm_batch).squeeze(-1).squeeze(0)
        else:
            action, log_prob = stack.policy.get_action(norm_obs)
            value = stack.critic(norm_batch).squeeze(-1).squeeze(0)
    return norm_obs.cpu(), action.cpu(), log_prob.cpu(), value.cpu()


def collect_rollout_buffer(
    env,
    stack: PerformanceStack,
    normalizer: PerformanceNormalizer,
    buffer_size: int,
    device: str,
    frame_recorder: BestEpisodeFrameRecorder | None = None,
    mico_embed_ball_radius: float | None = None,
) -> dict:
    buffer = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "terminations": [],
        "truncations": [],
        "log_probs": [],
        "values": [],
        "next_obs": [],
    }
    episode_returns = []
    current_return = 0.0

    while len(buffer["obs"]) < buffer_size:
        obs, _ = env.reset()
        normalizer.reward_norm.reset_episode()
        if frame_recorder is not None:
            frame_recorder.start_episode()
            frame_recorder.append_frame(obs)
        done = False

        while not done and len(buffer["obs"]) < buffer_size:
            norm_obs, action, log_prob, value = _select_action_value(
                stack,
                normalizer,
                obs,
                device,
                mico_embed_ball_radius=mico_embed_ball_radius,
            )

            step_action = action.item() if env.action_space_type == "discrete" else action
            next_obs, reward, terminated, truncated, _ = env.step(step_action)
            done = terminated or truncated
            norm_reward = normalizer.reward(reward, done)

            buffer["obs"].append(norm_obs)
            buffer["actions"].append(action)
            buffer["rewards"].append(norm_reward)
            buffer["dones"].append(done)
            buffer["terminations"].append(terminated)
            buffer["truncations"].append(truncated)
            buffer["log_probs"].append(log_prob)
            buffer["values"].append(value)

            next_norm_obs = normalizer.observe(next_obs.unsqueeze(0).to(device)).squeeze(0).cpu()
            buffer["next_obs"].append(next_norm_obs)

            current_return += reward
            obs = next_obs
            if frame_recorder is not None:
                frame_recorder.add_reward(reward)
                frame_recorder.append_frame(obs)

            if done:
                episode_returns.append(current_return)
                if frame_recorder is not None:
                    frame_recorder.finish_episode()
                current_return = 0.0

    n = min(len(buffer["obs"]), buffer_size)
    return {
        "obs": torch.stack(buffer["obs"][:n]),
        "actions": torch.stack(buffer["actions"][:n]),
        "rewards": torch.tensor(buffer["rewards"][:n], dtype=torch.float32),
        "dones": torch.tensor(buffer["dones"][:n], dtype=torch.bool),
        "terminations": torch.tensor(buffer["terminations"][:n], dtype=torch.bool),
        "truncations": torch.tensor(buffer["truncations"][:n], dtype=torch.bool),
        "log_probs": torch.stack(buffer["log_probs"][:n]),
        "values": torch.stack(buffer["values"][:n]),
        "next_obs": torch.stack(buffer["next_obs"][:n]),
        "episode_returns": episode_returns,
    }


def _select_action_value_batch(
    stack: PerformanceStack,
    norm_obs: torch.Tensor,
    mico_embed_ball_radius: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched action/log_prob/value for [N, *obs] normalized observations."""
    with torch.no_grad():
        if stack.policy_on_latent:
            phi = encode_phi(stack.critic, norm_obs, embed_ball_radius=mico_embed_ball_radius)
            action, log_prob = stack.policy.get_action(phi)
        else:
            action, log_prob = stack.policy.get_action(norm_obs)
        value = stack.critic(norm_obs).squeeze(-1)
    return action, log_prob, value


def compute_gae_vec(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-env GAE over [T, N] with V(s_{t+1}) provided as `next_values`.

    Bootstrap unless `terminations[t]`. GAE carry resets on terminations|truncations.
    """
    T = rewards.shape[0]
    dones = terminations | truncations
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_values[0])
    for t in reversed(range(T)):
        non_terminal = (~terminations[t]).float()
        non_done = (~dones[t]).float()
        delta = rewards[t] + gamma * next_values[t] * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def collect_rollout_buffer_vec(
    vec_env,
    stack: PerformanceStack,
    normalizer: PerformanceNormalizer,
    buffer_size: int,
    device: str,
    gamma: float,
    gae_lambda: float,
    mico_embed_ball_radius: float | None = None,
) -> dict:
    """Vectorized rollout: step N envs in lockstep, batch the forward pass.

    Collects ceil(buffer_size / N) steps per env, computes per-env GAE with a
    value bootstrap for envs still running at the horizon, then flattens the
    [T, N] transitions to a flat [T*N] batch for the PPO/CTRO update.
    """
    num_envs = vec_env.num_envs
    T = -(-buffer_size // num_envs)
    is_discrete = vec_env.action_space_type == "discrete"

    steps = {
        k: []
        for k in (
            "obs",
            "actions",
            "rewards",
            "dones",
            "terminations",
            "truncations",
            "log_probs",
            "values",
            "next_obs",
        )
    }
    episode_returns: list[float] = []
    running_return = np.zeros(num_envs, dtype=np.float64)
    reward_return_acc = np.zeros(num_envs, dtype=np.float64)

    obs = vec_env.reset()
    for _ in range(T):
        norm_obs = normalizer.observe(obs.to(device))
        action, log_prob, value = _select_action_value_batch(
            stack, norm_obs, mico_embed_ball_radius=mico_embed_ball_radius
        )

        actions_np = action.cpu().numpy()
        step_actions = actions_np.astype(np.int32) if is_discrete else actions_np.astype(np.float32)
        result = vec_env.step(step_actions)

        raw_rewards = result.rewards
        dones = result.dones
        terminations = result.terminations
        truncations = result.truncations
        norm_rewards, reward_return_acc = normalizer.reward_norm.normalize_batch(
            raw_rewards, dones, reward_return_acc
        )
        norm_next_obs = normalizer.normalize_obs(result.next_obs.to(device))

        steps["obs"].append(norm_obs.cpu())
        steps["actions"].append(action.cpu())
        steps["rewards"].append(torch.from_numpy(norm_rewards))
        steps["dones"].append(torch.from_numpy(dones))
        steps["terminations"].append(torch.from_numpy(terminations))
        steps["truncations"].append(torch.from_numpy(truncations))
        steps["log_probs"].append(log_prob.cpu())
        steps["values"].append(value.cpu())
        steps["next_obs"].append(norm_next_obs.cpu())

        running_return += raw_rewards
        for i in range(num_envs):
            if dones[i]:
                episode_returns.append(float(running_return[i]))
                running_return[i] = 0.0

        obs = result.obs

    next_obs_t = torch.stack(steps["next_obs"])
    with torch.no_grad():
        # V(s_{t+1}) from stored bootstrap obs (terminal state on truncate/terminate).
        flat_next = next_obs_t.reshape(T * num_envs, *next_obs_t.shape[2:]).to(device)
        next_values = (
            stack.critic(flat_next).squeeze(-1).cpu().reshape(T, num_envs)
        )

    rewards = torch.stack(steps["rewards"])
    values = torch.stack(steps["values"])
    terminations_t = torch.stack(steps["terminations"])
    truncations_t = torch.stack(steps["truncations"])
    advantages, returns = compute_gae_vec(
        rewards,
        values,
        terminations_t,
        truncations_t,
        next_values,
        gamma,
        gae_lambda,
    )

    def flatten(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])

    return {
        "obs": flatten(torch.stack(steps["obs"])),
        "actions": flatten(torch.stack(steps["actions"])),
        "rewards": flatten(rewards),
        "dones": flatten(torch.stack(steps["dones"])),
        "terminations": flatten(terminations_t),
        "truncations": flatten(truncations_t),
        "log_probs": flatten(torch.stack(steps["log_probs"])),
        "values": flatten(values),
        "next_obs": flatten(torch.stack(steps["next_obs"])),
        "advantages": flatten(advantages),
        "returns": flatten(returns),
        "episode_returns": episode_returns,
    }


def _write_run_status(run_dir: Path, status: str, reason: str, total_steps: int) -> None:
    payload = {"status": status, "reason": reason, "total_steps": total_steps}
    (run_dir / "run_status.json").write_text(json.dumps(payload, indent=2) + "\n")


def _infer_resume_from_metrics(
    run_dir: Path, buffer_size: int, checkpoint_freq: int
) -> tuple[int, int, int]:
    """Fallback for checkpoints saved before training_epoch was persisted.

    Aligns to the last weights_latest save (epoch divisible by checkpoint_freq).
    Returns (completed_epoch, total_env_steps, lr_update_count).
    """
    import pandas as pd

    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"resume requires training_epoch in weights_latest.pt or {metrics_path}"
        )
    df = pd.read_csv(metrics_path)
    if "epoch" not in df.columns:
        raise RuntimeError(f"{metrics_path} missing epoch column")
    max_epoch = int(df["epoch"].max())
    ckpt_epoch = (max_epoch // checkpoint_freq) * checkpoint_freq
    if ckpt_epoch < 1:
        raise RuntimeError(
            f"resume metrics fallback: max_epoch={max_epoch} < checkpoint_freq={checkpoint_freq}"
        )
    total_env_steps = ckpt_epoch * buffer_size
    return ckpt_epoch, total_env_steps, ckpt_epoch


def _resolve_resume_state(
    run_dir: Path, buffer_size: int, checkpoint_freq: int
) -> dict:
    ckpt_path = run_dir / "weights_latest.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"resume requested but missing {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "training_epoch" in ckpt:
        completed_epoch = int(ckpt["training_epoch"])
        total_env_steps = int(ckpt["total_env_steps"])
        lr_update_count = int(ckpt.get("lr_update_count", completed_epoch))
    else:
        completed_epoch, total_env_steps, lr_update_count = _infer_resume_from_metrics(
            run_dir, buffer_size, checkpoint_freq
        )
        print(
            f"Resume fallback from metrics.csv: completed_epoch={completed_epoch} "
            f"total_env_steps={total_env_steps} (weights_latest lacks training_epoch)",
            flush=True,
        )
    return {
        "path": ckpt_path,
        "completed_epoch": completed_epoch,
        "start_epoch": completed_epoch + 1,
        "total_env_steps": total_env_steps,
        "lr_update_count": lr_update_count,
    }


def _apply_arch_overrides(arch_cfg: dict, arch_overrides: dict | None) -> dict:
    arch = copy.deepcopy(arch_cfg)
    if not arch_overrides:
        return arch
    if "policy_hidden" in arch_overrides:
        arch["policy"] = {**arch["policy"], "hidden_sizes": list(arch_overrides["policy_hidden"])}
    if "policy" in arch_overrides:
        arch["policy"] = {**arch["policy"], **arch_overrides["policy"]}
    if "critic" in arch_overrides:
        arch["critic"] = {**arch["critic"], **arch_overrides["critic"]}
    return arch


def _resolve_train_cfg(train_cfg: dict, train_overrides: dict | None) -> dict:
    cfg = {**train_cfg, **(train_overrides or {})}
    if "total_steps" in cfg:
        buffer_size = cfg["buffer_size"]
        cfg["total_epochs"] = int(cfg["total_steps"]) // int(buffer_size)
        if cfg["total_epochs"] < 1:
            raise ValueError(
                f"total_steps={cfg['total_steps']} < buffer_size={buffer_size}"
            )
    return cfg


def run_performance_train(
    suite_name: str,
    task: str,
    seed: int,
    exp_name: str = "exp_full",
    agent_cls: type = CTRO,
    algo_overrides: dict | None = None,
    arch_overrides: dict | None = None,
    train_overrides: dict | None = None,
    results_root: str | Path = "results",
    device: str | None = None,
    num_envs: int | None = None,
    report_callback: Callable[[int, dict], bool] | None = None,
    collapse_floor: float | None = None,
    init_weights: str | Path | None = None,
    resume: bool = False,
) -> TrainResult:
    """Train one performance-suite task.

    `report_callback(total_steps, metrics) -> should_prune`. When True, training
    aborts with RunAborted(status="pruned") and does not write weights_final.pt.
    """
    suite = EVAL_SUITES[suite_name]
    suite_cfg = PERFORMANCE_SUITE_CONFIG[suite_name]
    algo_key = "ctro_algo" if agent_cls is CTRO else "ppo_algo"
    algo_cfg = {**BASE_ALGO_CONFIG, **suite_cfg[algo_key], **(algo_overrides or {})}
    arch_cfg = _apply_arch_overrides(suite_cfg["arch"], arch_overrides)
    train_cfg = _resolve_train_cfg(suite_cfg["training"], train_overrides)
    if num_envs is not None:
        train_cfg = {**train_cfg, "num_envs": num_envs}

    freeze_id = os.environ.get("FREEZE_ID")
    freeze_payload = None
    if freeze_id and bool(algo_cfg.get("dz_enabled", False)):
        from src.experiments.freeze_load import apply_ltro_freeze_overrides, load_freeze_file, freeze_path

        freeze_payload = load_freeze_file(freeze_path(freeze_id))
        algo_cfg = apply_ltro_freeze_overrides(algo_cfg, freeze_payload)

    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    prefix = suite_cfg["results_prefix"]
    run_dir = Path(results_root) / prefix / exp_name / f"seed_{seed}" / task
    run_dir.mkdir(parents=True, exist_ok=True)

    buffer_size = train_cfg["buffer_size"]
    checkpoint_freq = train_cfg["checkpoint_frequency"]

    if resume and init_weights is not None:
        raise ValueError("resume and init_weights are mutually exclusive")
    if resume and (run_dir / "weights_final.pt").is_file():
        raise RuntimeError(f"resume requested but run already finished: {run_dir / 'weights_final.pt'}")

    resume_state = None
    if resume:
        resume_state = _resolve_resume_state(run_dir, buffer_size, checkpoint_freq)

    num_envs = train_cfg.get("num_envs", 1)
    if num_envs > 1:
        env = make_train_vec_env(suite_name, task, num_envs, base_seed=seed * 1000)
        frame_recorder = None
    else:
        env = make_train_env(suite_name, task)
        frame_recorder = make_best_episode_frame_recorder(env)
    stack = build_performance_stack(arch_cfg, env, agent_cls, device)

    config_path = run_dir / "config.json"
    if resume and config_path.is_file():
        prev_cfg = json.loads(config_path.read_text())
        normalizer = PerformanceNormalizer.from_state_dict(prev_cfg["normalization"])
    else:
        normalizer = PerformanceNormalizer(
            _obs_norm_shape(env),
            algo_cfg["gamma"],
            obs_norm=train_cfg.get("obs_norm", "running_mean_std"),
            obs_norm_clip=float(train_cfg.get("obs_norm_clip", 10.0)),
            reward_norm_mode=train_cfg.get("reward_norm", "return_var_scale"),
        )
    algo_cfg = {
        **algo_cfg,
        "policy_on_latent": stack.policy_on_latent,
        "total_update_epochs": int(train_cfg["total_epochs"]),
    }
    agent = agent_cls(stack.policy, stack.critic, algo_cfg, device=device)
    if resume_state is not None:
        agent.load(str(resume_state["path"]))
        agent._lr_update_count = resume_state["lr_update_count"]
        truncate_metrics_csv(run_dir / "metrics.csv", resume_state["completed_epoch"])
        print(
            f"Resuming from epoch {resume_state['start_epoch']} "
            f"({resume_state['total_env_steps']} env steps) "
            f"<- {resume_state['path']}",
            flush=True,
        )
    elif init_weights is not None:
        init_path = Path(init_weights)
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        agent.policy.load_state_dict(ckpt["policy"])
        agent.critic.load_state_dict(ckpt["critic"])
        if "v_ref" in ckpt and ckpt["v_ref"] is not None:
            agent.v_ref = ckpt["v_ref"]
        print(f"Loaded init weights from {init_path}", flush=True)

    full_config = {
        "experiment": exp_name,
        "suite": suite_name,
        "task": task,
        "seed": seed,
        "algorithm": algo_cfg,
        "architecture": arch_cfg,
        "training": train_cfg,
        "agent_class": agent_cls.__name__,
        "stack_type": stack.stack_type,
        "policy_on_latent": stack.policy_on_latent,
        "pixel_obs": stack.pixel_obs,
        "obs_shape": list(env.obs_shape) if env.obs_shape is not None else None,
        "normalization": normalizer.state_dict(),
    }
    if freeze_id:
        full_config["freeze_id"] = freeze_id
    if freeze_payload is not None:
        full_config["freeze_git_tag"] = freeze_payload.get("git_tag")
        full_config["freeze_selection"] = freeze_payload.get("selection")
    if suite_name == "ale":
        import gymnasium
        import ale_py

        from src.experiments.freeze_load import protocol_hash

        full_config["ale_py_version"] = ale_py.__version__
        full_config["gymnasium_version"] = gymnasium.__version__
        full_config["protocol_hash"] = protocol_hash()
        full_config["frame_skip_note"] = (
            "AtariPreprocessing frame_skip=4 (CleanRL/SB3); Moalla torchrl used 3"
        )

    logger = CSVLogger(run_dir, "", clear_existing=not resume)
    logger.save_config(full_config)

    metric_eval = CTROMetricEvaluator(
        gamma=algo_cfg["gamma"],
        mu_pl_max_samples=train_cfg.get("metric_pl_max_samples"),
        mico_embed_ball_radius=algo_cfg.get("mico_embed_ball_radius"),
        tau_ref=algo_cfg.get("tau_ref", 0.01),
        f_floor=algo_cfg.get("f_floor", 1e-3),
        v_ref_quantile=algo_cfg.get("v_ref_quantile", 0.99),
    )
    if agent.v_ref is not None:
        metric_eval.v_ref = agent.v_ref
    total_epochs = train_cfg["total_epochs"]
    log_interval = train_cfg["log_interval_steps"]
    eval_freq = train_cfg["eval_frequency"]
    eval_episodes = train_cfg["eval_episodes"]
    print_every = train_cfg.get("print_every_epochs", 50)
    _cms = train_cfg.get("collapse_min_steps", 200_000)
    collapse_min_steps = 0 if _cms is None else int(_cms)
    collapse_streak_limit = int(train_cfg.get("collapse_streak", 3))
    # Explicit train_cfg key wins (including None to disable). Else DMC suite defaults.
    if "collapse_floor" in train_cfg:
        collapse_floor = train_cfg["collapse_floor"]
    elif collapse_floor is None and suite_name in ("dmcontrol_state", "dmcontrol_pixels"):
        # May be None for tasks that disable collapse (e.g. hopper-hop).
        collapse_floor = DMCONTROL_COLLAPSE_FLOORS[task]

    early_stop_enabled = bool(train_cfg.get("early_stop_enabled", False))
    early_stop_min_steps = int(train_cfg.get("early_stop_min_steps", 1_000_000))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 25))
    early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 1.0))
    early_stop_metric = str(train_cfg.get("early_stop_metric", "eval_full_return_mean"))
    if early_stop_patience < 1:
        raise ValueError(f"early_stop_patience must be >= 1, got {early_stop_patience}")

    total_steps = 0 if resume_state is None else resume_state["total_env_steps"]
    start_epoch = 1 if resume_state is None else resume_state["start_epoch"]
    last_logged_step = total_steps - log_interval
    collapse_streak = 0
    early_stop_best: float | None = None
    early_stop_stale = 0
    final_eval_return: float | None = None
    status = "ok"
    reason = ""
    diag_s_ref_store: dict = {}

    print(
        f"Starting {suite_name}/{task} seed={seed} exp={exp_name} "
        f"agent={agent_cls.__name__} stack={stack.stack_type} device={device} "
        f"epochs={start_epoch}-{total_epochs} obs_norm={normalizer.obs_norm} "
        f"reward_norm={normalizer.reward_norm_mode} "
        f"early_stop={early_stop_enabled} resume={resume}",
        flush=True,
    )

    try:
        for epoch in range(start_epoch, total_epochs + 1):
            if num_envs > 1:
                buffer = collect_rollout_buffer_vec(
                    env,
                    stack,
                    normalizer,
                    buffer_size,
                    device,
                    gamma=algo_cfg["gamma"],
                    gae_lambda=algo_cfg["gae_lambda"],
                    mico_embed_ball_radius=algo_cfg.get("mico_embed_ball_radius"),
                )
            else:
                buffer = collect_rollout_buffer(
                    env,
                    stack,
                    normalizer,
                    buffer_size,
                    device,
                    frame_recorder=frame_recorder,
                    mico_embed_ball_radius=algo_cfg.get("mico_embed_ball_radius"),
                )
            for key in ("obs", "actions", "rewards", "dones", "log_probs", "values", "next_obs"):
                buffer[key] = buffer[key].to(device)

            total_steps += len(buffer["obs"])

            if num_envs > 1:
                advantages = buffer["advantages"].to(device)
                returns = buffer["returns"].to(device)
            else:
                with torch.no_grad():
                    next_values = stack.critic(buffer["next_obs"]).squeeze(-1)
                advantages, returns = agent.compute_gae(
                    buffer["rewards"],
                    buffer["values"],
                    buffer["dones"],
                    terminations=buffer["terminations"],
                    next_values=next_values,
                )

            update_kwargs = dict(
                obs=buffer["obs"],
                actions=buffer["actions"],
                old_log_probs=buffer["log_probs"],
                advantages=advantages,
                returns=returns,
                training_epoch=epoch,
            )
            if agent.needs_transition_batch:
                update_kwargs["rewards"] = buffer["rewards"]
                update_kwargs["next_obs"] = buffer["next_obs"]

            update_stats = agent.update(**update_kwargs)

            if not math.isfinite(update_stats["policy_loss"]) or not math.isfinite(
                update_stats["value_loss"]
            ):
                raise RunAborted(
                    "failed",
                    f"non-finite losses policy={update_stats['policy_loss']} "
                    f"value={update_stats['value_loss']}",
                )

            # Only use completed episodes. Empty windows are common with short
            # rollouts (e.g. T=512/env on 1000-step cartpole) — do not treat as 0.
            n_ep = len(buffer["episode_returns"])
            mean_ret = (
                float(np.mean(buffer["episode_returns"])) if n_ep > 0 else float("nan")
            )

            should_eval = epoch % eval_freq == 0 or epoch == total_epochs
            should_log = (
                total_steps - last_logged_step >= log_interval
                or epoch == total_epochs
                or should_eval
            )
            metrics: dict = {}
            if should_log:
                metrics = {
                    "epoch": epoch,
                    "mean_episode_return": mean_ret,
                    "n_completed_episodes": float(n_ep),
                    **update_stats,
                }
                if suite_name != "ale":
                    metrics.update(
                        metric_eval.evaluate(
                            stack.critic,
                            buffer["obs"],
                            buffer["next_obs"],
                            buffer["rewards"],
                            v_ref=agent.v_ref,
                        )
                    )
                    _save_f_gap_hist(
                        run_dir,
                        stack.critic,
                        buffer["obs"],
                        float(metrics["v_ref"]),
                        total_steps,
                        epoch,
                        algo_cfg.get("mico_embed_ball_radius"),
                    )
                    # T1.4: first-class pairwise series every metrics dump (hist still at ckpts).
                    with torch.no_grad():
                        z_log = encode_phi(
                            stack.critic,
                            buffer["obs"],
                            repr_net=None,
                            embed_ball_radius=algo_cfg.get("mico_embed_ball_radius"),
                        )
                        pair_hist = pairwise_distance_histogram(z_log)
                    metrics["latent_pair_p05"] = float(pair_hist["p05"])
                    metrics["latent_pair_p50"] = float(pair_hist["p50"])
                    metrics["latent_pair_p95"] = float(pair_hist["p95"])
                    metrics["latent_pair_mean"] = float(pair_hist["mean"])
                else:
                    # ALE: skip PL-grad evaluator (login cgroup); keep light critic rank only.
                    with torch.no_grad():
                        z_log = encode_phi(
                            stack.critic,
                            buffer["obs"][: min(128, buffer["obs"].shape[0])],
                            repr_net=None,
                            embed_ball_radius=algo_cfg.get("mico_embed_ball_radius"),
                        )
                    from src.metrics.feature_rank import compute_feature_rank_metrics

                    metrics.update(
                        {f"critic_{k}": v for k, v in compute_feature_rank_metrics(z_log).items()}
                    )
                    if agent.v_ref is not None:
                        metrics["v_ref"] = float(agent.v_ref)

                if should_eval:
                    for distribution in suite.distributions:
                        eval_env = make_env(suite, task, distribution)
                        seed_offset = distribution.dmcontrol_seed_offset or 0
                        result = run_eval_episodes(
                            eval_env,
                            stack,
                            normalizer,
                            device,
                            eval_episodes,
                            deterministic=train_cfg["eval_deterministic"],
                            dmcontrol_seed_offset=seed_offset,
                            mico_embed_ball_radius=algo_cfg.get("mico_embed_ball_radius"),
                        )
                        eval_env.close()
                        metrics[f"eval_{distribution.name}_return_mean"] = result["return_mean"]
                        metrics[f"eval_{distribution.name}_return_std"] = result["return_std"]
                    final_eval_return = metrics.get("eval_full_return_mean", final_eval_return)
                    diag = _ensure_dz_diag(
                        run_dir,
                        suite_name,
                        task,
                        buffer["obs"],
                        int(algo_cfg.get("dz_n_pairs", 2048 if suite_name != "ale" else 256)),
                    )
                    if suite_name == "ale":
                        metrics.update(
                            _eval_actor_ale_metrics(
                                stack.policy,
                                buffer["obs"],
                                diag,
                                device,
                                prefix="onpolicy",
                            )
                        )
                        metrics.update(
                            _eval_actor_ale_metrics(
                                stack.policy,
                                diag["obs"],
                                diag,
                                device,
                                prefix="diag",
                            )
                        )
                    else:
                        metrics.update(
                            _eval_functional_probe(
                                stack.critic,
                                buffer["obs"],
                                returns,
                                algo_cfg.get("mico_embed_ball_radius"),
                            )
                        )
                        metrics.update(
                            _eval_dz_diag_metrics(
                                agent,
                                stack.critic,
                                diag,
                                device,
                                algo_cfg.get("mico_embed_ball_radius"),
                                diag_s_ref_store,
                            )
                        )

                logger.log_metrics(total_steps, metrics)
                last_logged_step = total_steps

                # Collapse prune only when we observed finished eps this window.
                if (
                    collapse_floor is not None
                    and total_steps >= collapse_min_steps
                    and n_ep > 0
                    and mean_ret < collapse_floor
                ):
                    collapse_streak += 1
                    if collapse_streak >= collapse_streak_limit:
                        raise RunAborted(
                            "pruned",
                            f"return collapse mean_ret={mean_ret:.4f} < floor={collapse_floor} "
                            f"for {collapse_streak} log intervals after {total_steps} steps",
                        )
                elif n_ep > 0:
                    collapse_streak = 0

                if report_callback is not None and metrics:
                    if report_callback(total_steps, metrics):
                        raise RunAborted("pruned", "optuna median pruner")

                # Plateau early-stop: success path (still writes weights_final).
                # Only score when the chosen metric is present (typically after eval).
                if early_stop_enabled and metrics and early_stop_metric in metrics:
                    score = metrics[early_stop_metric]
                    if score is not None and math.isfinite(float(score)):
                        score_f = float(score)
                        if early_stop_best is None or score_f > early_stop_best + early_stop_min_delta:
                            early_stop_best = score_f
                            early_stop_stale = 0
                        else:
                            early_stop_stale += 1
                        if (
                            total_steps >= early_stop_min_steps
                            and early_stop_stale >= early_stop_patience
                        ):
                            reason = (
                                f"early_stop_plateau metric={early_stop_metric} "
                                f"best={early_stop_best:.4f} stale={early_stop_stale} "
                                f"min_delta={early_stop_min_delta} after {total_steps} steps"
                            )
                            status = "ok"
                            print(
                                f"Early stop {suite_name}/{task} seed={seed}: {reason}",
                                flush=True,
                            )
                            break

            if epoch % checkpoint_freq == 0:
                agent.save(
                    str(run_dir / "weights_latest.pt"),
                    training_epoch=epoch,
                    total_env_steps=total_steps,
                )
                _write_run_status(run_dir, "running", "", total_steps)
                if suite_name != "ale":
                    _save_pairwise_distance_hist(
                        run_dir,
                        stack.critic,
                        buffer["obs"],
                        total_steps,
                        epoch,
                        algo_cfg.get("mico_embed_ball_radius"),
                    )

            if epoch % print_every == 0:
                ret_str = f"{mean_ret:.3f}" if n_ep > 0 else "nan"
                print(
                    f"Epoch {epoch}/{total_epochs} steps={total_steps} "
                    f"return={ret_str} n_ep={n_ep}",
                    flush=True,
                )
        else:
            # No early break: finished full budget.
            reason = ""
            status = "ok"

        agent.save(
            str(run_dir / "weights_final.pt"),
            training_epoch=epoch,
            total_env_steps=total_steps,
        )
        if suite_name != "ale":
            _save_pairwise_distance_hist(
                run_dir,
                stack.critic,
                buffer["obs"],
                total_steps,
                epoch,
                algo_cfg.get("mico_embed_ball_radius"),
            )
    except RunAborted as exc:
        status = exc.status
        reason = exc.reason
        print(f"Aborted {suite_name}/{task} seed={seed}: {status} — {reason}", flush=True)
        raise
    except Exception as exc:
        status = "failed"
        reason = str(exc)
        print(f"Failed {suite_name}/{task} seed={seed}: {reason}", flush=True)
        raise
    finally:
        full_config["normalization"] = normalizer.state_dict()
        logger.save_config(full_config)
        _write_run_status(run_dir, status, reason, total_steps)
        if frame_recorder is not None and status == "ok":
            best_episode_path = run_dir / "best_episode_frames.npz"
            frame_recorder.save(best_episode_path)
            if frame_recorder.best_frames is not None:
                print(
                    f"Saved best episode (return={frame_recorder.best_return:.3f}, "
                    f"length={frame_recorder.best_frames.shape[0]}) -> {best_episode_path}",
                    flush=True,
                )
        env.close()
        logger.close()

    print(f"Finished {suite_name}/{task} seed={seed} -> {run_dir}", flush=True)
    return TrainResult(
        run_dir=run_dir,
        status=status,
        reason=reason,
        eval_full_return_mean=final_eval_return,
        total_steps=total_steps,
    )
