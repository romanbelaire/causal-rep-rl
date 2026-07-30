"""
Training runner for Procgen and DMControl performance environments.

Trains one task and writes checkpoints to the layout expected by run_performance_eval:
  results/{suite_prefix}/{exp_name}/seed_{seed}/{task}/weights_final.pt
"""

from __future__ import annotations

import copy
import json
import math
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
from src.utils.best_episode_recorder import BestEpisodeFrameRecorder, make_best_episode_frame_recorder
from src.utils.bisimulation_utils import encode_phi
from src.utils.ctro_metric_evaluator import CTROMetricEvaluator
from src.utils.logging import CSVLogger
from src.utils.normalization import PerformanceNormalizer


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

    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    prefix = suite_cfg["results_prefix"]
    run_dir = Path(results_root) / prefix / exp_name / f"seed_{seed}" / task
    run_dir.mkdir(parents=True, exist_ok=True)

    num_envs = train_cfg.get("num_envs", 1)
    if num_envs > 1:
        env = make_train_vec_env(suite_name, task, num_envs, base_seed=seed * 1000)
        frame_recorder = None
    else:
        env = make_train_env(suite_name, task)
        frame_recorder = make_best_episode_frame_recorder(env)
    stack = build_performance_stack(arch_cfg, env, agent_cls, device)
    normalizer = PerformanceNormalizer(
        _obs_norm_shape(env),
        algo_cfg["gamma"],
        obs_norm=train_cfg.get("obs_norm", "running_mean_std"),
        obs_norm_clip=float(train_cfg.get("obs_norm_clip", 10.0)),
        reward_norm_mode=train_cfg.get("reward_norm", "return_var_scale"),
    )
    algo_cfg = {**algo_cfg, "policy_on_latent": stack.policy_on_latent}
    agent = agent_cls(stack.policy, stack.critic, algo_cfg, device=device)

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

    logger = CSVLogger(run_dir, "", clear_existing=True)
    logger.save_config(full_config)

    metric_eval = CTROMetricEvaluator(
        gamma=algo_cfg["gamma"],
        mu_pl_max_samples=train_cfg.get("metric_pl_max_samples"),
        mico_embed_ball_radius=algo_cfg.get("mico_embed_ball_radius"),
    )
    buffer_size = train_cfg["buffer_size"]
    total_epochs = train_cfg["total_epochs"]
    log_interval = train_cfg["log_interval_steps"]
    eval_freq = train_cfg["eval_frequency"]
    eval_episodes = train_cfg["eval_episodes"]
    checkpoint_freq = train_cfg["checkpoint_frequency"]
    print_every = train_cfg.get("print_every_epochs", 50)
    collapse_min_steps = int(train_cfg.get("collapse_min_steps", 200_000))
    collapse_streak_limit = int(train_cfg.get("collapse_streak", 3))
    if collapse_floor is None and suite_name == "dmcontrol_state":
        # May be None for tasks that disable collapse (e.g. hopper-hop).
        collapse_floor = DMCONTROL_COLLAPSE_FLOORS[task]

    total_steps = 0
    last_logged_step = -log_interval
    collapse_streak = 0
    final_eval_return: float | None = None
    status = "ok"
    reason = ""

    print(
        f"Starting {suite_name}/{task} seed={seed} exp={exp_name} "
        f"agent={agent_cls.__name__} stack={stack.stack_type} device={device} "
        f"epochs={total_epochs} obs_norm={normalizer.obs_norm} "
        f"reward_norm={normalizer.reward_norm_mode}",
        flush=True,
    )

    try:
        for epoch in range(1, total_epochs + 1):
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

            mean_ret = (
                float(np.mean(buffer["episode_returns"]))
                if buffer["episode_returns"]
                else 0.0
            )

            should_log = total_steps - last_logged_step >= log_interval or epoch == total_epochs
            metrics: dict = {}
            if should_log:
                metrics = {
                    "epoch": epoch,
                    "mean_episode_return": mean_ret,
                    **update_stats,
                }
                metrics.update(
                    metric_eval.evaluate(
                        stack.critic,
                        buffer["obs"],
                        buffer["next_obs"],
                        buffer["rewards"],
                    )
                )

                if epoch % eval_freq == 0 or epoch == total_epochs:
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

                logger.log_metrics(total_steps, metrics)
                last_logged_step = total_steps

                if (
                    collapse_floor is not None
                    and total_steps >= collapse_min_steps
                    and mean_ret < collapse_floor
                ):
                    collapse_streak += 1
                    if collapse_streak >= collapse_streak_limit:
                        raise RunAborted(
                            "pruned",
                            f"return collapse mean_ret={mean_ret:.4f} < floor={collapse_floor} "
                            f"for {collapse_streak} log intervals after {total_steps} steps",
                        )
                else:
                    collapse_streak = 0

                if report_callback is not None and metrics:
                    if report_callback(total_steps, metrics):
                        raise RunAborted("pruned", "optuna median pruner")

            if epoch % checkpoint_freq == 0:
                agent.save(str(run_dir / "weights_latest.pt"))

            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch}/{total_epochs} steps={total_steps} return={mean_ret:.3f}",
                    flush=True,
                )

        agent.save(str(run_dir / "weights_final.pt"))
        status = "ok"
        reason = ""
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
