"""
Performance evaluation runner for Procgen and DMControl suites.

Evaluates a policy+critic checkpoint on each task and distribution (full / test),
logging per-task return means and standard deviations.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.suites import (
    DistributionSpec,
    EvalSuite,
    EVAL_SUITES,
    parse_dmcontrol_task,
)
from src.experiments.performance_models import PerformanceStack, build_performance_stack_from_config
from src.utils.bisimulation_utils import encode_phi
from src.utils.logging import CSVLogger
from src.utils.normalization import PerformanceNormalizer


def make_env(
    suite: EvalSuite,
    task: str,
    distribution: DistributionSpec,
    num_envs: int = 1,
):
    if suite.env_type == "procgen":
        from src.environments.procgen_wrapper import ProcgenWrapper

        return ProcgenWrapper(
            env_name=task,
            distribution_mode=suite.distribution_mode,
            num_levels=distribution.procgen_num_levels,
            start_level=distribution.procgen_start_level,
            keep_image_format=True,
            num_envs=num_envs,
        )
    if num_envs != 1:
        raise ValueError(f"num_envs>1 only supported for procgen, got env_type={suite.env_type}")
    if suite.env_type == "dmcontrol":
        from src.environments.dmcontrol_wrapper import DMControlWrapper

        domain, task_name = parse_dmcontrol_task(task)
        return DMControlWrapper(
            domain_name=domain,
            task_name=task_name,
            random_seed=distribution.dmcontrol_seed_offset,
        )
    raise ValueError(f"Unknown env_type: {suite.env_type}")


def run_eval_episodes(
    env,
    stack: PerformanceStack,
    normalizer: PerformanceNormalizer,
    device: str,
    n_episodes: int,
    deterministic: bool = True,
    dmcontrol_seed_offset: int = 0,
    mico_embed_ball_radius: float | None = None,
) -> dict[str, float]:
    returns = []
    stack.policy.eval()

    for episode_idx in range(n_episodes):
        reset_seed = None
        if env.action_space_type == "continuous":
            reset_seed = dmcontrol_seed_offset + episode_idx
        obs, _ = env.reset(seed=reset_seed)
        normalizer.reward_norm.reset_episode()
        total = 0.0
        done = False

        while not done:
            obs_tensor = obs.unsqueeze(0).to(device)
            norm_obs = normalizer.observe(obs_tensor)
            with torch.no_grad():
                if stack.policy_on_latent:
                    phi = encode_phi(
                        stack.critic,
                        norm_obs,
                        embed_ball_radius=mico_embed_ball_radius,
                    )
                    action, _ = stack.policy.get_action(phi.squeeze(0), deterministic=deterministic)
                else:
                    action, _ = stack.policy.get_action(norm_obs, deterministic=deterministic)
            step_action = action.item() if env.action_space_type == "discrete" else action
            obs, reward, terminated, truncated, _ = env.step(step_action)
            total += float(reward)
            done = terminated or truncated

        returns.append(total)

    stack.policy.train()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
    }


def resolve_task_path(base: Path, task: str, filename: str) -> Path:
    """Resolve a per-task file under base/{task}/ or return base if it is the file itself."""
    if base.is_file():
        return base
    task_path = base / task / filename
    if not task_path.exists():
        raise FileNotFoundError(f"Missing {task_path} (expected per-task layout: {base}/{{task}}/{filename})")
    return task_path


def load_stack_from_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path,
    obs_dim: int,
    action_dim: int,
    action_space_type: str,
    obs_shape: tuple[int, ...] | None,
    device: str,
    task: str,
) -> tuple[PerformanceStack, PerformanceNormalizer]:
    ckpt_file = resolve_task_path(Path(checkpoint_path), task, "weights_final.pt")
    cfg_file = resolve_task_path(Path(config_path), task, "config.json")

    with open(cfg_file) as f:
        config = json.load(f)

    stack = build_performance_stack_from_config(
        config,
        obs_dim,
        action_dim,
        action_space_type,
        tuple(config["obs_shape"]) if config["obs_shape"] is not None else obs_shape,
        device,
    )
    normalizer = PerformanceNormalizer.from_state_dict(config["normalization"])

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=True)
    stack.policy.load_state_dict(checkpoint["policy"])
    stack.critic.load_state_dict(checkpoint["critic"])
    stack.policy.eval()
    stack.critic.eval()
    return stack, normalizer, config


def _metric_key(task: str, distribution: str, stat: str) -> str:
    return f"eval_{task}_{distribution}_{stat}"


def run_eval_suite(
    suite: EvalSuite,
    device: str,
    checkpoint_path: str | Path,
    config_path: str | Path,
    tasks: list[str] | None = None,
    eval_episodes: int | None = None,
    deterministic: bool | None = None,
) -> dict[str, float]:
    n_episodes = suite.eval_episodes if eval_episodes is None else eval_episodes
    det = suite.eval_deterministic if deterministic is None else deterministic
    task_list = list(tasks) if tasks is not None else list(suite.tasks)

    metrics: dict[str, float] = {}
    for task in task_list:
        probe_env = make_env(suite, task, suite.distributions[0])
        stack, normalizer, config = load_stack_from_checkpoint(
            checkpoint_path,
            config_path,
            probe_env.obs_dim,
            probe_env.action_dim,
            probe_env.action_space_type,
            probe_env.obs_shape,
            device,
            task,
        )
        mico_embed_ball_radius = config["algorithm"].get("mico_embed_ball_radius")
        probe_env.close()

        for distribution in suite.distributions:
            env = make_env(suite, task, distribution)
            seed_offset = distribution.dmcontrol_seed_offset or 0
            result = run_eval_episodes(
                env,
                stack,
                normalizer,
                device,
                n_episodes,
                deterministic=det,
                dmcontrol_seed_offset=seed_offset,
                mico_embed_ball_radius=mico_embed_ball_radius,
            )
            env.close()
            metrics[_metric_key(task, distribution.name, "return_mean")] = result["return_mean"]
            metrics[_metric_key(task, distribution.name, "return_std")] = result["return_std"]

    return metrics


def run_eval_suite_from_checkpoint(
    suite_name: str,
    checkpoint_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    device: str | None = None,
    tasks: list[str] | None = None,
    eval_episodes: int | None = None,
    step: int = 0,
) -> Path:
    suite = EVAL_SUITES[suite_name]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_list = list(tasks) if tasks is not None else list(suite.tasks)

    metrics = run_eval_suite(
        suite,
        device,
        checkpoint_path,
        config_path,
        tasks=task_list,
        eval_episodes=eval_episodes,
    )

    logger = CSVLogger(output_dir, "performance_eval", clear_existing=True)
    logger.save_config(
        {
            "suite": suite_name,
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "tasks": task_list,
            "eval_episodes": eval_episodes or suite.eval_episodes,
            "distributions": [d.name for d in suite.distributions],
        }
    )
    logger.log_metrics(step, metrics)
    logger.close()

    print(f"Performance eval ({suite_name}) -> {output_dir / 'performance_eval_metrics.csv'}")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.4f}")

    return output_dir
