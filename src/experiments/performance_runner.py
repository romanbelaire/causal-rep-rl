"""
Training runner for Procgen and DMControl performance environments.

Trains one task and writes checkpoints to the layout expected by run_performance_eval:
  results/{suite_prefix}/{exp_name}/seed_{seed}/{task}/weights_final.pt
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.agents.ctro import CTRO
from src.agents.ppo import PPO
from src.evaluation.runner import make_env, run_eval_episodes
from src.evaluation.suites import DistributionSpec, EVAL_SUITES
from src.experiments.config import BASE_ALGO_CONFIG, PERFORMANCE_SUITE_CONFIG
from src.experiments.performance_models import PerformanceStack, build_performance_stack
from src.experiments.runner import set_seed
from src.utils.best_episode_recorder import BestEpisodeFrameRecorder, make_best_episode_frame_recorder
from src.utils.bisimulation_utils import encode_phi
from src.utils.ctro_metric_evaluator import CTROMetricEvaluator
from src.utils.logging import CSVLogger
from src.utils.normalization import PerformanceNormalizer


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
        "log_probs": torch.stack(buffer["log_probs"][:n]),
        "values": torch.stack(buffer["values"][:n]),
        "next_obs": torch.stack(buffer["next_obs"][:n]),
        "episode_returns": episode_returns,
    }


def run_performance_train(
    suite_name: str,
    task: str,
    seed: int,
    exp_name: str = "exp_full",
    agent_cls: type = CTRO,
    algo_overrides: dict | None = None,
    results_root: str | Path = "results",
    device: str | None = None,
) -> Path:
    suite = EVAL_SUITES[suite_name]
    suite_cfg = PERFORMANCE_SUITE_CONFIG[suite_name]
    algo_key = "ctro_algo" if agent_cls is CTRO else "ppo_algo"
    algo_cfg = {**BASE_ALGO_CONFIG, **suite_cfg[algo_key], **(algo_overrides or {})}
    arch_cfg = suite_cfg["arch"]
    train_cfg = suite_cfg["training"]

    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    prefix = suite_cfg["results_prefix"]
    run_dir = Path(results_root) / prefix / exp_name / f"seed_{seed}" / task
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_train_env(suite_name, task)
    frame_recorder = make_best_episode_frame_recorder(env)
    stack = build_performance_stack(arch_cfg, env, agent_cls, device)
    normalizer = PerformanceNormalizer(_obs_norm_shape(env), algo_cfg["gamma"])
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

    metric_eval = None
    if agent_cls is CTRO:
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

    total_steps = 0
    last_logged_step = -log_interval

    print(
        f"Starting {suite_name}/{task} seed={seed} exp={exp_name} "
        f"agent={agent_cls.__name__} stack={stack.stack_type} device={device} "
        f"epochs={total_epochs}",
        flush=True,
    )

    for epoch in range(1, total_epochs + 1):
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

        advantages, returns = agent.compute_gae(
            buffer["rewards"],
            buffer["values"],
            buffer["dones"],
            next_value=0.0,
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

        should_log = total_steps - last_logged_step >= log_interval or epoch == total_epochs
        if should_log:
            metrics = {
                "epoch": epoch,
                "mean_episode_return": (
                    float(np.mean(buffer["episode_returns"]))
                    if buffer["episode_returns"]
                    else 0.0
                ),
                **update_stats,
            }
            if metric_eval is not None:
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

            logger.log_metrics(total_steps, metrics)
            last_logged_step = total_steps

        if epoch % checkpoint_freq == 0:
            agent.save(str(run_dir / "weights_latest.pt"))

        if epoch % print_every == 0:
            mean_ret = (
                float(np.mean(buffer["episode_returns"]))
                if buffer["episode_returns"]
                else 0.0
            )
            print(
                f"Epoch {epoch}/{total_epochs} steps={total_steps} return={mean_ret:.3f}",
                flush=True,
            )

    agent.save(str(run_dir / "weights_final.pt"))
    full_config["normalization"] = normalizer.state_dict()
    logger.save_config(full_config)

    if frame_recorder is not None:
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
    return run_dir
