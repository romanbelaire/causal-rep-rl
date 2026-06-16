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
from src.evaluation.runner import make_env, run_eval_episodes
from src.evaluation.suites import DistributionSpec, EVAL_SUITES
from src.experiments.config import BASE_ALGO_CONFIG, PERFORMANCE_SUITE_CONFIG
from src.experiments.runner import create_critic, set_seed
from src.architectures.policies.impala import IMPALAPolicy
from src.utils.ctro_metric_evaluator import CTROMetricEvaluator
from src.utils.logging import CSVLogger


def create_policy(
    repr_dim: int,
    action_dim: int,
    arch: dict,
    device: str,
    action_space_type: str,
) -> IMPALAPolicy:
    p = arch["policy"]
    return IMPALAPolicy(
        repr_dim,
        action_dim,
        p["hidden_sizes"],
        p["activation"],
        action_space_type,
        p["num_residual_blocks"],
    ).to(device)


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


def collect_rollout_buffer(
    env,
    policy: nn.Module,
    critic: nn.Module,
    buffer_size: int,
    device: str,
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
        done = False

        while not done and len(buffer["obs"]) < buffer_size:
            obs_tensor = obs.unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = critic.encode(obs_tensor)
                action, log_prob = policy.get_action(mu)
                value = critic(obs_tensor).squeeze(-1)
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
                value = value.squeeze(0)

            step_action = action.item() if env.action_space_type == "discrete" else action
            next_obs, reward, terminated, truncated, _ = env.step(step_action)
            done = terminated or truncated

            buffer["obs"].append(obs.cpu())
            buffer["actions"].append(action.cpu())
            buffer["rewards"].append(reward)
            buffer["dones"].append(done)
            buffer["log_probs"].append(log_prob.cpu())
            buffer["values"].append(value.cpu())
            buffer["next_obs"].append(next_obs.cpu())

            current_return += reward
            obs = next_obs

            if done:
                episode_returns.append(current_return)
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
    algo_cfg = {**BASE_ALGO_CONFIG, **(algo_overrides or {})}
    arch_cfg = suite_cfg["arch"]
    train_cfg = suite_cfg["training"]

    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    prefix = suite_cfg["results_prefix"]
    run_dir = Path(results_root) / prefix / exp_name / f"seed_{seed}" / task
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_train_env(suite_name, task)
    critic = create_critic(env.obs_dim, arch_cfg, device)
    policy = create_policy(
        arch_cfg["critic"]["latent_dim"],
        env.action_dim,
        arch_cfg,
        device,
        env.action_space_type,
    )
    agent = agent_cls(policy, critic, algo_cfg, device=device)

    full_config = {
        "experiment": exp_name,
        "suite": suite_name,
        "task": task,
        "seed": seed,
        "algorithm": algo_cfg,
        "architecture": arch_cfg,
        "training": train_cfg,
        "agent_class": agent_cls.__name__,
    }

    logger = CSVLogger(run_dir, "", clear_existing=True)
    logger.save_config(full_config)

    metric_eval = CTROMetricEvaluator(gamma=algo_cfg["gamma"])
    buffer_size = train_cfg["buffer_size"]
    total_epochs = train_cfg["total_epochs"]
    log_interval = train_cfg["log_interval_steps"]
    eval_freq = train_cfg["eval_frequency"]
    eval_episodes = train_cfg["eval_episodes"]
    checkpoint_freq = train_cfg["checkpoint_frequency"]

    total_steps = 0
    last_logged_step = -log_interval

    print(f"Starting {suite_name}/{task} seed={seed} exp={exp_name} device={device}")

    for epoch in range(1, total_epochs + 1):
        buffer = collect_rollout_buffer(env, policy, critic, buffer_size, device)
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
            metrics.update(
                metric_eval.evaluate(
                    critic,
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
                        policy,
                        critic,
                        device,
                        eval_episodes,
                        deterministic=train_cfg["eval_deterministic"],
                        dmcontrol_seed_offset=seed_offset,
                    )
                    eval_env.close()
                    metrics[f"eval_{distribution.name}_return_mean"] = result["return_mean"]
                    metrics[f"eval_{distribution.name}_return_std"] = result["return_std"]

            logger.log_metrics(total_steps, metrics)
            last_logged_step = total_steps

        if epoch % checkpoint_freq == 0:
            agent.save(str(run_dir / "weights_latest.pt"))

        if epoch % 50 == 0:
            mean_ret = (
                float(np.mean(buffer["episode_returns"]))
                if buffer["episode_returns"]
                else 0.0
            )
            print(f"Epoch {epoch}/{total_epochs} steps={total_steps} return={mean_ret:.3f}")

    agent.save(str(run_dir / "weights_final.pt"))
    env.close()
    logger.close()
    print(f"Finished {suite_name}/{task} seed={seed} -> {run_dir}")
    return run_dir
