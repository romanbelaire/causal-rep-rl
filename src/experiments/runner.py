"""
CTRO experiment training runner.

Builds Minigrid + VAE/IMPALA stack, runs PPO/CTRO training, logs every 10k steps.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.agents.ctro import CTRO
from src.agents.ppo import PPO
from src.architectures.critics.vae_critic import VAECritic
from src.architectures.policies.impala import IMPALAPolicy
from src.environments.minigrid_wrapper import MinigridWrapper
from src.experiments.config import BASE_ALGO_CONFIG, BASE_ARCH_CONFIG, BASE_TRAINING_CONFIG
from src.utils.ctro_metric_evaluator import CTROMetricEvaluator
from src.utils.logging import CSVLogger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_critic(obs_dim: int, arch: dict, device: str) -> VAECritic:
    c = arch["critic"]
    return VAECritic(
        obs_dim,
        c["latent_dim"],
        c["encoder_hidden"],
        c["decoder_hidden"],
        c["value_hidden"],
        c["activation"],
        c["beta"],
    ).to(device)


def create_policy(repr_dim: int, action_dim: int, arch: dict, device: str) -> IMPALAPolicy:
    p = arch["policy"]
    return IMPALAPolicy(
        repr_dim,
        action_dim,
        p["hidden_sizes"],
        p["activation"],
        "discrete",
        p["num_residual_blocks"],
    ).to(device)


def collect_rollout_buffer(
    env: MinigridWrapper,
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
    episode_lengths = []
    current_return = 0.0
    current_length = 0

    while len(buffer["obs"]) < buffer_size:
        obs, _ = env.reset()
        done = False

        while not done and len(buffer["obs"]) < buffer_size:
            obs_tensor = obs.unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = critic.encode(obs_tensor)
                z = mu
                action, log_prob = policy.get_action(z)
                value = critic(obs_tensor).squeeze(-1)
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
                value = value.squeeze(0)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            buffer["obs"].append(obs.cpu())
            buffer["actions"].append(action.cpu())
            buffer["rewards"].append(reward)
            buffer["dones"].append(done)
            buffer["log_probs"].append(log_prob.cpu())
            buffer["values"].append(value.cpu())
            buffer["next_obs"].append(next_obs.cpu())

            current_return += reward
            current_length += 1
            obs = next_obs

            if done:
                episode_returns.append(current_return)
                episode_lengths.append(current_length)
                current_return = 0.0
                current_length = 0

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
        "episode_lengths": episode_lengths,
    }


def run_eval_rollout(
    env: MinigridWrapper,
    policy: nn.Module,
    critic: nn.Module,
    device: str,
    n_episodes: int,
    deterministic: bool = True,
) -> dict:
    rewards = []
    policy.eval()
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            obs_tensor = obs.unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = critic.encode(obs_tensor)
                action, _ = policy.get_action(mu, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
    policy.train()
    return {
        "eval_return_mean": float(np.mean(rewards)),
        "eval_return_std": float(np.std(rewards)),
    }


def run_experiment(
    exp_name: str,
    seed: int,
    agent_cls: type,
    algo_overrides: dict | None = None,
    results_root: str | Path = "results",
    device: str | None = None,
    intrinsic_reward: bool = False,
) -> Path:
    """
    Run a single CTRO experiment seed.

    Results: results/{exp_name}/seed_{seed}/metrics.csv, config.json, weights_final.pt
    """
    algo_cfg = {**BASE_ALGO_CONFIG, **(algo_overrides or {})}
    arch_cfg = BASE_ARCH_CONFIG
    train_cfg = BASE_TRAINING_CONFIG

    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path(results_root) / exp_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = MinigridWrapper("MiniGrid-Unlock-v0", seed=seed, keep_image_format=False)
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    latent_dim = arch_cfg["critic"]["latent_dim"]

    critic = create_critic(obs_dim, arch_cfg, device)
    policy = create_policy(latent_dim, action_dim, arch_cfg, device)

    agent = agent_cls(policy, critic, algo_cfg, device=device)

    full_config = {
        "experiment": exp_name,
        "seed": seed,
        "algorithm": algo_cfg,
        "architecture": arch_cfg,
        "training": train_cfg,
        "agent_class": agent_cls.__name__,
        "intrinsic_reward": intrinsic_reward,
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
    dispersion_threshold = train_cfg["reward_dispersion_threshold"]
    dispersion_warn_steps = train_cfg["reward_dispersion_warn_steps"]

    total_steps = 0
    last_logged_step = -log_interval
    low_dispersion_steps = 0
    dispersion_warned = False

    print(f"Starting {exp_name} seed={seed} agent={agent_cls.__name__} device={device}")

    for epoch in range(1, total_epochs + 1):
        buffer = collect_rollout_buffer(env, policy, critic, buffer_size, device)
        buffer["obs"] = buffer["obs"].to(device)
        buffer["actions"] = buffer["actions"].to(device)
        buffer["rewards"] = buffer["rewards"].to(device)
        buffer["dones"] = buffer["dones"].to(device)
        buffer["log_probs"] = buffer["log_probs"].to(device)
        buffer["values"] = buffer["values"].to(device)
        buffer["next_obs"] = buffer["next_obs"].to(device)

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

        dispersion = buffer["rewards"].std(unbiased=False).item()
        if dispersion < dispersion_threshold:
            low_dispersion_steps += len(buffer["obs"])
        else:
            low_dispersion_steps = 0

        if (
            not dispersion_warned
            and low_dispersion_steps >= dispersion_warn_steps
        ):
            print(
                f"WARNING: reward dispersion < {dispersion_threshold} for "
                f"{low_dispersion_steps} steps — MICo target may be degenerate."
            )
            dispersion_warned = True

        should_log = total_steps - last_logged_step >= log_interval or epoch == total_epochs
        if should_log:
            metrics = {
                "epoch": epoch,
                "mean_episode_return": (
                    float(np.mean(buffer["episode_returns"]))
                    if buffer["episode_returns"]
                    else 0.0
                ),
                "reward_dispersion": dispersion,
                **update_stats,
            }
            metric_results = metric_eval.evaluate(
                critic,
                buffer["obs"],
                buffer["next_obs"],
                buffer["rewards"],
            )
            metrics.update(metric_results)

            if epoch % eval_freq == 0 or epoch == total_epochs:
                eval_metrics = run_eval_rollout(
                    env,
                    policy,
                    critic,
                    device,
                    eval_episodes,
                    deterministic=train_cfg["eval_deterministic"],
                )
                metrics.update(eval_metrics)

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
    logger.close()
    print(f"Finished {exp_name} seed={seed} -> {run_dir}")
    return run_dir
