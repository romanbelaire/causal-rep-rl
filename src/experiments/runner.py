"""
CTRO experiment training runner.

Builds MiniGrid + VAE/IMPALA stack, runs PPO/CTRO/ActiveCTRO training.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.agents.active_ctro import ActiveCTRO
from src.agents.ctro import CTRO
from src.agents.ppo import PPO
from src.architectures.critics.vae_critic import VAECritic
from src.architectures.policies.impala import IMPALAPolicy
from src.environments.minigrid_wrapper import MinigridWrapper
from src.experiments.config import BASE_ALGO_CONFIG, BASE_ARCH_CONFIG, BASE_TRAINING_CONFIG
from src.metrics.action_coverage import action_coverage_stats
from src.metrics.behavior_streams import stream_policy_stats
from src.utils.best_episode_recorder import BestEpisodeFrameRecorder
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


def _rollout_common_append(
    buffer: dict,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    action: torch.Tensor,
    log_prob: torch.Tensor,
    control_log_prob: torch.Tensor,
    value: torch.Tensor,
    reward: float,
    terminated: bool,
    truncated: bool,
    episode_id: int,
    step_id: int,
    source: str,
) -> None:
    buffer["obs"].append(obs.cpu())
    buffer["actions"].append(action.cpu())
    buffer["rewards"].append(reward)
    buffer["terminations"].append(terminated)
    buffer["truncations"].append(truncated)
    buffer["dones"].append(terminated or truncated)
    buffer["log_probs"].append(log_prob.cpu())
    buffer["control_log_probs"].append(control_log_prob.cpu())
    buffer["values"].append(value.cpu())
    buffer["next_obs"].append(next_obs.cpu())
    buffer["episode_id"].append(episode_id)
    buffer["step_id"].append(step_id)
    buffer["source"].append(source)


def collect_ppo_rollout(
    env: MinigridWrapper,
    policy: nn.Module,
    critic: nn.Module,
    buffer_size: int,
    device: str,
    episode_id_start: int = 0,
    frame_recorder: BestEpisodeFrameRecorder | None = None,
) -> dict:
    buffer = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "terminations": [],
        "truncations": [],
        "log_probs": [],
        "control_log_probs": [],
        "values": [],
        "next_obs": [],
        "episode_id": [],
        "step_id": [],
        "source": [],
    }
    episode_returns = []
    episode_lengths = []
    current_return = 0.0
    current_length = 0
    episode_id = episode_id_start
    step_id = 0

    while len(buffer["obs"]) < buffer_size:
        obs, _ = env.reset()
        if frame_recorder is not None:
            frame_recorder.start_episode()
            frame_recorder.append_frame(obs)
        done = False
        step_id = 0

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

            _rollout_common_append(
                buffer,
                obs,
                next_obs,
                action,
                log_prob,
                log_prob,
                value,
                reward,
                terminated,
                truncated,
                episode_id,
                step_id,
                "ppo",
            )

            current_return += reward
            current_length += 1
            step_id += 1
            obs = next_obs
            if frame_recorder is not None:
                frame_recorder.add_reward(reward)
                frame_recorder.append_frame(obs)

            if done:
                episode_returns.append(current_return)
                episode_lengths.append(current_length)
                if frame_recorder is not None:
                    frame_recorder.finish_episode()
                current_return = 0.0
                current_length = 0
                episode_id += 1
                step_id = 0

    return _finalize_rollout(buffer, buffer_size, episode_returns, episode_lengths)


def collect_query_rollout(
    agent: ActiveCTRO,
    env: MinigridWrapper,
    buffer_size: int,
    device: str,
    episode_id_start: int = 0,
) -> dict:
    buffer = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "terminations": [],
        "truncations": [],
        "log_probs": [],
        "control_log_probs": [],
        "values": [],
        "next_obs": [],
        "episode_id": [],
        "step_id": [],
        "source": [],
    }
    episode_returns = []
    episode_lengths = []
    current_return = 0.0
    current_length = 0
    episode_id = episode_id_start
    step_id = 0
    policy = agent.policy
    critic = agent.critic

    while len(buffer["obs"]) < buffer_size:
        obs, _ = env.reset()
        done = False
        step_id = 0

        while not done and len(buffer["obs"]) < buffer_size:
            obs_tensor = obs.unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_beta, log_ctrl = agent.sample_query_action(obs_tensor)
                action = action.squeeze(0)
                log_beta = log_beta.squeeze(0)
                log_ctrl = log_ctrl.squeeze(0)
                mu, _ = critic.encode(obs_tensor)
                value = critic(obs_tensor).squeeze(-1).squeeze(0)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            _rollout_common_append(
                buffer,
                obs,
                next_obs,
                action,
                log_beta,
                log_ctrl,
                value,
                reward,
                terminated,
                truncated,
                episode_id,
                step_id,
                "query",
            )

            current_return += reward
            current_length += 1
            step_id += 1
            obs = next_obs

            if done:
                episode_returns.append(current_return)
                episode_lengths.append(current_length)
                current_return = 0.0
                current_length = 0
                episode_id += 1
                step_id = 0

    return _finalize_rollout(buffer, buffer_size, episode_returns, episode_lengths)


def _finalize_rollout(
    buffer: dict,
    buffer_size: int,
    episode_returns: list,
    episode_lengths: list,
) -> dict:
    n = min(len(buffer["obs"]), buffer_size)
    next_values = torch.zeros(n, dtype=torch.float32)
    for t in range(n - 1):
        if buffer["dones"][t]:
            next_values[t] = 0.0
        else:
            next_values[t] = buffer["values"][t + 1]
    return {
        "obs": torch.stack(buffer["obs"][:n]),
        "actions": torch.stack(buffer["actions"][:n]),
        "rewards": torch.tensor(buffer["rewards"][:n], dtype=torch.float32),
        "dones": torch.tensor(buffer["dones"][:n], dtype=torch.bool),
        "terminations": torch.tensor(buffer["terminations"][:n], dtype=torch.bool),
        "truncations": torch.tensor(buffer["truncations"][:n], dtype=torch.bool),
        "log_probs": torch.stack(buffer["log_probs"][:n]),
        "control_log_probs": torch.stack(buffer["control_log_probs"][:n]),
        "behavior_log_prob": torch.stack(buffer["log_probs"][:n]),
        "values": torch.stack(buffer["values"][:n]),
        "next_obs": torch.stack(buffer["next_obs"][:n]),
        "next_values": next_values,
        "episode_id": torch.tensor(buffer["episode_id"][:n], dtype=torch.long),
        "step_id": torch.tensor(buffer["step_id"][:n], dtype=torch.long),
        "source_tag": buffer["source"][:n],
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
    }


def collect_rollout_buffer(
    env: MinigridWrapper,
    policy: nn.Module,
    critic: nn.Module,
    buffer_size: int,
    device: str,
    frame_recorder: BestEpisodeFrameRecorder | None = None,
) -> dict:
    """Legacy alias: PPO-only rollout with term/trunc split."""
    return collect_ppo_rollout(
        env, policy, critic, buffer_size, device, frame_recorder=frame_recorder
    )


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


def _tensor_stats_to_float(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = float(v.mean().item()) if v.numel() > 1 else float(v.item())
        else:
            out[k] = v
    return out


def _json_ready(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = float(v.mean().item()) if v.numel() > 1 else float(v.item())
        else:
            out[k] = v
    return out


_DIAG_STR_KEYS = frozenset({"ref_latent_name"})


def _csv_floats(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in _DIAG_STR_KEYS}


def run_experiment(
    exp_name: str,
    seed: int,
    agent_cls: type,
    algo_overrides: dict | None = None,
    train_overrides: dict | None = None,
    results_root: str | Path = "results",
    device: str | None = None,
    intrinsic_reward: bool = False,
) -> Path:
    algo_cfg = {**BASE_ALGO_CONFIG, **(algo_overrides or {})}
    arch_cfg = BASE_ARCH_CONFIG
    train_cfg = {**BASE_TRAINING_CONFIG, **(train_overrides or {})}

    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path(results_root) / exp_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = MinigridWrapper("MiniGrid-Unlock-v0", seed=seed, keep_image_format=False)
    frame_recorder = BestEpisodeFrameRecorder(env.obs_shape)
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    latent_dim = arch_cfg["critic"]["latent_dim"]

    critic = create_critic(obs_dim, arch_cfg, device)
    policy = create_policy(latent_dim, action_dim, arch_cfg, device)

    agent = agent_cls(policy, critic, algo_cfg, device=device)
    is_active = isinstance(agent, ActiveCTRO) and agent.active_enabled

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
    shared_ref_path = train_cfg["shared_ref_path"]
    dump_ref_path = train_cfg["dump_ref_path"]
    diag_log_epochs = set(train_cfg["diag_log_epochs"])
    min_old_distance = float(algo_cfg["active"]["relational_tr"]["min_old_distance"])
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    last_logged_step = -log_interval
    low_dispersion_steps = 0
    dispersion_warned = False
    episode_counter = 0
    query_buf = None
    uses_query = is_active and agent.uses_query_stream
    eta_ctrl = 0.0

    if shared_ref_path != "":
        ref = torch.load(shared_ref_path, map_location=device, weights_only=False)
        agent.set_reference_buffer(ref)
        agent._pl_ref_frozen = True
        agent.ref_freeze_epoch = 0

    print(f"Starting {exp_name} seed={seed} agent={agent_cls.__name__} device={device}")

    def emit_metrics(epoch_id: int, steps: int, update_stats: dict, aux_stats: dict, pre_update: bool) -> None:
        metrics = {
            "epoch": epoch_id,
            "pre_update": float(pre_update),
            "mean_episode_return": (
                float(np.mean(buffer["episode_returns"]))
                if buffer["episode_returns"]
                else 0.0
            ),
            "reward_dispersion": buffer["rewards"].std(unbiased=False).item(),
            **_tensor_stats_to_float(update_stats),
            **_tensor_stats_to_float(aux_stats),
        }
        metrics.update(
            stream_policy_stats(
                "pi_ctrl",
                buffer["rewards"],
                buffer["actions"],
                action_dim,
                buffer["behavior_log_prob"],
                buffer["control_log_probs"],
                buffer["episode_returns"],
                eta_ctrl,
            )
        )
        metrics["pi_ctrl_stream_frac"] = 1.0
        if uses_query and query_buf is not None:
            metrics.update(
                stream_policy_stats(
                    "pi_query",
                    query_buf["rewards"],
                    query_buf["actions"],
                    action_dim,
                    query_buf["behavior_log_prob"],
                    query_buf["control_log_probs"],
                    query_buf["episode_returns"],
                    agent.uniform_eta,
                )
            )
            n_ppo = float(len(buffer["actions"]))
            n_q = float(len(query_buf["actions"]))
            metrics["pi_ctrl_stream_frac"] = n_ppo / (n_ppo + n_q)
            metrics["pi_query_stream_frac"] = n_q / (n_ppo + n_q)
            metrics["beta_beh_stream_frac"] = n_q / (n_ppo + n_q)
            cov = action_coverage_stats(
                query_buf["actions"],
                action_dim,
                query_buf["behavior_log_prob"],
                agent.uniform_eta,
            )
            metrics.update(cov)
            metrics["query_stream_steps"] = n_q
        if is_active:
            metrics.update(agent.replay_sparse_stats())
        if agent.ref_obs is not None:
            ref_stats = agent.reference_validity_report(min_old_distance)
            (diag_dir / f"epoch_{epoch_id}.json").write_text(
                json.dumps(_json_ready(ref_stats), indent=2) + "\n"
            )
            metrics.update(_csv_floats(ref_stats))
            metrics.update(_csv_floats(agent.target_landscape_pl_stats()))
        metric_results = metric_eval.evaluate(
            critic,
            buffer_dev["obs"],
            buffer_dev["next_obs"],
            buffer_dev["rewards"],
        )
        metrics.update(metric_results)
        if epoch_id % eval_freq == 0 or epoch_id == total_epochs or epoch_id == 0:
            eval_metrics = run_eval_rollout(
                env,
                policy,
                critic,
                device,
                eval_episodes,
                deterministic=train_cfg["eval_deterministic"],
            )
            metrics.update(eval_metrics)
        logger.log_metrics(steps, metrics)

    for epoch in range(1, total_epochs + 1):
        if is_active:
            agent.begin_collection()

        buffer = collect_ppo_rollout(
            env,
            policy,
            critic,
            buffer_size,
            device,
            episode_id_start=episode_counter,
            frame_recorder=frame_recorder,
        )
        episode_counter += len(buffer["episode_returns"]) + 1

        if is_active:
            agent.ingest_rollout(buffer, source="ppo")
            aux_stats = {}
            if uses_query:
                query_size = int(agent.active_cfg["query_rollout_size"])
                query_buf = collect_query_rollout(
                    agent,
                    env,
                    query_size,
                    device,
                    episode_id_start=episode_counter,
                )
                episode_counter += len(query_buf["episode_returns"]) + 1
                agent.ingest_rollout(query_buf, source="query")
            aux_stats = agent.auxiliary_update()
        else:
            aux_stats = {}

        buffer_dev = {k: buffer[k].to(device) if torch.is_tensor(buffer[k]) else buffer[k]
                      for k in buffer if k not in ("episode_returns", "episode_lengths", "source_tag")}

        advantages, returns = agent.compute_gae(
            buffer_dev["rewards"],
            buffer_dev["values"],
            buffer_dev["dones"],
            terminations=buffer_dev["terminations"],
            next_values=buffer_dev["next_values"].to(device),
        )

        if agent.ref_obs is None:
            agent.maybe_init_frozen_pl_ref_buffer(buffer_dev["obs"])
        if dump_ref_path != "" and agent.ref_obs is not None:
            Path(dump_ref_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(agent.ref_obs.detach().cpu(), dump_ref_path)
        if agent.ref_obs is not None:
            agent.update_v_ref_from_obs(agent.ref_obs)
            agent.refresh_pl_target()

        if epoch == 1 and 0 in diag_log_epochs:
            emit_metrics(0, total_steps, {}, aux_stats, pre_update=True)

        update_kwargs = dict(
            obs=buffer_dev["obs"],
            actions=buffer_dev["actions"],
            old_log_probs=buffer_dev["log_probs"],
            advantages=advantages,
            returns=returns,
            training_epoch=epoch,
        )
        if agent.needs_transition_batch:
            update_kwargs["rewards"] = buffer_dev["rewards"]
            update_kwargs["next_obs"] = buffer_dev["next_obs"]

        update_stats = agent.update(**update_kwargs)

        if is_active:
            agent.end_collection()
            agent.refresh_targets()

        total_steps += len(buffer["obs"])

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

        should_log = (
            total_steps - last_logged_step >= log_interval
            or epoch == total_epochs
            or epoch in diag_log_epochs
        )
        if should_log:
            emit_metrics(epoch, total_steps, update_stats, aux_stats, pre_update=False)
            last_logged_step = total_steps

        if epoch % checkpoint_freq == 0:
            torch.save(agent.checkpoint_dict(), str(run_dir / "weights_latest.pt"))

        if epoch % 50 == 0:
            mean_ret = (
                float(np.mean(buffer["episode_returns"]))
                if buffer["episode_returns"]
                else 0.0
            )
            print(f"Epoch {epoch}/{total_epochs} steps={total_steps} return={mean_ret:.3f}")

    torch.save(agent.checkpoint_dict(), str(run_dir / "weights_final.pt"))
    best_episode_path = run_dir / "best_episode_frames.npz"
    frame_recorder.save(best_episode_path)
    if frame_recorder.best_frames is not None:
        print(
            f"Saved best episode (return={frame_recorder.best_return:.3f}, "
            f"length={frame_recorder.best_frames.shape[0]}) -> {best_episode_path}"
        )
    logger.close()
    print(f"Finished {exp_name} seed={seed} -> {run_dir}")
    return run_dir
