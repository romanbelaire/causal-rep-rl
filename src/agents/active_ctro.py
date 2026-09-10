"""Active self-supervised CTRO: PPO + PL + replay Q + action-conditioned MICo + U."""

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.agents.behavior_mixture import sample_behavior_actions
from src.agents.ctro import CTRO
from src.architectures.critics.information_critic import InformationCritic
from src.architectures.critics.q_ensemble import QEnsemble
from src.losses.action_conditioned_mico import action_conditioned_mico_loss, pair_metric
from src.losses.information_value import information_reward, u_td_loss
from src.losses.q_td import (
    action_gap_uncertainty,
    member_bootstrap_mask,
    policy_value_from_q,
    q_td_loss,
)
from src.losses.separation import separation_loss
from src.metrics.pair_equivalence import pair_equivalence_stats
from src.metrics.relational_trust_region import relational_distortion
from src.metrics.reward_test_signature import estimated_d_hat_sig_mean
from src.replay.pair_index import pair_confidence_weights, pair_lower_bound, select_same_action_pairs
from src.replay.priorities import mix_coverage, priority_scores, replay_is_weights
from src.replay.transition_replay import TransitionReplay
from src.utils.ownership import delta_l2, grad_l2, param_l2
from src.utils.target_network import TargetNetwork


class ActiveCTRO(CTRO):
    """CTRO plus dual-stream replay auxiliaries. Disable `active.enabled` for E0."""

    def __init__(self, policy, critic, config, device="cuda", repr_net=None):
        super().__init__(policy, critic, config, device, repr_net=repr_net)
        self.active_cfg = config["active"]
        self.active_enabled = bool(self.active_cfg["enabled"])
        self.n_actions = policy.action_dim
        self.latent_dim = critic.latent_dim
        self.aux_steps = 0
        self.relational_accepts = 0
        self.relational_rejects = 0
        self.q_ensemble = None
        self.u_critic = None
        self.replay = None
        if not self.active_enabled:
            return

        replay_cfg = self.active_cfg["replay"]
        self.replay = TransitionReplay(replay_cfg["capacity"])
        self.replay_batch_size = int(replay_cfg["batch_size"])
        self.coverage_mix = float(replay_cfg["coverage_mix"])
        self.priority_floor = float(replay_cfg["priority_floor"])
        self.is_exponent = float(replay_cfg["importance_exponent"])
        self.n_aux_updates = int(self.active_cfg["n_aux_updates"])

        q_cfg = self.active_cfg["q_ensemble"]
        self.q_enabled = bool(q_cfg["enabled"])
        u_cfg = self.active_cfg["information_critic"]
        self.u_enabled = bool(u_cfg["enabled"])
        mico_cfg = self.active_cfg["mico_ac"]
        self.mico_ac_coef = float(mico_cfg["coefficient"])
        expl = self.active_cfg["exploration"]
        self.query_eps = float(expl["query_mix_epsilon"])
        self.uniform_eta = float(expl["uniform_floor_eta"])
        self.query_stream_enabled = bool(expl["enabled"])
        if self.query_stream_enabled and self.uniform_eta <= 0:
            raise RuntimeError("active CTRO query stream requires uniform_floor_eta > 0")
        sep_cfg = self.active_cfg["separation"]
        self.sep_enabled = bool(sep_cfg["enabled"])
        self.sep_coef = float(sep_cfg["coefficient"])
        self.alpha_sep = float(sep_cfg["alpha_sep"])
        self.cov_eig_floor = float(sep_cfg["cov_eig_floor"])

        repr_params = list(self.critic.parameters())
        if self.repr_net is not None:
            repr_params = list(self.repr_net.parameters()) + repr_params

        if self.q_enabled:
            self.q_ensemble = QEnsemble(
                self.latent_dim,
                self.n_actions,
                int(q_cfg["members"]),
                list(q_cfg["hidden_sizes"]),
            ).to(device)
            self.q_target = TargetNetwork(self.q_ensemble)
            repr_params = repr_params + list(self.q_ensemble.parameters())
        self.policy_target = TargetNetwork(self.policy)
        self.critic_target = TargetNetwork(self.critic)

        if self.u_enabled:
            self.u_critic = InformationCritic(
                self.latent_dim,
                self.n_actions,
                list(u_cfg["hidden_sizes"]),
            ).to(device)
            self.u_target = TargetNetwork(self.u_critic)
            repr_params = repr_params + list(self.u_critic.parameters())

        self.representation_optimizer = optim.Adam(repr_params, lr=self.lr)
        self.rel_cfg = self.active_cfg["relational_tr"]
        self.pri_cfg = self.active_cfg["priorities"]
        self.refresh_targets()

    @property
    def uses_query_stream(self) -> bool:
        return self.active_enabled and self.query_stream_enabled

    def _after_optimizer_step(self, phase: str = "joint") -> None:
        if self.active_enabled:
            return
        super()._after_optimizer_step(phase=phase)

    def begin_collection(self) -> None:
        if not self.active_enabled:
            return
        self.policy_target.begin_collection()
        self.critic_target.begin_collection()
        if self.q_ensemble is not None:
            self.q_target.begin_collection()
        if self.u_critic is not None:
            self.u_target.begin_collection()

    def end_collection(self) -> None:
        if not self.active_enabled:
            return
        self.policy_target.end_collection()
        self.critic_target.end_collection()
        if self.q_ensemble is not None:
            self.q_target.end_collection()
        if self.u_critic is not None:
            self.u_target.end_collection()

    def refresh_targets(self) -> None:
        if not self.active_enabled:
            return
        self.policy_target.hard_copy_from(self.policy)
        self.critic_target.hard_copy_from(self.critic)
        if self.q_ensemble is not None:
            self.q_target.hard_copy_from(self.q_ensemble)
        if self.u_critic is not None:
            self.u_target.hard_copy_from(self.u_critic)

    def encode_online(self, obs: torch.Tensor) -> torch.Tensor:
        if self.repr_net is not None:
            return self.repr_net(obs)
        mu, _ = self.critic.encode(obs)
        return mu

    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self.critic_target.module.encode(obs)
        return mu

    def sample_query_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode_online(obs)
        ctrl_logits = self.policy.forward(z)
        if self.u_critic is not None:
            tau = float(self.active_cfg["information_critic"]["temperature"])
            query_logits = self.u_critic(z) / tau
        else:
            query_logits = torch.zeros_like(ctrl_logits)
        return sample_behavior_actions(
            ctrl_logits, query_logits, self.query_eps, self.uniform_eta
        )

    def ingest_rollout(self, batch: dict, source: str) -> None:
        if not self.active_enabled:
            return
        self.replay.add_batch(
            obs=batch["obs"],
            next_obs=batch["next_obs"],
            action=batch["actions"],
            reward=batch["rewards"],
            terminated=batch["terminations"],
            truncated=batch["truncations"],
            episode_id=batch["episode_id"],
            step_id=batch["step_id"],
            behavior_log_prob=batch["behavior_log_prob"],
            control_log_prob=batch["control_log_probs"],
            source=source,
            policy_version=self.policy_version,
            encoder_version=self.encoder_version,
        )

    def _sample_replay(self) -> tuple[dict, torch.Tensor, torch.Tensor]:
        comps = self.replay.priority_components()
        raw = priority_scores(
            comps["ig"],
            comps["gap"],
            comps["pair"],
            comps["geom"],
            floor=self.priority_floor,
            lambda_ig=float(self.pri_cfg["lambda_ig"]),
            lambda_gap=float(self.pri_cfg["lambda_gap"]),
            lambda_pair=float(self.pri_cfg["lambda_pair"]),
            lambda_geom=float(self.pri_cfg["lambda_geom"]),
        )
        sample_p = mix_coverage(raw, self.coverage_mix)
        idx = self.replay.sample_indices(self.replay_batch_size, sample_p)
        batch = self.replay.gather(idx)
        is_w = replay_is_weights(sample_p[idx], self.replay.size, self.is_exponent)
        return batch, is_w, sample_p

    def auxiliary_update(self) -> dict:
        if not self.active_enabled:
            return {}
        if self.replay.size < self.replay_batch_size:
            return {"aux_skipped": 1.0, "replay_size": float(self.replay.size)}
        stats_acc: dict[str, list] = {}
        for _ in range(self.n_aux_updates):
            step_stats = self._one_aux_step()
            for k, v in step_stats.items():
                stats_acc.setdefault(k, []).append(v)
            self.aux_steps += 1
        out = {}
        for k, vals in stats_acc.items():
            first = vals[0]
            if torch.is_tensor(first):
                out[k] = float(torch.stack([v.detach().cpu() if v.dim() == 0 else v.mean().detach().cpu() for v in vals]).mean().item())
            else:
                out[k] = sum(vals) / len(vals)
        out["replay_size"] = float(self.replay.size)
        out["aux_steps"] = float(self.aux_steps)
        return out

    def _one_aux_step(self) -> dict:
        batch, is_w, _sample_p = self._sample_replay()
        device = self.device
        obs = batch["obs"].to(device)
        next_obs = batch["next_obs"].to(device)
        actions = batch["action"].to(device).long()
        rewards = batch["reward"].to(device)
        terminated = batch["terminated"].to(device)
        is_w = is_w.to(device)
        z = self.encode_online(obs)
        with torch.no_grad():
            z_next_t = self.encode_target(next_obs)
        extra = torch.tensor(0.0, device=device)
        stats: dict = {
            "replay_is_weight_mean": is_w.mean().detach(),
            "replay_ppo_frac": (batch["source"] == 0).float().mean(),
        }

        if self.q_ensemble is not None:
            q = self.q_ensemble(z)
            b = torch.arange(actions.shape[0], device=device)
            q_sa = q[b, :, actions]
            with torch.no_grad():
                next_logits = self.policy_target.module.forward(z_next_t)
                pi_next = F.softmax(next_logits, dim=-1)
                q_bar = self.q_target.module(z_next_t)
                v_bar = policy_value_from_q(q_bar, pi_next)
            mask = member_bootstrap_mask(actions.shape[0], q.shape[1], device)
            q_loss, q_stats, disagreement = q_td_loss(
                q_sa,
                rewards,
                terminated,
                v_bar,
                self.gamma,
                float(self.active_cfg["q_ensemble"]["td_huber_delta"]),
                mask,
            )
            q_loss = (q_loss * is_w.mean())
            extra = extra + q_loss
            stats.update(q_stats)
            stats["own_q_loss"] = q_loss.detach()
            stats["own_q_coef"] = torch.tensor(1.0, device=device)
            gap_unc = action_gap_uncertainty(q.detach(), F.softmax(self.policy.forward(z.detach()), dim=-1))
            ig = disagreement.detach()
            q_mean = q.detach().mean(dim=1)
            stats.update(estimated_d_hat_sig_mean(q_mean))
            stats["train_q_action_gap_unc_mean"] = gap_unc.mean().detach()
            stats["train_q_info_gain_mean"] = ig.mean().detach()
        else:
            ig = torch.zeros(actions.shape[0], device=device)
            gap_unc = torch.zeros(actions.shape[0], device=device)

        mico_cfg = self.active_cfg["mico_ac"]
        pair_i, pair_j, diversity = select_same_action_pairs(
            actions, z.detach(), int(mico_cfg["n_candidates"])
        )
        with torch.no_grad():
            dhat = pair_metric(
                z_next_t[pair_i],
                z_next_t[pair_j],
                mico_cfg["pair_metric"],
                float(mico_cfg["beta_mico"]),
            )
            live = (~terminated[pair_i]).float() * (~terminated[pair_j]).float()
            dhat = (rewards[pair_i] - rewards[pair_j]).abs() + self.gamma * live * dhat
            if self.q_ensemble is not None:
                v_members = v_bar
                sigma = (v_members[pair_i] - v_members[pair_j]).abs().std(dim=1)
            else:
                sigma = torch.full_like(dhat, float("nan"))
        warmup = self.aux_steps < int(mico_cfg["warmup_aux_steps"])
        w_pos, w_neg, undecided = pair_confidence_weights(
            dhat,
            sigma,
            float(mico_cfg["positive_ucb_threshold"]),
            float(mico_cfg["negative_lcb_threshold"]),
            float(mico_cfg["confidence_z"]),
            diversity,
            warmup=warmup,
        )
        d_hat_lower = pair_lower_bound(dhat, sigma, float(mico_cfg["confidence_z"]))
        stats["D_hat_pair"] = dhat.mean().detach()
        stats["D_hat_pair_p50"] = dhat.median().detach()
        stats["D_hat_lower"] = torch.nan_to_num(d_hat_lower, nan=float("nan")).mean().detach()
        stats["D_hat_lower_p50"] = torch.nan_to_num(d_hat_lower, nan=float("nan")).median().detach()
        stats["pair_action_equal"] = 1.0
        stats["pair_reward_abs_diff_mean"] = (rewards[pair_i] - rewards[pair_j]).abs().mean().detach()
        stats.update(pair_equivalence_stats(w_pos, w_neg, undecided))
        if self.mico_ac_coef > 0:
            mico_loss, mico_stats = action_conditioned_mico_loss(
                z,
                z_next_t,
                rewards,
                terminated,
                actions,
                pair_i,
                pair_j,
                self.gamma,
                mico_cfg["pair_metric"],
                float(mico_cfg["beta_mico"]),
                float(mico_cfg["huber_delta"]),
                pair_weight=w_pos,
            )
            extra = extra + self.mico_ac_coef * mico_loss
            stats.update(mico_stats)
            stats["own_mico_loss"] = mico_loss.detach()
            stats["own_mico_coef"] = torch.tensor(self.mico_ac_coef, device=device)
        if self.sep_enabled and self.sep_coef > 0:
            sep_loss, sep_stats = separation_loss(
                z, pair_i, pair_j, d_hat_lower, w_neg, self.alpha_sep
            )
            extra = extra + self.sep_coef * sep_loss
            stats.update(sep_stats)
            stats["own_sep_loss"] = sep_loss.detach()
            stats["own_sep_coef"] = torch.tensor(self.sep_coef, device=device)

        pair_unc = torch.zeros(actions.shape[0], device=device)
        pair_unc[pair_i] = torch.nan_to_num(sigma, nan=0.0)
        geom = torch.zeros(actions.shape[0], device=device)
        close = diversity < 0.1
        geom[pair_i[close]] = 1.0

        if self.u_critic is not None:
            u_cfg = self.active_cfg["information_critic"]
            u = self.u_critic(z)
            u_sa = u[torch.arange(actions.shape[0], device=device), actions]
            i_q = ig
            i_gap = gap_unc
            i_pair = pair_unc
            i = information_reward(
                i_q,
                i_gap,
                i_pair,
                float(u_cfg["lambda_q"]),
                float(u_cfg["lambda_gap"]),
                float(u_cfg["lambda_pair"]),
            )
            with torch.no_grad():
                u_bar = self.u_target.module(z_next_t)
                u_max = u_bar.max(dim=-1).values
            u_loss, u_stats = u_td_loss(
                u_sa, i, terminated, u_max, float(u_cfg["gamma_u"])
            )
            extra = extra + u_loss
            stats.update(u_stats)
            stats["own_u_loss"] = u_loss.detach()
            stats["own_u_coef"] = torch.tensor(1.0, device=device)
            stats["train_u_value_mean"] = u_sa.mean().detach()
            stats["train_u_info_reward_p50"] = i.median().detach()

        enc_params = self._encoder_params()
        enc_before = param_l2(enc_params)
        self.representation_optimizer.zero_grad()
        extra.backward()
        stats["own_aux_encoder_grad"] = grad_l2(enc_params)
        if self.q_ensemble is not None:
            stats["own_q_grad"] = grad_l2(list(self.q_ensemble.parameters()))
        if self.u_critic is not None:
            stats["own_u_grad"] = grad_l2(list(self.u_critic.parameters()))
        torch.nn.utils.clip_grad_norm_(
            self.representation_optimizer.param_groups[0]["params"],
            self.max_grad_norm,
        )
        self.representation_optimizer.step()
        stats["own_aux_encoder_delta"] = delta_l2(enc_before, enc_params)
        stats["own_aux_encoder_step"] = 1.0
        self.encoder_version += 1
        self.replay.set_priorities(batch["indices"], ig.cpu(), gap_unc.cpu(), pair_unc.cpu(), geom.cpu())
        return stats

    def apply_relational_gate(self, snap: dict | None, z_old: torch.Tensor | None) -> dict:
        rel = self.rel_cfg
        if not rel["enabled"]:
            return {}
        ref = self.ref_obs
        z_new = self.encode_online(ref)
        if z_old is None:
            z_old = z_new.detach()
        metric, stats = relational_distortion(
            z_new.detach(), z_old.detach(), float(rel["min_old_distance"])
        )
        out = {k: float(v.item()) if torch.is_tensor(v) else v for k, v in stats.items()}
        out["relational_gate_accept"] = 1.0
        all_excluded = out.get("dz_rel_all_excluded", 0.0) > 0.5
        if all_excluded:
            if rel["mode"] == "hard":
                if snap is None:
                    raise RuntimeError("hard-gate relational mode requires a representation snapshot")
                self.restore_representation(snap)
                self.relational_rejects += 1
                self.own_rollback_count += 1
                out["relational_gate_accept"] = 0.0
            else:
                self.relational_accepts += 1
            out["relational_accept_rate"] = float(
                self.relational_accepts
                / max(1, self.relational_accepts + self.relational_rejects)
            )
            return out
        if rel["mode"] == "hard":
            if snap is None:
                raise RuntimeError("hard-gate relational mode requires a representation snapshot")
            if float(metric.item()) > float(rel["epsilon_z"]):
                self.restore_representation(snap)
                self.relational_rejects += 1
                self.own_rollback_count += 1
                out["relational_gate_accept"] = 0.0
            else:
                self.relational_accepts += 1
        out["relational_accept_rate"] = float(
            self.relational_accepts
            / max(1, self.relational_accepts + self.relational_rejects)
        )
        return out

    def replay_sparse_stats(self) -> dict[str, float]:
        from src.metrics.behavior_streams import sparse_reward_by_source

        return sparse_reward_by_source(self.replay.all_rewards(), self.replay.all_sources())

    def checkpoint_dict(self) -> dict:
        save_dict = super().checkpoint_dict()
        save_dict["active_enabled"] = self.active_enabled
        save_dict["aux_steps"] = self.aux_steps
        if self.q_ensemble is not None:
            save_dict["q_ensemble"] = self.q_ensemble.state_dict()
            save_dict["q_target"] = self.q_target.module.state_dict()
        if self.u_critic is not None:
            save_dict["u_critic"] = self.u_critic.state_dict()
            save_dict["u_target"] = self.u_target.module.state_dict()
        return save_dict

    def update(self, **kwargs) -> dict:
        """PPO update plus optional relational gate on the PL reference batch."""
        rel_tr = self.active_cfg["relational_tr"] if self.active_enabled else {}
        if self.active_enabled and rel_tr.get("enabled", False):
            obs = kwargs.get("obs")
            if obs is not None and self.ref_obs is None:
                self.maybe_init_frozen_pl_ref_buffer(obs)
        snap = None
        z_old = None
        if self.active_enabled and rel_tr.get("enabled", False):
            if self.ref_obs is None:
                raise RuntimeError("relational gate requires a frozen reference batch")
            snap = self.snapshot_representation()
            z_old = self.encode_online(self.ref_obs).detach()
        stats = super().update(**kwargs)
        if self.active_enabled and rel_tr.get("enabled", False):
            gate_stats = self.apply_relational_gate(snap, z_old)
            for k, v in gate_stats.items():
                stats[k] = v
        return stats

    def _load_checkpoint(self, checkpoint: dict, weights_only: bool = False) -> None:
        super()._load_checkpoint(checkpoint, weights_only=weights_only)
        if not self.active_enabled:
            return
        if self.q_ensemble is not None and "q_ensemble" in checkpoint:
            self.q_ensemble.load_state_dict(checkpoint["q_ensemble"])
            self.q_target.module.load_state_dict(checkpoint["q_target"])
        if self.u_critic is not None and "u_critic" in checkpoint:
            self.u_critic.load_state_dict(checkpoint["u_critic"])
            self.u_target.module.load_state_dict(checkpoint["u_target"])
        if "aux_steps" in checkpoint:
            self.aux_steps = int(checkpoint["aux_steps"])
