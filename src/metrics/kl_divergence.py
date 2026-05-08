"""
KL Divergence metric computation.
"""

import torch
import torch.nn as nn


def compute_policy_kl(
    policy: nn.Module,
    obs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> float:
    """
    Compute KL divergence between current policy and old policy.
    
    Args:
        policy: Current policy network
        obs: Observations [N, obs_dim]
        old_log_probs: Old log probabilities [N]
        
    Returns:
        Average KL divergence
    """
    with torch.no_grad():
        # Get current policy log probs for the same actions
        # Note: This is approximate - we'd need the old actions to compute exact KL
        # For now, we compute KL using the state distribution
        # This requires sampling actions from current policy
        
        # Sample actions from current policy
        actions, _ = policy.get_action(obs)
        
        # Get current log probs
        if hasattr(policy, 'evaluate_actions'):
            current_log_probs, _ = policy.evaluate_actions(obs, actions)
        else:
            # Fallback: estimate from policy output
            logits = policy(obs)
            dist = torch.distributions.Categorical(logits=logits)
            current_log_probs = dist.log_prob(actions)
        
        # KL divergence: E[log(pi_old / pi_new)] = E[log_probs_old - log_probs_new]
        kl = (old_log_probs - current_log_probs).mean().item()
        
        return abs(kl)


def compute_kl_between_policies(
    policy1: nn.Module,
    policy2: nn.Module,
    obs: torch.Tensor,
) -> float:
    """
    Compute KL divergence between two policies.
    
    Args:
        policy1: First policy
        policy2: Second policy
        obs: Observations [N, obs_dim]
        
    Returns:
        Average KL divergence
    """
    with torch.no_grad():
        # Sample actions from policy1
        actions, log_probs1 = policy1.get_action(obs)
        
        # Get log probs under policy2
        if hasattr(policy2, 'evaluate_actions'):
            log_probs2, _ = policy2.evaluate_actions(obs, actions)
        else:
            logits = policy2(obs)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs2 = dist.log_prob(actions)
        
        # KL divergence
        kl = (log_probs1 - log_probs2).mean().item()
        
        return abs(kl)

