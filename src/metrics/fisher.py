"""
Fisher Information computation for policies.
"""

import torch
import torch.nn as nn


def compute_fisher_information(
    policy: nn.Module,
    obs: torch.Tensor,
    num_samples: int = 100,
) -> dict:
    """
    Estimate policy Fisher information at sampled states.
    
    Fisher Information: F(θ) = E[∇ log π(a|s) ∇ log π(a|s)^T]
    
    Args:
        policy: Policy network
        obs: Observations [N, obs_dim]
        num_samples: Number of action samples per state
        
    Returns:
        Dictionary with:
            - fisher_matrix: Fisher information matrix (if small enough)
            - fisher_trace: Trace of Fisher matrix
            - fisher_norm: Frobenius norm of Fisher matrix
    """
    # Sample actions from policy
    actions_list = []
    log_probs_list = []
    
    for _ in range(num_samples):
        actions, log_probs = policy.get_action(obs)
        actions_list.append(actions)
        log_probs_list.append(log_probs)
    
    # Stack
    actions = torch.stack(actions_list)  # [num_samples, N]
    log_probs = torch.stack(log_probs_list)  # [num_samples, N]
    
    # Compute gradients of log probabilities
    fisher_grads = []
    
    for i in range(len(obs)):
        # Average over action samples for this state
        avg_log_prob = log_probs[:, i].mean()
        
        # Compute gradient
        grad = torch.autograd.grad(avg_log_prob, policy.parameters(), retain_graph=True)
        flat_grad = torch.cat([g.view(-1) for g in grad])
        fisher_grads.append(flat_grad)
    
    # Stack gradients
    fisher_grads = torch.stack(fisher_grads)  # [N, param_dim]
    
    # Compute Fisher matrix: E[g g^T]
    fisher_matrix = torch.mean(fisher_grads.unsqueeze(-1) * fisher_grads.unsqueeze(-2), dim=0)
    
    # Compute statistics
    fisher_trace = torch.trace(fisher_matrix).item()
    fisher_norm = torch.norm(fisher_matrix, p='fro').item()
    
    result = {
        "fisher_trace": fisher_trace,
        "fisher_norm": fisher_norm,
    }
    
    # Only store full matrix if it's small enough
    if fisher_matrix.numel() < 10000:  # Reasonable size limit
        result["fisher_matrix"] = fisher_matrix
    
    return result


def compute_fisher_information_index(
    policy: nn.Module,
    obs: torch.Tensor,
) -> float:
    """
    Compute Fisher information index (scalar summary).
    
    Args:
        policy: Policy network
        obs: Observations [N, obs_dim]
        
    Returns:
        Fisher information index (trace)
    """
    fisher_info = compute_fisher_information(policy, obs, num_samples=50)
    return fisher_info["fisher_trace"]

