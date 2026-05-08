"""
Policy regret computation.
"""

import torch
import numpy as np


def compute_policy_regret(
    policy_returns: list[float],
    optimal_return: float | None = None,
    baseline_return: float | None = None,
) -> dict:
    """
    Compute policy regret: difference to optimal/baseline policy return.
    
    Args:
        policy_returns: List of episode returns from current policy
        optimal_return: Optimal policy return (if known)
        baseline_return: Baseline policy return (for comparison)
        
    Returns:
        Dictionary with regret metrics
    """
    policy_mean = np.mean(policy_returns)
    policy_std = np.std(policy_returns)
    
    result = {
        "mean_return": policy_mean,
        "std_return": policy_std,
    }
    
    if optimal_return is not None:
        regret = optimal_return - policy_mean
        result["regret_vs_optimal"] = regret
        result["regret_pct_vs_optimal"] = (regret / optimal_return) * 100 if optimal_return > 0 else 0
    
    if baseline_return is not None:
        regret_baseline = baseline_return - policy_mean
        result["regret_vs_baseline"] = regret_baseline
        result["regret_pct_vs_baseline"] = (regret_baseline / baseline_return) * 100 if baseline_return > 0 else 0
    
    return result


def estimate_optimal_return(
    env,
    num_episodes: int = 100,
    max_steps: int = 1000,
) -> float:
    """
    Estimate optimal return by running many episodes with random/optimal policy.
    
    This is a placeholder - actual implementation depends on environment.
    
    Args:
        env: Environment
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        Estimated optimal return
    """
    # Placeholder: would need environment-specific optimal policy
    # For now, return None to indicate unknown
    return None

