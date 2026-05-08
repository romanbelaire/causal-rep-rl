"""
Occupancy measure stability metrics.
"""

import torch
import numpy as np
from collections import defaultdict


def compute_occupancy_measure(
    states: torch.Tensor,
    discretize: bool = True,
    grid_size: int = 10,
) -> dict:
    """
    Compute state occupancy measure (state visitation distribution).
    
    Args:
        states: States [N, state_dim]
        discretize: Whether to discretize continuous states
        grid_size: Grid size for discretization
        
    Returns:
        Dictionary with:
            - occupancy: Occupancy distribution
            - entropy: Entropy of distribution
            - unique_states: Number of unique states visited
    """
    if discretize and states.shape[1] > 1:
        # Discretize continuous states
        states_discrete = discretize_states(states, grid_size)
    else:
        states_discrete = states
    
    # Count state visits
    state_counts = defaultdict(int)
    for state in states_discrete:
        state_key = tuple(state.tolist() if isinstance(state, torch.Tensor) else state)
        state_counts[state_key] += 1
    
    # Normalize to get distribution
    total = len(states_discrete)
    occupancy = {k: v / total for k, v in state_counts.items()}
    
    # Compute entropy
    probs = np.array(list(occupancy.values()))
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return {
        "occupancy": occupancy,
        "entropy": entropy,
        "unique_states": len(occupancy),
        "total_states": total,
    }


def discretize_states(states: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Discretize continuous states to grid."""
    # Normalize to [0, 1]
    states_norm = (states - states.min(dim=0)[0]) / (states.max(dim=0)[0] - states.min(dim=0)[0] + 1e-8)
    
    # Discretize
    states_discrete = (states_norm * grid_size).long()
    states_discrete = torch.clamp(states_discrete, 0, grid_size - 1)
    
    return states_discrete


def compute_occupancy_stability(
    occupancy1: dict,
    occupancy2: dict,
    metric: str = "kl",
) -> float:
    """
    Compute stability between two occupancy measures.
    
    Args:
        occupancy1: First occupancy measure
        occupancy2: Second occupancy measure
        metric: "kl" (KL divergence) or "tv" (Total Variation)
        
    Returns:
        Stability metric value
    """
    # Get all unique states
    all_states = set(occupancy1["occupancy"].keys()) | set(occupancy2["occupancy"].keys())
    
    # Create probability vectors
    p1 = np.array([occupancy1["occupancy"].get(s, 0) for s in all_states])
    p2 = np.array([occupancy2["occupancy"].get(s, 0) for s in all_states])
    
    # Normalize
    p1 = p1 / (p1.sum() + 1e-10)
    p2 = p2 / (p2.sum() + 1e-10)
    
    if metric == "kl":
        # KL divergence: D_KL(p1 || p2)
        kl = np.sum(p1 * np.log(p1 / (p2 + 1e-10) + 1e-10))
        return kl
    elif metric == "tv":
        # Total Variation: 0.5 * sum(|p1 - p2|)
        tv = 0.5 * np.sum(np.abs(p1 - p2))
        return tv
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_occupancy_kl(
    occupancy1: dict,
    occupancy2: dict,
) -> float:
    """Compute KL divergence between occupancy measures."""
    return compute_occupancy_stability(occupancy1, occupancy2, metric="kl")


def compute_occupancy_tv(
    occupancy1: dict,
    occupancy2: dict,
) -> float:
    """Compute Total Variation between occupancy measures."""
    return compute_occupancy_stability(occupancy1, occupancy2, metric="tv")

