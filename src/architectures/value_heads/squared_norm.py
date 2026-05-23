"""
Value head: 2-layer MLP with squared output V(x) = ||f(x)||².

At f ≈ 0, ∇²V ≈ 2 J_f^T J_f. μ is tracked via λ_min(2 J_f^T J_f) per state.
"""

import torch
import torch.nn as nn


class SquaredNormValueHead(nn.Module):
    """linear → ReLU → linear → ||f||²."""

    def __init__(self, latent_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, feature_dim)

        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)
        nn.init.zeros_(self.fc2.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        return (f * f).sum(dim=-1, keepdim=True)

    def mu_jacobian_proxy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample proxy μ_i = 2 λ_min(J_f^T J_f) at x_i.

        Dominant Hessian term when ||f|| is small; cheap for small latent_dim.
        """
        mus = []
        for i in range(x.shape[0]):

            def f_vec(xi: torch.Tensor) -> torch.Tensor:
                return self.features(xi.unsqueeze(0)).squeeze(0)

            J = torch.autograd.functional.jacobian(f_vec, x[i], create_graph=False)
            G = J.T @ J
            G = (G + G.T) * 0.5
            mus.append(2.0 * torch.linalg.eigvalsh(G)[0].relu())

        return torch.stack(mus)

    def mu_jacobian_proxy_batch_min(self, x: torch.Tensor) -> torch.Tensor:
        """Minimum μ across batch (for repr loss weighting)."""
        return self.mu_jacobian_proxy(x).min()
