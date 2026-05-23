"""
PSD quadratic value heads.

QuadraticLatentPSDValueHead: V(Z) = Z^T A^T A Z + b^T Z + c on encoder latent Z.
  ∇²_Z V = 2 A^T A (constant). μ_latent = 2 σ_min(A)².

QuadraticBottleneckValueHead (legacy): MLP then quadratic on bottleneck — μ w.r.t. Z ≠ 2σ_min(A)².
"""

import torch
import torch.nn as nn


class _QuadraticPSDCore(nn.Module):
    """Shared A, b, c and analytic μ for V(z) = z^T A^T A z + b^T z + c."""

    def __init__(self, dim: int, mu_min_floor: float = 0.0):
        super().__init__()
        self.dim = dim
        self.mu_min_floor = mu_min_floor

        self.A_free = nn.Parameter(torch.empty(dim, dim))
        if mu_min_floor > 0.0:
            nn.init.zeros_(self.A_free)
        else:
            nn.init.orthogonal_(self.A_free, gain=0.01)
        self.b = nn.Parameter(torch.zeros(dim))
        self.c = nn.Parameter(torch.zeros(1))

        floor_scale = (mu_min_floor / 2.0) ** 0.5 if mu_min_floor > 0.0 else 0.0
        self.register_buffer("_floor_scale", torch.tensor(floor_scale))

    def A_matrix(self) -> torch.Tensor:
        A = self.A_free
        if self.mu_min_floor > 0.0:
            A = A + self._floor_scale * torch.eye(self.dim, device=A.device, dtype=A.dtype)
        return A

    def quadratic_value(self, z: torch.Tensor) -> torch.Tensor:
        A = self.A_matrix()
        Az = z @ A.T
        quad = (Az * Az).sum(dim=-1, keepdim=True)
        linear = (self.b * z).sum(dim=-1, keepdim=True)
        return quad + linear + self.c

    def analytic_mu_latent(self) -> torch.Tensor:
        """λ_min(∇²_z V) = 2 σ_min(A)² (constant w.r.t. z)."""
        s = torch.linalg.svdvals(self.A_matrix())
        return 2.0 * s[-1] ** 2

    def hessian_latent(self) -> torch.Tensor:
        A = self.A_matrix()
        return 2.0 * (A.T @ A)


class QuadraticLatentPSDValueHead(_QuadraticPSDCore):
    """V(Z) = Z^T A^T A Z + b^T Z + c with Z the VAE encoder latent (no intervening nonlinearity)."""

    def __init__(self, latent_dim: int, mu_min_floor: float = 0.0):
        super().__init__(latent_dim, mu_min_floor)
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.quadratic_value(z)


class QuadraticBottleneckValueHead(nn.Module):
    """Legacy: linear → ReLU → linear → quadratic on bottleneck (μ w.r.t. Z requires chain rule)."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        mu_min_floor: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim
        self.mu_min_floor = mu_min_floor

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, bottleneck_dim)

        self._quad = _QuadraticPSDCore(bottleneck_dim, mu_min_floor)

    def A_matrix(self) -> torch.Tensor:
        return self._quad.A_matrix()

    def bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        return self._quad.quadratic_value(z)

    def analytic_mu_bottleneck(self) -> torch.Tensor:
        return self._quad.analytic_mu_latent()
