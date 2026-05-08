"""
VAE-based Critic: Uses variational autoencoder for causal feature encoding,
then value function over latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAECritic(nn.Module):
    """
    VAE-based critic that:
    1. Encodes observations to latent causal representation
    2. Predicts value from latent representation
    """
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 32,
        encoder_hidden: list[int] = [256, 256],
        decoder_hidden: list[int] = [256, 256],
        value_hidden: list[int] = [128, 128],
        activation: str = "relu",
        beta: float = 1.0,  # VAE KL loss weight
    ):
        """
        Initialize VAE-based critic.
        
        Args:
            obs_dim: Observation dimension
            latent_dim: Latent representation dimension
            encoder_hidden: Encoder hidden layer sizes
            decoder_hidden: Decoder hidden layer sizes
            value_hidden: Value head hidden layer sizes
            activation: Activation function
            beta: VAE beta parameter (KL loss weight)
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Encoder: obs -> latent mean and log_std
        encoder_layers = []
        input_dim = obs_dim
        for hidden_size in encoder_hidden:
            encoder_layers.append(nn.Linear(input_dim, hidden_size))
            encoder_layers.append(self.activation)
            input_dim = hidden_size
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_log_std = nn.Linear(input_dim, latent_dim)
        
        # Decoder: latent -> obs (for reconstruction)
        decoder_layers = []
        input_dim = latent_dim
        for hidden_size in decoder_hidden:
            decoder_layers.append(nn.Linear(input_dim, hidden_size))
            decoder_layers.append(self.activation)
            input_dim = hidden_size
        decoder_layers.append(nn.Linear(input_dim, obs_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Value head: latent -> value
        value_layers = []
        input_dim = latent_dim
        for hidden_size in value_hidden:
            value_layers.append(nn.Linear(input_dim, hidden_size))
            value_layers.append(self.activation)
            input_dim = hidden_size
        value_layers.append(nn.Linear(input_dim, 1))
        self.value_head = nn.Sequential(*value_layers)
        
        # Initialize output layer
        nn.init.orthogonal_(self.value_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.value_head[-1].bias)
    
    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation to latent distribution.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            
        Returns:
            mu: Latent mean [batch_size, latent_dim]
            log_std: Latent log std [batch_size, latent_dim]
        """
        h = self.encoder(obs)
        mu = self.fc_mu(h)
        log_std = self.fc_log_std(h)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Numerical stability
        return mu, log_std
    
    def reparameterize(self, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation."""
        return self.decoder(z)
    
    def forward(self, obs: torch.Tensor, return_latent: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Forward pass.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            return_latent: If True, return latent representation and VAE loss
            
        Returns:
            If return_latent=False: values [batch_size, 1]
            If return_latent=True: (values, dict with mu, log_std, z, recon_loss, kl_loss)
        """
        mu, log_std = self.encode(obs)
        z = self.reparameterize(mu, log_std)
        values = self.value_head(z)
        
        if return_latent:
            # Compute VAE losses
            recon = self.decode(z)
            recon_loss = F.mse_loss(recon, obs, reduction='mean')
            
            # KL divergence: -0.5 * sum(1 + log_std^2 - mu^2 - exp(log_std)^2)
            kl_loss = -0.5 * torch.sum(1 + 2 * log_std - mu.pow(2) - log_std.exp().pow(2), dim=1).mean()
            
            return values, {
                "mu": mu,
                "log_std": log_std,
                "z": z,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "vae_loss": recon_loss + self.beta * kl_loss,
            }
        
        return values
    
    def get_latent_representation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get deterministic latent representation (using mean).
        
        Args:
            obs: Observations [batch_size, obs_dim]
            
        Returns:
            Latent representation [batch_size, latent_dim]
        """
        mu, _ = self.encode(obs)
        return mu

