"""
Minigrid environment wrapper with ground-truth representation extraction.
"""

import gymnasium as gym
import numpy as np
import torch

# Import minigrid to register environments with gymnasium
# This must be imported before gym.make() is called
import minigrid  # noqa: F401


class MinigridWrapper:
    """
    Wrapper for Minigrid environments that provides:
    - Standardized interface
    - Ground-truth causal representation extraction
    - State/action space normalization
    """
    
    def __init__(self, env_name: str, seed: int | None = None, keep_image_format: bool = False):
        """
        Initialize Minigrid environment.
        
        Args:
            env_name: Environment name (e.g., "MiniGrid-Unlock-v0")
            seed: Random seed
            keep_image_format: If True, keep observations as [H, W, C] tensors (for CNN).
                              If False, flatten to [H*W*C] (default, for MLP policies).
        """
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.keep_image_format = keep_image_format
        
        if seed is not None:
            self.env.reset(seed=seed)
        
        # Get observation and action spaces
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Minigrid uses Dict observation space with 'image', 'direction', 'mission'
        if isinstance(self.obs_space, gym.spaces.Dict):
            # Extract image observation from dict
            if 'image' in self.obs_space.spaces:
                image_space = self.obs_space.spaces['image']
                if isinstance(image_space, gym.spaces.Box):
                    # Image observation: [height, width, channels]
                    self.obs_shape = image_space.shape  # (H, W, C)
                    self.obs_dim = np.prod(self.obs_shape)
                else:
                    raise ValueError(f"Unexpected image space type: {image_space}")
            else:
                raise ValueError(f"Dict observation space missing 'image' key: {self.obs_space}")
        elif isinstance(self.obs_space, gym.spaces.Box):
            # Direct Box observation (for compatibility)
            self.obs_shape = self.obs_space.shape
            self.obs_dim = np.prod(self.obs_shape)
        else:
            raise ValueError(f"Unexpected observation space: {self.obs_space}")
        
        self.action_dim = self.action_space.n  # Discrete actions
    
    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        """
        Reset environment.
        
        Returns:
            obs: Flattened observation tensor
            info: Info dict
        """
        obs, info = self.env.reset(seed=seed)
        obs_tensor = self._process_obs(obs)
        return obs_tensor, info
    
    def step(self, action: int | torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        """
        Step environment.
        
        Args:
            action: Action (int or tensor)
            
        Returns:
            obs: Next observation
            reward: Reward
            terminated: Episode terminated
            truncated: Episode truncated
            info: Info dict
        """
        if isinstance(action, torch.Tensor):
            action = action.item()
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_tensor = self._process_obs(obs)
        return obs_tensor, reward, terminated, truncated, info
    
    def _process_obs(self, obs: np.ndarray | dict) -> torch.Tensor:
        """
        Process observation to tensor.
        
        Returns:
            If keep_image_format=True: [H, W, C] tensor (normalized to [0, 1])
            If keep_image_format=False: [H*W*C] flattened tensor (normalized to [0, 1])
        """
        # Handle Dict observation space (minigrid default)
        if isinstance(obs, dict):
            if 'image' in obs:
                obs = obs['image']
            else:
                raise ValueError(f"Dict observation missing 'image' key: {obs.keys()}")
        
        # Normalize to [0, 1]
        obs_normalized = obs.astype(np.float32) / 255.0
        
        if self.keep_image_format:
            # Keep as image format [H, W, C] for CNN processing
            return torch.from_numpy(obs_normalized)
        else:
            # Flatten image observation [H, W, C] -> [H*W*C] for MLP policies
            obs_flat = obs_normalized.flatten()
            return torch.from_numpy(obs_flat)
    
    def get_ground_truth_representation(self, obs: np.ndarray | torch.Tensor = None) -> torch.Tensor:
        """
        Extract ground-truth causal representation from environment grid state.
        
        For Minigrid, this includes:
        - Agent position (x, y) - normalized to [0, 1]
        - Agent direction (one-hot: 0=right, 1=down, 2=left, 3=up)
        - Key position (x, y) if exists, else (-1, -1)
        - Door position (x, y) if exists, else (-1, -1)
        - Goal position (x, y)
        
        Args:
            obs: Optional observation (used to determine device if torch.Tensor)
            
        Returns:
            Ground-truth representation vector [agent_x, agent_y, dir_0, dir_1, dir_2, dir_3, 
                                                key_x, key_y, door_x, door_y, goal_x, goal_y]
        """
        # Determine device from observation if provided
        device = None
        if isinstance(obs, torch.Tensor):
            device = obs.device
        
        # Unwrap environment to get actual MiniGridEnv (handles OrderEnforcing and other wrappers)
        env = self.env
        while hasattr(env, 'unwrapped'):
            if env == env.unwrapped:
                break
            env = env.unwrapped
        # Also try accessing .env attribute (for wrappers that don't have .unwrapped)
        while hasattr(env, 'env') and env.env != env:
            env = env.env
        
        # Access underlying grid from unwrapped environment
        grid = env.grid
        width = env.width
        height = env.height
        
        # Agent state
        agent_pos = env.agent_pos
        agent_dir = env.agent_dir
        
        # Normalize positions to [0, 1]
        agent_x = agent_pos[0] / width
        agent_y = agent_pos[1] / height
        
        # Agent direction (one-hot encoding) - create on same device as obs
        if device is not None:
            direction_onehot = torch.zeros(4, device=device)
        else:
            direction_onehot = torch.zeros(4)
        direction_onehot[agent_dir] = 1.0
        
        # Find objects in grid
        if device is not None:
            key_pos = torch.tensor([-1.0, -1.0], dtype=torch.float32, device=device)  # Default: not found
            door_pos = torch.tensor([-1.0, -1.0], dtype=torch.float32, device=device)
            goal_pos = torch.tensor([-1.0, -1.0], dtype=torch.float32, device=device)
        else:
            key_pos = torch.tensor([-1.0, -1.0], dtype=torch.float32)  # Default: not found
            door_pos = torch.tensor([-1.0, -1.0], dtype=torch.float32)
            goal_pos = torch.tensor([-1.0, -1.0], dtype=torch.float32)
        
        for i in range(width):
            for j in range(height):
                cell = grid.get(i, j)
                if cell is None:
                    continue
                
                # Check cell type
                if hasattr(cell, 'type'):
                    if device is not None:
                        pos_tensor = torch.tensor([i / width, j / height], dtype=torch.float32, device=device)
                    else:
                        pos_tensor = torch.tensor([i / width, j / height], dtype=torch.float32)
                    
                    if cell.type == 'key':
                        key_pos = pos_tensor
                    elif cell.type == 'door':
                        door_pos = pos_tensor
                    elif cell.type == 'goal':
                        goal_pos = pos_tensor
        
        # Concatenate representation - ensure all tensors are on same device
        if device is not None:
            agent_tensor = torch.tensor([agent_x, agent_y], dtype=torch.float32, device=device)
        else:
            agent_tensor = torch.tensor([agent_x, agent_y], dtype=torch.float32)
        
        representation = torch.cat([
            agent_tensor,
            direction_onehot,
            key_pos,
            door_pos,
            goal_pos,
        ])
        
        return representation
    
    def close(self):
        """Close environment."""
        self.env.close()
    
    def render(self, mode: str = "human"):
        """Render environment."""
        return self.env.render(mode=mode)


class MinigridColorAugWrapper:
    """
    Non-causal observation change: permute RGB channels in the image observation.

    Grid topology and get_ground_truth_representation are unchanged (Exp 5 transfer).
    """

    def __init__(self, base: MinigridWrapper, color_perm_seed: int = 0):
        self.base = base
        self.env_name = base.env_name
        self.obs_shape = base.obs_shape
        self.obs_dim = base.obs_dim
        self.action_dim = base.action_dim
        self.keep_image_format = base.keep_image_format
        rng = np.random.default_rng(color_perm_seed)
        self.color_perm = rng.permutation(3)

    def _permute_colors(self, obs: torch.Tensor) -> torch.Tensor:
        H, W, C = self.obs_shape
        if self.keep_image_format:
            return obs[..., self.color_perm]
        img = obs.view(H, W, C)
        return img[..., self.color_perm].reshape(-1)

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        obs, info = self.base.reset(seed=seed)
        return self._permute_colors(obs), info

    def step(self, action: int | torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.base.step(action)
        return self._permute_colors(obs), reward, terminated, truncated, info

    def get_ground_truth_representation(self, obs: np.ndarray | torch.Tensor = None) -> torch.Tensor:
        return self.base.get_ground_truth_representation(obs)

    def close(self):
        self.base.close()

    def render(self, mode: str = "human"):
        return self.base.render(mode=mode)

