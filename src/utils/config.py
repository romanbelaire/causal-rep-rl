"""
Configuration management using JSON files.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration manager for experiments."""
    
    def __init__(self, config_path: str | Path | Dict[str, Any]):
        """
        Initialize configuration from file or dict.
        
        Args:
            config_path: Path to JSON config file or dict with config
        """
        if isinstance(config_path, dict):
            self.config = config_path
        else:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        self._validate()
    
    def _validate(self):
        """Basic validation of required config keys."""
        required_keys = ['experiment', 'environment', 'architecture', 'algorithm', 'training']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation (e.g., 'experiment.name')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Get config value."""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self.config
    
    def save(self, path: str | Path):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        def deep_update(base: dict, updates: dict):
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
        
        deep_update(self.config, updates)

