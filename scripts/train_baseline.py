#!/usr/bin/env python3
"""
Script to train baseline PPO agent.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import train

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train baseline PPO agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minigrid_config.json",
        help="Path to config JSON file",
    )
    args = parser.parse_args()
    
    train(args.config)

