"""
Plotting script for training and evaluation metrics.

Reads CSV files from logs directory and generates visualization plots.
Supports plotting multiple experiments on the same plots for comparison.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Color palette for multiple experiments
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def load_metrics(csv_path: Path) -> pd.DataFrame:
    """
    Load metrics from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with metrics
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, na_values=['', 'nan', 'NaN', 'None'])
    
    # Replace empty strings with NaN (in case some weren't caught)
    df = df.replace('', np.nan)
    
    # Convert numeric columns, handling empty strings
    for col in df.columns:
        if col != 'eval_action_distribution':  # Skip string columns
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare training data by filtering and sorting by step."""
    df_train = df.dropna(subset=['step']).copy()
    df_train['step'] = pd.to_numeric(df_train['step'], errors='coerce')
    df_train = df_train[df_train['step'].notna()]
    df_train = df_train.sort_values('step').reset_index(drop=True)
    return df_train


def plot_training_metrics(
    experiments: List[Tuple[pd.DataFrame, str]], 
    output_dir: Path, 
    title_suffix: str = "Comparison"
):
    """
    Plot training metrics (losses, entropy, episode returns) for multiple experiments.
    
    Args:
        experiments: List of (DataFrame, experiment_name) tuples
        output_dir: Directory to save plots
        title_suffix: Suffix for plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Metrics - {title_suffix}', fontsize=14, fontweight='bold')
    
    # Compute global x-axis range across all experiments
    all_steps = []
    for df, _ in experiments:
        df_train = prepare_training_data(df)
        if len(df_train) > 0:
            all_steps.extend(df_train['step'].tolist())
    if all_steps:
        global_x_min = min(all_steps)
        global_x_max = max(all_steps)
        # Add small padding (1% on each side) to ensure full range is visible
        x_range = global_x_max - global_x_min
        global_x_min = max(0, global_x_min - x_range * 0.01)
        global_x_max = global_x_max + x_range * 0.01
        print(f"Training metrics: Setting global x-axis range to [{global_x_min:.0f}, {global_x_max:.0f}]")
    else:
        global_x_min = 0
        global_x_max = 1
    
    # 1. Losses overview
    ax = axes[0, 0]
    for i, (df, exp_name) in enumerate(experiments):
        df_train = prepare_training_data(df)
        color = COLORS[i % len(COLORS)]
        
        if 'policy_loss' in df_train.columns:
            policy_loss = pd.to_numeric(df_train['policy_loss'], errors='coerce')
            valid_mask = policy_loss.notna()
            if valid_mask.sum() > 0:
                ax.plot(df_train.loc[valid_mask, 'step'], policy_loss.loc[valid_mask], 
                       label=f'{exp_name} - Policy', alpha=0.7, linewidth=1.2, color=color, linestyle='-')
        
        if 'value_loss' in df_train.columns:
            value_loss = pd.to_numeric(df_train['value_loss'], errors='coerce')
            valid_mask = value_loss.notna()
            if valid_mask.sum() > 0:
                ax.plot(df_train.loc[valid_mask, 'step'], value_loss.loc[valid_mask], 
                       label=f'{exp_name} - Value', alpha=0.7, linewidth=1.2, color=color, linestyle='--')
        
        if 'contrastive_loss' in df_train.columns:
            contrastive_loss = pd.to_numeric(df_train['contrastive_loss'], errors='coerce')
            valid_mask = contrastive_loss.notna()
            if valid_mask.sum() > 0:
                ax.plot(df_train.loc[valid_mask, 'step'], contrastive_loss.loc[valid_mask], 
                       label=f'{exp_name} - Contrastive', alpha=0.7, linewidth=1.2, color=color, linestyle=':')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses (see detailed plot)')
    ax.set_xlim(global_x_min, global_x_max)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. Entropy
    ax = axes[0, 1]
    for i, (df, exp_name) in enumerate(experiments):
        df_train = prepare_training_data(df)
        color = COLORS[i % len(COLORS)]
        
        if 'entropy' in df_train.columns:
            entropy = pd.to_numeric(df_train['entropy'], errors='coerce')
            valid_mask = entropy.notna()
            if valid_mask.sum() > 0:
                ax.plot(df_train.loc[valid_mask, 'step'], entropy.loc[valid_mask], 
                       label=exp_name, alpha=0.7, linewidth=1.5, color=color)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.set_xlim(global_x_min, global_x_max)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 3. Episode Returns
    ax = axes[1, 0]
    for i, (df, exp_name) in enumerate(experiments):
        df_train = prepare_training_data(df)
        color = COLORS[i % len(COLORS)]
        
        if 'mean_episode_return' in df_train.columns:
            returns = pd.to_numeric(df_train['mean_episode_return'], errors='coerce')
            valid_mask = returns.notna()
            if valid_mask.sum() > 0:
                ax.plot(df_train.loc[valid_mask, 'step'], returns.loc[valid_mask], 
                       label=exp_name, alpha=0.7, linewidth=1.5, color=color)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Episode Return')
    ax.set_title('Training Episode Returns')
    ax.set_xlim(global_x_min, global_x_max)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. Episode Length
    ax = axes[1, 1]
    for i, (df, exp_name) in enumerate(experiments):
        df_train = prepare_training_data(df)
        color = COLORS[i % len(COLORS)]
        
        if 'mean_episode_length' in df_train.columns:
            lengths = pd.to_numeric(df_train['mean_episode_length'], errors='coerce')
            valid_mask = lengths.notna()
            if valid_mask.sum() > 0:
                ax.plot(df_train.loc[valid_mask, 'step'], lengths.loc[valid_mask], 
                       label=exp_name, alpha=0.7, linewidth=1.5, color=color)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Episode Length')
    ax.set_title('Training Episode Lengths')
    ax.set_xlim(global_x_min, global_x_max)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'training_metrics_{title_suffix.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training metrics plot to {output_path}")
    plt.close()

    # Additional detailed loss plots (each loss in its own subplot)
    loss_plots = [
        ("policy_loss", "Policy Loss", False),
        ("value_loss", "Value Loss", True),
        ("contrastive_loss", "Contrastive Loss", True),
    ]
    fig_losses, loss_axes = plt.subplots(1, len(loss_plots), figsize=(5.5 * len(loss_plots), 4))
    fig_losses.suptitle(f'Loss Components - {title_suffix}', fontsize=14, fontweight='bold')
    
    for ax, (column, title, use_log) in zip(loss_axes, loss_plots):
        for i, (df, exp_name) in enumerate(experiments):
            df_train = prepare_training_data(df)
            color = COLORS[i % len(COLORS)]
            
            if column in df_train.columns:
                series = pd.to_numeric(df_train[column], errors='coerce')
                valid_mask = series.notna()
                if valid_mask.sum() > 0:
                    y = series.loc[valid_mask]
                    x = df_train.loc[valid_mask, 'step']
                    
                    if use_log:
                        positive_mask = y > 0
                        if positive_mask.sum() > 0:
                            ax.plot(x.loc[positive_mask], y.loc[positive_mask], 
                                   label=exp_name, linewidth=1.5, color=color, alpha=0.7)
                            ax.set_yscale('log')
                    else:
                        ax.plot(x, y, label=exp_name, linewidth=1.5, color=color, alpha=0.7)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xlim(global_x_min, global_x_max)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    losses_output_path = output_dir / f'loss_breakdown_{title_suffix.lower().replace(" ", "_")}.png'
    plt.savefig(losses_output_path, dpi=150, bbox_inches='tight')
    print(f"Saved loss breakdown plot to {losses_output_path}")
    plt.close(fig_losses)


def plot_evaluation_metrics(
    experiments: List[Tuple[pd.DataFrame, str]], 
    output_dir: Path, 
    title_suffix: str = "Comparison"
):
    """
    Plot evaluation metrics (reward mean/std/max, episode length) for multiple experiments.
    
    Args:
        experiments: List of (DataFrame, experiment_name) tuples
        output_dir: Directory to save plots
        title_suffix: Suffix for plot title
    """
    # Filter experiments that have evaluation data
    experiments_with_eval = []
    for df, exp_name in experiments:
        df_eval = df[df['eval_reward_mean'].notna() & (df['eval_reward_mean'] != '')]
        if len(df_eval) > 0:
            experiments_with_eval.append((df, exp_name))
    
    if len(experiments_with_eval) == 0:
        print("No evaluation data found in any CSV files.")
        return
    
    # Compute global x-axis range across all evaluation data
    all_eval_steps = []
    for df, _ in experiments_with_eval:
        df_eval = df[df['eval_reward_mean'].notna() & (df['eval_reward_mean'] != '')]
        if 'step' in df_eval.columns:
            df_eval['step'] = pd.to_numeric(df_eval['step'], errors='coerce')
            valid_steps = df_eval[df_eval['step'].notna()]['step']
            if len(valid_steps) > 0:
                all_eval_steps.extend(valid_steps.tolist())
    
    if all_eval_steps:
        global_eval_x_min = min(all_eval_steps)
        global_eval_x_max = max(all_eval_steps)
        # Add small padding (1% on each side)
        x_range = global_eval_x_max - global_eval_x_min
        global_eval_x_min = max(0, global_eval_x_min - x_range * 0.01)
        global_eval_x_max = global_eval_x_max + x_range * 0.01
        print(f"Evaluation metrics: Setting global x-axis range to [{global_eval_x_min:.0f}, {global_eval_x_max:.0f}]")
    else:
        global_eval_x_min = None
        global_eval_x_max = None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Evaluation Metrics - {title_suffix}', fontsize=14, fontweight='bold')
    
    # Determine x-axis label from first experiment (they should all use the same)
    x_label = 'Training Steps'  # Default
    for df, _ in experiments_with_eval:
        df_eval = df[df['eval_reward_mean'].notna() & (df['eval_reward_mean'] != '')]
        if 'epoch' in df_eval.columns and df_eval['epoch'].notna().any():
            x_label = 'Epoch'
            break
        elif 'step' in df_eval.columns and df_eval['step'].notna().any():
            x_label = 'Training Steps'
            break
    
    # Helper to get x-axis for evaluation data
    def get_eval_x_axis(df_eval):
        x_axis_series = None
        if 'epoch' in df_eval.columns:
            x_axis_series = df_eval['epoch']
        elif 'step' in df_eval.columns:
            x_axis_series = df_eval['step']
        elif 'total_steps' in df_eval.columns:
            x_axis_series = df_eval['total_steps']
        
        if x_axis_series is None:
            x_axis_series = pd.Series(range(len(df_eval)), index=df_eval.index)
        
        x_axis_raw = pd.to_numeric(
            x_axis_series.ffill().bfill(),
            errors='coerce'
        )
        if x_axis_raw.isna().all():
            x_axis_raw = pd.Series(range(len(df_eval)), index=df_eval.index, dtype=float)
        
        return x_axis_raw
    
    # 1. Reward Mean with Std Error Bars
    ax = axes[0, 0]
    for i, (df, exp_name) in enumerate(experiments_with_eval):
        df_eval = df[df['eval_reward_mean'].notna() & (df['eval_reward_mean'] != '')].copy()
        x_axis_raw = get_eval_x_axis(df_eval)
        color = COLORS[i % len(COLORS)]
        
        reward_mean = pd.to_numeric(df_eval['eval_reward_mean'], errors='coerce')
        reward_std = pd.to_numeric(df_eval.get('eval_reward_std', pd.Series(index=df_eval.index)), errors='coerce')
        
        valid_mask = reward_mean.notna() & x_axis_raw.notna()
        if valid_mask.any():
            x_axis = x_axis_raw.loc[valid_mask]
            reward_mean_aligned = reward_mean.loc[valid_mask]
            reward_std_aligned = reward_std.loc[valid_mask].fillna(0.0)
            
            ax.errorbar(x_axis, reward_mean_aligned, yerr=reward_std_aligned, 
                       fmt='o-', capsize=3, capthick=1.5, alpha=0.7, linewidth=1.5,
                       label=exp_name, color=color, markersize=4)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('Reward')
    ax.set_title('Evaluation Reward (Mean ± Std)')
    if global_eval_x_min is not None and global_eval_x_max is not None:
        ax.set_xlim(global_eval_x_min, global_eval_x_max)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Reward Max
    ax = axes[0, 1]
    for i, (df, exp_name) in enumerate(experiments_with_eval):
        df_eval = df[df['eval_reward_mean'].notna() & (df['eval_reward_mean'] != '')].copy()
        x_axis_raw = get_eval_x_axis(df_eval)
        color = COLORS[i % len(COLORS)]
        
        if 'eval_reward_max' in df_eval.columns:
            reward_max = pd.to_numeric(df_eval['eval_reward_max'], errors='coerce').dropna()
            reward_mean = pd.to_numeric(df_eval['eval_reward_mean'], errors='coerce')
            valid_mask = reward_mean.notna() & x_axis_raw.notna()
            reward_max_idx = reward_max.index.intersection(x_axis_raw.loc[valid_mask].index)
            
            if len(reward_max_idx) > 0:
                ax.plot(x_axis_raw.loc[reward_max_idx], reward_max.loc[reward_max_idx], 
                       's-', color=color, alpha=0.7, linewidth=1.5, markersize=5,
                       label=exp_name)
        else:
            # Fallback: plot mean
            reward_mean = pd.to_numeric(df_eval['eval_reward_mean'], errors='coerce')
            valid_mask = reward_mean.notna() & x_axis_raw.notna()
            if valid_mask.any():
                ax.plot(x_axis_raw.loc[valid_mask], reward_mean.loc[valid_mask], 
                       'o-', color=color, alpha=0.7, linewidth=1.5, markersize=4,
                       label=exp_name)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('Reward')
    ax.set_title('Evaluation Reward (Max)')
    if global_eval_x_min is not None and global_eval_x_max is not None:
        ax.set_xlim(global_eval_x_min, global_eval_x_max)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 3. Episode Length
    ax = axes[1, 0]
    for i, (df, exp_name) in enumerate(experiments_with_eval):
        df_eval = df[df['eval_reward_mean'].notna() & (df['eval_reward_mean'] != '')].copy()
        x_axis_raw = get_eval_x_axis(df_eval)
        color = COLORS[i % len(COLORS)]
        
        if 'eval_episode_length' in df_eval.columns:
            eval_length = pd.to_numeric(df_eval['eval_episode_length'], errors='coerce').dropna()
            reward_mean = pd.to_numeric(df_eval['eval_reward_mean'], errors='coerce')
            valid_mask = reward_mean.notna() & x_axis_raw.notna()
            eval_length_idx = eval_length.index.intersection(x_axis_raw.loc[valid_mask].index)
            
            if len(eval_length_idx) > 0:
                ax.plot(x_axis_raw.loc[eval_length_idx], eval_length.loc[eval_length_idx], 
                       'o-', color=color, alpha=0.7, linewidth=1.5, markersize=4,
                       label=exp_name)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('Episode Length')
    ax.set_title('Evaluation Episode Length')
    if global_eval_x_min is not None and global_eval_x_max is not None:
        ax.set_xlim(global_eval_x_min, global_eval_x_max)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. Combined Reward View (Mean lines only, no error bars for clarity)
    ax = axes[1, 1]
    for i, (df, exp_name) in enumerate(experiments_with_eval):
        df_eval = df[df['eval_reward_mean'].notna() & (df['eval_reward_mean'] != '')].copy()
        x_axis_raw = get_eval_x_axis(df_eval)
        color = COLORS[i % len(COLORS)]
        
        reward_mean = pd.to_numeric(df_eval['eval_reward_mean'], errors='coerce')
        valid_mask = reward_mean.notna() & x_axis_raw.notna()
        if valid_mask.any():
            x_axis = x_axis_raw.loc[valid_mask]
            reward_mean_aligned = reward_mean.loc[valid_mask]
            ax.plot(x_axis, reward_mean_aligned, 'o-', color=color, alpha=0.7, 
                   linewidth=1.5, markersize=4, label=exp_name)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('Reward')
    ax.set_title('Evaluation Reward (Mean)')
    if global_eval_x_min is not None and global_eval_x_max is not None:
        ax.set_xlim(global_eval_x_min, global_eval_x_max)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'evaluation_metrics_{title_suffix.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved evaluation metrics plot to {output_path}")
    plt.close()


def plot_representation_metrics(
    experiments: List[Tuple[pd.DataFrame, str]], 
    output_dir: Path, 
    title_suffix: str = "Comparison"
):
    """
    Plot representation-specific metrics (contrastive loss, representation distance) for multiple experiments.
    
    Args:
        experiments: List of (DataFrame, experiment_name) tuples
        output_dir: Directory to save plots
        title_suffix: Suffix for plot title
    """
    # Check which experiments have representation metrics
    experiments_with_repr = []
    for df, exp_name in experiments:
        df_train = prepare_training_data(df)
        has_contrastive = 'contrastive_loss' in df_train.columns
        has_repr_dist = 'representation_distance' in df_train.columns
        if has_contrastive or has_repr_dist:
            experiments_with_repr.append((df, exp_name))
    
    if len(experiments_with_repr) == 0:
        return
    
    # Compute global x-axis range across all experiments
    all_steps = []
    for df, _ in experiments_with_repr:
        df_train = prepare_training_data(df)
        if len(df_train) > 0:
            all_steps.extend(df_train['step'].tolist())
    if all_steps:
        global_x_min = min(all_steps)
        global_x_max = max(all_steps)
        # Add small padding (1% on each side) to ensure full range is visible
        x_range = global_x_max - global_x_min
        global_x_min = max(0, global_x_min - x_range * 0.01)
        global_x_max = global_x_max + x_range * 0.01
    else:
        global_x_min = 0
        global_x_max = 1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Representation Metrics - {title_suffix}', fontsize=14, fontweight='bold')
    
    # 1. Contrastive Loss
    ax = axes[0]
    has_any_contrastive = False
    for i, (df, exp_name) in enumerate(experiments_with_repr):
        df_train = prepare_training_data(df)
        color = COLORS[i % len(COLORS)]
        
        if 'contrastive_loss' in df_train.columns:
            contrastive_loss = pd.to_numeric(df_train['contrastive_loss'], errors='coerce')
            valid_mask = contrastive_loss.notna()
            if valid_mask.sum() > 0:
                ax.plot(df_train.loc[valid_mask, 'step'], contrastive_loss.loc[valid_mask], 
                       label=exp_name, alpha=0.7, linewidth=1.5, color=color)
                has_any_contrastive = True
    
    if has_any_contrastive:
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Contrastive Loss')
        ax.set_title('Contrastive Loss (Forward Model)')
        ax.set_yscale('log')
        ax.set_xlim(global_x_min, global_x_max)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No contrastive loss data', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Contrastive Loss (Forward Model)')
    
    # 2. Representation Distance
    ax = axes[1]
    has_any_repr_dist = False
    for i, (df, exp_name) in enumerate(experiments_with_repr):
        df_train = prepare_training_data(df)
        color = COLORS[i % len(COLORS)]
        
        if 'representation_distance' in df_train.columns:
            repr_dist = pd.to_numeric(df_train['representation_distance'], errors='coerce')
            valid_mask = repr_dist.notna()
            if valid_mask.sum() > 0:
                ax.plot(df_train.loc[valid_mask, 'step'], repr_dist.loc[valid_mask], 
                       label=exp_name, alpha=0.7, linewidth=1.5, color=color)
                has_any_repr_dist = True
    
    if has_any_repr_dist:
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Representation Distance')
        ax.set_title('Representation-Space Distance')
        ax.set_xlim(global_x_min, global_x_max)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No representation distance data', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Representation-Space Distance')
    
    plt.tight_layout()
    output_path = output_dir / f'representation_metrics_{title_suffix.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved representation metrics plot to {output_path}")
    plt.close()


def find_all_experiments(logs_dir: Path) -> List[Tuple[Path, str]]:
    """
    Find all experiment CSV files in the logs directory.
    
    Args:
        logs_dir: Directory containing CSV files
        
    Returns:
        List of (csv_path, experiment_name) tuples
    """
    experiments = []
    for csv_file in logs_dir.glob('*_metrics.csv'):
        exp_name = csv_file.stem.replace('_metrics', '')
        experiments.append((csv_file, exp_name))
    return sorted(experiments)


def plot_all_metrics(
    experiments: List[Tuple[Path, str]], 
    output_dir: Optional[Path] = None,
    title_suffix: Optional[str] = None
):
    """
    Generate all plots for multiple experiments.
    
    Args:
        experiments: List of (csv_path, experiment_name) tuples
        output_dir: Directory to save plots (default: plots/)
        title_suffix: Suffix for plot titles (default: "Comparison")
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'plots'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if title_suffix is None:
        if len(experiments) == 1:
            title_suffix = experiments[0][1]
        else:
            title_suffix = "Comparison"
    
    # Load all experiments
    loaded_experiments = []
    for csv_path, exp_name in experiments:
        print(f"Loading metrics from {csv_path}")
        try:
            df = load_metrics(csv_path)
            print(f"  Loaded {len(df)} rows for {exp_name}")
            loaded_experiments.append((df, exp_name))
        except Exception as e:
            print(f"  Warning: Failed to load {csv_path}: {e}")
    
    if len(loaded_experiments) == 0:
        print("No experiments loaded successfully.")
        return
    
    print(f"\nGenerating plots for {len(loaded_experiments)} experiment(s)...")
    plot_training_metrics(loaded_experiments, output_dir, title_suffix)
    plot_evaluation_metrics(loaded_experiments, output_dir, title_suffix)
    plot_representation_metrics(loaded_experiments, output_dir, title_suffix)
    
    print(f"\nAll plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot training and evaluation metrics from CSV files. '
                    'Can plot single experiment or compare multiple experiments.'
    )
    parser.add_argument(
        'csv_files',
        type=str,
        nargs='*',
        help='Path(s) to CSV metrics file(s). If not provided, will auto-discover all *_metrics.csv files in logs/'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: plots/)'
    )
    parser.add_argument(
        '--experiment-names',
        type=str,
        nargs='*',
        default=None,
        help='Names for experiments (default: inferred from filenames). Must match number of CSV files.'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Title suffix for plots (default: "Comparison" for multiple, experiment name for single)'
    )
    parser.add_argument(
        '--logs-dir',
        type=str,
        default='logs',
        help='Directory to search for CSV files if none provided (default: logs/)'
    )
    
    args = parser.parse_args()
    
    # Determine which experiments to plot
    experiments = []
    
    if len(args.csv_files) == 0:
        # Auto-discover experiments
        logs_dir = Path(args.logs_dir)
        if not logs_dir.is_absolute():
            logs_dir = Path(__file__).parent.parent / logs_dir
        
        if not logs_dir.exists():
            print(f"Error: Logs directory not found: {logs_dir}", file=sys.stderr)
            sys.exit(1)
        
        experiments = find_all_experiments(logs_dir)
        print(f"Auto-discovered {len(experiments)} experiment(s) in {logs_dir}")
    else:
        # Use provided CSV files
        for i, csv_file in enumerate(args.csv_files):
            csv_path = Path(csv_file)
            if not csv_path.is_absolute():
                # Try relative to current directory, then relative to project root
                if not csv_path.exists():
                    csv_path = Path(__file__).parent.parent / csv_path
            
            if not csv_path.exists():
                print(f"Warning: CSV file not found: {csv_path}", file=sys.stderr)
                continue
            
            exp_name = args.experiment_names[i] if args.experiment_names and i < len(args.experiment_names) else None
            if exp_name is None:
                exp_name = csv_path.stem.replace('_metrics', '')
            
            experiments.append((csv_path, exp_name))
    
    if len(experiments) == 0:
        print("Error: No experiments to plot.", file=sys.stderr)
        sys.exit(1)
    
    try:
        plot_all_metrics(experiments, args.output_dir, args.title)
    except Exception as e:
        print(f"Error generating plots: {e}", file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
