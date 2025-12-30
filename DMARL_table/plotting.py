"""
Plotting Utilities for NMARL Algorithm.

This module provides visualization tools for training progress,
agent performance, and algorithm analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
import os


def set_plot_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'lines.linewidth': 2,
        'figure.dpi': 100
    })


def smooth_curve(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing to a curve."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='valid')
    # Pad to original length
    pad_size = len(values) - len(smoothed)
    return np.concatenate([values[:pad_size], smoothed])


def plot_training_rewards(stats: Dict, save_path: Optional[str] = None,
                         title: str = "Training Rewards", 
                         smooth_window: int = 20):
    """
    Plot training rewards over episodes.
    
    Args:
        stats: Training statistics dictionary
        save_path: Path to save the figure
        title: Plot title
        smooth_window: Window size for smoothing
    """
    set_plot_style()
    
    rewards = np.array(stats['episode_rewards'])
    episodes = np.arange(1, len(rewards) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Raw rewards with transparency
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Rewards')
    
    # Smoothed rewards
    smoothed = smooth_curve(rewards, smooth_window)
    ax.plot(episodes, smoothed, color='blue', linewidth=2, 
            label=f'Smoothed (window={smooth_window})')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax


def plot_comparison(stats_dict: Dict[str, Dict], save_path: Optional[str] = None,
                   title: str = "Algorithm Comparison", smooth_window: int = 20):
    """
    Plot comparison of multiple algorithms.
    
    Args:
        stats_dict: Dictionary mapping algorithm names to their stats
        save_path: Path to save the figure
        title: Plot title
        smooth_window: Window size for smoothing
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use distinct colors for NMARL and IQL
    color_map = {
        'NMARL': '#1f77b4',
        'NMARL (Ours)': '#1f77b4',
        'IQL': '#ff7f0e',
        'IQL Baseline': '#ff7f0e'
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (name, stats) in enumerate(stats_dict.items()):
        rewards = np.array(stats['episode_rewards'])
        episodes = np.arange(1, len(rewards) + 1)
        
        # Get color
        color = color_map.get(name, colors[idx % len(colors)])
        
        # Raw with low alpha
        ax.plot(episodes, rewards, alpha=0.15, color=color)
        
        # Smoothed
        smoothed = smooth_curve(rewards, smooth_window)
        ax.plot(episodes, smoothed, color=color, linewidth=2.5, label=name)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax


def plot_local_rewards(stats: Dict, save_path: Optional[str] = None,
                      n_agents_to_show: int = 4, smooth_window: int = 20):
    """
    Plot local rewards for individual agents.
    
    Args:
        stats: Training statistics
        save_path: Path to save the figure
        n_agents_to_show: Number of agents to display
        smooth_window: Smoothing window
    """
    set_plot_style()
    
    local_rewards = stats.get('local_rewards', {})
    if not local_rewards:
        print("No local rewards data available")
        return None, None
    
    n_agents = min(len(local_rewards), n_agents_to_show)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_agents))
    
    for i in range(n_agents):
        rewards = np.array(local_rewards[i])
        episodes = np.arange(1, len(rewards) + 1)
        
        ax = axes[i]
        ax.plot(episodes, rewards, alpha=0.3, color=colors[i])
        smoothed = smooth_curve(rewards, smooth_window)
        ax.plot(episodes, smoothed, color=colors[i], linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Local Reward')
        ax.set_title(f'Agent {i}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Local Rewards per Agent', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def plot_gradient_norms(stats: Dict, save_path: Optional[str] = None,
                       smooth_window: int = 20):
    """
    Plot gradient norms over training.
    
    Args:
        stats: Training statistics
        save_path: Path to save the figure
        smooth_window: Smoothing window
    """
    set_plot_style()
    
    grad_norms = np.array(stats.get('gradient_norms', []))
    if len(grad_norms) == 0:
        print("No gradient norm data available")
        return None, None
    
    episodes = np.arange(1, len(grad_norms) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(episodes, grad_norms, alpha=0.3, color='green')
    smoothed = smooth_curve(grad_norms, smooth_window)
    ax.plot(episodes, smoothed, color='green', linewidth=2, label='Mean Gradient Norm')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Policy Gradient Norms During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax


def plot_q_values(stats: Dict, save_path: Optional[str] = None):
    """
    Plot Q-value statistics over training.
    
    Args:
        stats: Training statistics
        save_path: Path to save the figure
    """
    set_plot_style()
    
    q_values = stats.get('q_values', [])
    if not q_values:
        print("No Q-value data available")
        return None, None
    
    episodes = np.arange(1, len(q_values) + 1)
    mean_q = [q['mean'] for q in q_values]
    max_q = [q['max'] for q in q_values]
    min_q = [q['min'] for q in q_values]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(episodes, mean_q, color='blue', label='Mean Q')
    ax.fill_between(episodes, min_q, max_q, alpha=0.2, color='blue')
    ax.plot(episodes, max_q, '--', color='green', alpha=0.7, label='Max Q')
    ax.plot(episodes, min_q, '--', color='red', alpha=0.7, label='Min Q')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q-Value')
    ax.set_title('Q-Value Estimates During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax


def plot_kappa_analysis(results: Dict[Tuple[int, int], Dict], 
                       save_path: Optional[str] = None):
    """
    Plot analysis of different (κ_o, κ_r) configurations.
    
    Args:
        results: Dictionary mapping (kappa_o, kappa_r) to training stats
        save_path: Path to save the figure
    """
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract final performance for each configuration
    configs = list(results.keys())
    final_rewards = []
    
    for config in configs:
        stats = results[config]
        # Mean of last 100 episodes
        final_rewards.append(np.mean(stats['episode_rewards'][-100:]))
    
    # Bar plot of final performance
    ax1 = axes[0]
    x = np.arange(len(configs))
    labels = [f"κo={c[0]}, κr={c[1]}" for c in configs]
    bars = ax1.bar(x, final_rewards, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(configs))))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Final Mean Reward')
    ax1.set_title('Final Performance by Configuration')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Learning curves
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(configs)))
    
    for (config, stats), color in zip(results.items(), colors):
        rewards = np.array(stats['episode_rewards'])
        smoothed = smooth_curve(rewards, 20)
        label = f"κo={config[0]}, κr={config[1]}"
        ax2.plot(smoothed, color=color, label=label)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Learning Curves by Configuration')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def plot_comprehensive_summary(stats: Dict, config_name: str = "NMARL",
                              save_path: Optional[str] = None):
    """
    Create a comprehensive summary plot with multiple panels.
    
    Args:
        stats: Training statistics
        config_name: Name for the configuration
        save_path: Path to save the figure
    """
    set_plot_style()
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 1. Training rewards (top left, spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    rewards = np.array(stats['episode_rewards'])
    episodes = np.arange(1, len(rewards) + 1)
    ax1.plot(episodes, rewards, alpha=0.3, color='blue')
    smoothed = smooth_curve(rewards, 20)
    ax1.plot(episodes, smoothed, color='blue', linewidth=2, label='Smoothed Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title(f'{config_name} - Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode lengths (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    lengths = np.array(stats.get('episode_lengths', []))
    if len(lengths) > 0:
        ax2.plot(episodes[:len(lengths)], lengths, alpha=0.3, color='orange')
        smoothed_len = smooth_curve(lengths, 20)
        ax2.plot(episodes[:len(lengths)], smoothed_len, color='orange', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Lengths')
    ax2.grid(True, alpha=0.3)
    
    # 3. Gradient norms (middle right)
    ax3 = fig.add_subplot(gs[1, 1])
    grad_norms = np.array(stats.get('gradient_norms', []))
    if len(grad_norms) > 0:
        ax3.plot(episodes[:len(grad_norms)], grad_norms, alpha=0.3, color='green')
        smoothed_grad = smooth_curve(grad_norms, 20)
        ax3.plot(episodes[:len(grad_norms)], smoothed_grad, color='green', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Policy Gradient Norms')
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-values (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    q_values = stats.get('q_values', [])
    if q_values:
        q_episodes = np.arange(1, len(q_values) + 1)
        mean_q = [q['mean'] for q in q_values]
        ax4.plot(q_episodes, mean_q, color='purple', linewidth=2, label='Mean Q')
        ax4.fill_between(q_episodes, 
                        [q['min'] for q in q_values],
                        [q['max'] for q in q_values],
                        alpha=0.2, color='purple')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Q-Value')
    ax4.set_title('Q-Value Estimates')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Reward distribution (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(rewards, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax5.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(rewards):.2f}')
    ax5.set_xlabel('Total Reward')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Reward Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'{config_name} Training Summary', fontsize=18, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def create_all_plots(stats: Dict, output_dir: str, prefix: str = "nmarl"):
    """
    Create and save all standard plots.
    
    Args:
        stats: Training statistics
        output_dir: Directory to save plots
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Training rewards
    plot_training_rewards(stats, 
                         save_path=os.path.join(output_dir, f"{prefix}_rewards.png"))
    
    # Gradient norms
    plot_gradient_norms(stats,
                       save_path=os.path.join(output_dir, f"{prefix}_gradients.png"))
    
    # Q-values
    plot_q_values(stats,
                 save_path=os.path.join(output_dir, f"{prefix}_qvalues.png"))
    
    # Local rewards
    plot_local_rewards(stats,
                      save_path=os.path.join(output_dir, f"{prefix}_local_rewards.png"))
    
    # Comprehensive summary
    plot_comprehensive_summary(stats, config_name=prefix.upper(),
                              save_path=os.path.join(output_dir, f"{prefix}_summary.png"))
    
    print(f"\nAll plots saved to {output_dir}")
    
    # Close all figures to free memory
    plt.close('all')