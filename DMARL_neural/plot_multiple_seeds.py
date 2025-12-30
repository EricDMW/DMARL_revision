# ============================================================================
# File: plot_multiple_seeds.py
# Description: Plot training curves with confidence intervals from multiple seeds
# ============================================================================
"""
Plot Multiple Seeds Results

This script aggregates training results from multiple random seeds and creates
publication-quality plots with mean and standard deviation (shaded confidence intervals).

Features:
- Academic top conference style plots
- Mean curve with shaded confidence intervals (std)
- Smooth curves with optional moving average
- Multiple metrics (rewards, losses)
- High-resolution output suitable for papers
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Tuple
import glob

# Set matplotlib style for academic papers
matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,  # TrueType fonts for PDF
    'ps.fonttype': 42,
})


def load_results_from_seed(seed_dir: str) -> Dict:
    """
    Load training results from a seed directory
    
    Args:
        seed_dir: Directory containing training results for one seed
        
    Returns:
        Dictionary with training statistics, or None if not found
    """
    results_file = os.path.join(seed_dir, "training_results.json")
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    
    return None


def aggregate_results(results_dir: str, seeds: List[int]) -> Dict:
    """
    Aggregate results from multiple seeds
    
    Args:
        results_dir: Directory containing seed subdirectories
        seeds: List of seed values
        
    Returns:
        Dictionary with aggregated statistics
    """
    all_rewards = []
    all_critic_losses = []
    all_actor_losses = []
    all_lengths = []
    
    valid_seeds = []
    
    for seed in seeds:
        seed_dir = os.path.join(results_dir, f"seed_{seed}")
        results = load_results_from_seed(seed_dir)
        
        if results and 'episode_rewards' in results:
            all_rewards.append(results['episode_rewards'])
            if 'critic_losses' in results:
                all_critic_losses.append(results['critic_losses'])
            if 'actor_losses' in results:
                all_actor_losses.append(results['actor_losses'])
            if 'episode_lengths' in results:
                all_lengths.append(results['episode_lengths'])
            valid_seeds.append(seed)
    
    if len(all_rewards) == 0:
        raise ValueError(f"No valid results found in {results_dir}")
    
    # Find minimum length to align all runs
    min_length = min(len(r) for r in all_rewards)
    
    # Truncate all to same length
    all_rewards = [r[:min_length] for r in all_rewards]
    if all_critic_losses:
        all_critic_losses = [l[:min_length] for l in all_critic_losses]
    if all_actor_losses:
        all_actor_losses = [l[:min_length] for l in all_actor_losses]
    if all_lengths:
        all_lengths = [l[:min_length] for l in all_lengths]
    
    # Convert to numpy arrays
    rewards_array = np.array(all_rewards)  # [n_seeds, n_episodes]
    
    # Compute statistics
    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)
    
    aggregated = {
        'episodes': np.arange(min_length),
        'mean_rewards': mean_rewards,
        'std_rewards': std_rewards,
        'all_rewards': rewards_array,
        'valid_seeds': valid_seeds,
        'n_seeds': len(valid_seeds)
    }
    
    if all_critic_losses:
        critic_array = np.array(all_critic_losses)
        aggregated['mean_critic_losses'] = np.mean(critic_array, axis=0)
        aggregated['std_critic_losses'] = np.std(critic_array, axis=0)
    
    if all_actor_losses:
        actor_array = np.array(all_actor_losses)
        aggregated['mean_actor_losses'] = np.mean(actor_array, axis=0)
        aggregated['std_actor_losses'] = np.std(actor_array, axis=0)
    
    if all_lengths:
        lengths_array = np.array(all_lengths)
        aggregated['mean_lengths'] = np.mean(lengths_array, axis=0)
        aggregated['std_lengths'] = np.std(lengths_array, axis=0)
    
    return aggregated


def plot_training_curves_with_std(
    aggregated: Dict,
    output_path: str,
    window: int = 100,
    show_individual: bool = False
):
    """
    Plot training curves with confidence intervals (academic style)
    
    Args:
        aggregated: Aggregated results dictionary
        output_path: Path to save the figure
        window: Moving average window size
        show_individual: Whether to show individual seed curves (faded)
    """
    episodes = aggregated['episodes']
    mean_rewards = aggregated['mean_rewards']
    std_rewards = aggregated['std_rewards']
    n_seeds = aggregated['n_seeds']
    
    # Determine number of subplots
    n_plots = 1
    has_critic = 'mean_critic_losses' in aggregated
    has_actor = 'mean_actor_losses' in aggregated
    
    if has_critic:
        n_plots += 1
    if has_actor:
        n_plots += 1
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    # Color palette (academic style - professional colors)
    # Using colors suitable for academic papers and colorblind-friendly
    colors = {
        'reward': '#2E86AB',      # Blue (primary)
        'reward_fill': '#2E86AB', # Same blue for fill (with alpha)
        'critic': '#06A77D',      # Green
        'actor': '#F18F01',       # Orange
    }
    
    plot_idx = 0
    
    # ========== Plot 1: Episode Rewards ==========
    ax = axes[plot_idx]
    
    # Apply moving average if window is specified
    if window > 1 and len(mean_rewards) >= window:
        # Moving average for mean
        kernel = np.ones(window) / window
        mean_smooth = np.convolve(mean_rewards, kernel, mode='valid')
        std_smooth = np.convolve(std_rewards, kernel, mode='valid')
        episodes_smooth = episodes[window-1:]
        
        # Plot individual runs (faded) if requested
        if show_individual:
            for seed_idx, rewards in enumerate(aggregated['all_rewards']):
                rewards_smooth = np.convolve(rewards, kernel, mode='valid')
                ax.plot(episodes_smooth, rewards_smooth, 
                       alpha=0.15, linewidth=0.5, color=colors['reward'])
        
        # Plot mean with confidence interval
        ax.plot(episodes_smooth, mean_smooth, 
               label=f'Mean (n={n_seeds})', 
               color=colors['reward'], 
               linewidth=2.5)
        ax.fill_between(episodes_smooth, 
                       mean_smooth - std_smooth,
                       mean_smooth + std_smooth,
                       alpha=0.3, 
                       color=colors['reward'],
                       label='±1 Std')
    else:
        # Plot without smoothing
        if show_individual:
            for seed_idx, rewards in enumerate(aggregated['all_rewards']):
                ax.plot(episodes, rewards, 
                       alpha=0.15, linewidth=0.5, color=colors['reward'])
        
        ax.plot(episodes, mean_rewards, 
               label=f'Mean (n={n_seeds})', 
               color=colors['reward'], 
               linewidth=2.5)
        ax.fill_between(episodes, 
                       mean_rewards - std_rewards,
                       mean_rewards + std_rewards,
                       alpha=0.3, 
                       color=colors['reward'],
                       label='±1 Std')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Training Progress: Episode Rewards', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plot_idx += 1
    
    # ========== Plot 2: Critic Loss (if available) ==========
    if has_critic:
        ax = axes[plot_idx]
        mean_critic = aggregated['mean_critic_losses']
        std_critic = aggregated['std_critic_losses']
        
        if window > 1 and len(mean_critic) >= window:
            kernel = np.ones(window) / window
            mean_smooth = np.convolve(mean_critic, kernel, mode='valid')
            std_smooth = np.convolve(std_critic, kernel, mode='valid')
            episodes_smooth = episodes[window-1:]
            
            ax.plot(episodes_smooth, mean_smooth, 
                   color=colors['critic'], linewidth=2.5)
            ax.fill_between(episodes_smooth, 
                           mean_smooth - std_smooth,
                           mean_smooth + std_smooth,
                           alpha=0.3, color=colors['critic'])
        else:
            ax.plot(episodes, mean_critic, 
                   color=colors['critic'], linewidth=2.5)
            ax.fill_between(episodes, 
                           mean_critic - std_critic,
                           mean_critic + std_critic,
                           alpha=0.3, color=colors['critic'])
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Critic Loss', fontsize=12)
        ax.set_title('Training Progress: Critic Loss', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plot_idx += 1
    
    # ========== Plot 3: Actor Loss (if available) ==========
    if has_actor:
        ax = axes[plot_idx]
        mean_actor = aggregated['mean_actor_losses']
        std_actor = aggregated['std_actor_losses']
        
        if window > 1 and len(mean_actor) >= window:
            kernel = np.ones(window) / window
            mean_smooth = np.convolve(mean_actor, kernel, mode='valid')
            std_smooth = np.convolve(std_actor, kernel, mode='valid')
            episodes_smooth = episodes[window-1:]
            
            ax.plot(episodes_smooth, mean_smooth, 
                   color=colors['actor'], linewidth=2.5)
            ax.fill_between(episodes_smooth, 
                           mean_smooth - std_smooth,
                           mean_smooth + std_smooth,
                           alpha=0.3, color=colors['actor'])
        else:
            ax.plot(episodes, mean_actor, 
                   color=colors['actor'], linewidth=2.5)
            ax.fill_between(episodes, 
                           mean_actor - std_actor,
                           mean_actor + std_actor,
                           alpha=0.3, color=colors['actor'])
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Actor Loss', fontsize=12)
        ax.set_title('Training Progress: Actor Loss', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {output_path}")
    plt.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Plot aggregated results from multiple seeds'
    )
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing seed subdirectories')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots (default: results_dir)')
    parser.add_argument('--seeds', type=int, nargs='+', required=True,
                       help='List of seed values used')
    parser.add_argument('--window', type=int, default=100,
                       help='Moving average window size')
    parser.add_argument('--show_individual', action='store_true',
                       help='Show individual seed curves (faded)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Aggregating results from multiple seeds...")
    aggregated = aggregate_results(args.results_dir, args.seeds)
    
    print(f"Found {aggregated['n_seeds']} valid seeds: {aggregated['valid_seeds']}")
    print(f"Episodes per run: {len(aggregated['episodes'])}")
    
    # Generate plot
    output_path = os.path.join(args.output_dir, 'training_curves_with_std.png')
    plot_training_curves_with_std(
        aggregated, 
        output_path,
        window=args.window,
        show_individual=args.show_individual
    )
    
    # Save aggregated statistics
    stats_path = os.path.join(args.output_dir, 'aggregated_stats.json')
    stats = {
        'n_seeds': aggregated['n_seeds'],
        'valid_seeds': aggregated['valid_seeds'],
        'n_episodes': len(aggregated['episodes']),
        'final_mean_reward': float(aggregated['mean_rewards'][-1]),
        'final_std_reward': float(aggregated['std_rewards'][-1]),
        'max_mean_reward': float(np.max(aggregated['mean_rewards'])),
        'mean_rewards': aggregated['mean_rewards'].tolist(),
        'std_rewards': aggregated['std_rewards'].tolist()
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved aggregated statistics to: {stats_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
