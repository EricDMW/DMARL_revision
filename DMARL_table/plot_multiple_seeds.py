# ============================================================================
# File: plot_multiple_seeds.py
# Description: Plot training curves with confidence intervals from multiple seeds
# ============================================================================
"""
Plot Multiple Seeds Results for NMARL

This script aggregates training results from multiple random seeds and creates
publication-quality plots with mean and standard deviation (shaded confidence intervals).

Features:
- Academic top conference style plots
- Mean curve with shaded confidence intervals (std)
- Shows both NMARL and IQL baseline in same graph
- Optional: Individual seed runs for transparency
- Smooth curves with optional moving average
- High-resolution output suitable for papers
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Tuple, Optional
import glob

# Set matplotlib style for academic papers (top conference style)
matplotlib.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.7,
    'grid.alpha': 0.25,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,  # TrueType fonts for PDF
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
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


def smooth_curve(values: np.ndarray, window: int = 20) -> np.ndarray:
    """Apply moving average smoothing to a curve"""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='valid')
    # Pad to original length
    pad_size = len(values) - len(smoothed)
    return np.concatenate([values[:pad_size], smoothed])


def aggregate_results(results_dir: str, seeds: List[int], algorithm: str = 'nmarl') -> Dict:
    """
    Aggregate results from multiple seeds
    
    Args:
        results_dir: Directory containing seed subdirectories
        seeds: List of seed values
        algorithm: 'nmarl' or 'iql'
        
    Returns:
        Dictionary with aggregated statistics
    """
    all_rewards = []
    all_lengths = []
    all_gradient_norms = []
    
    valid_seeds = []
    all_rewards_raw = []  # Store raw (unsmoothed) for individual plotting
    
    for seed in seeds:
        seed_dir = os.path.join(results_dir, f"seed_{seed}")
        results = load_results_from_seed(seed_dir)
        
        if results and algorithm in results:
            algo_data = results[algorithm]
            if 'episode_rewards' in algo_data:
                rewards = algo_data['episode_rewards']
                all_rewards.append(rewards)
                all_rewards_raw.append(rewards)
                if 'episode_lengths' in algo_data:
                    all_lengths.append(algo_data['episode_lengths'])
                if 'gradient_norms' in algo_data:
                    all_gradient_norms.append(algo_data['gradient_norms'])
                valid_seeds.append(seed)
    
    if not all_rewards:
        return None
    
    # Find minimum length to align all runs
    min_length = min(len(r) for r in all_rewards)
    
    # Truncate all to same length
    all_rewards = [r[:min_length] for r in all_rewards]
    all_rewards_raw = [r[:min_length] for r in all_rewards_raw]
    all_lengths = [l[:min_length] for l in all_lengths] if all_lengths else []
    all_gradient_norms = [g[:min_length] for g in all_gradient_norms] if all_gradient_norms else []
    
    # Convert to numpy arrays
    rewards_array = np.array(all_rewards)
    lengths_array = np.array(all_lengths) if all_lengths else None
    grad_norms_array = np.array(all_gradient_norms) if all_gradient_norms else None
    
    # Compute statistics
    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)
    
    mean_lengths = np.mean(lengths_array, axis=0) if lengths_array is not None else None
    std_lengths = np.std(lengths_array, axis=0) if lengths_array is not None else None
    
    mean_grad_norms = np.mean(grad_norms_array, axis=0) if grad_norms_array is not None else None
    std_grad_norms = np.std(grad_norms_array, axis=0) if grad_norms_array is not None else None
    
    return {
        'mean_rewards': mean_rewards,
        'std_rewards': std_rewards,
        'mean_lengths': mean_lengths,
        'std_lengths': std_lengths,
        'mean_grad_norms': mean_grad_norms,
        'std_grad_norms': std_grad_norms,
        'all_rewards': rewards_array,
        'all_rewards_raw': all_rewards_raw,  # For individual seed plotting
        'valid_seeds': valid_seeds,
        'n_seeds': len(valid_seeds)
    }


def plot_training_curves_with_std(results_dir: str, output_dir: str, seeds: List[int],
                                  smooth_window: int = 50, show_individual_seeds: bool = True):
    """
    Plot training curves with confidence intervals for both NMARL and IQL
    
    Args:
        results_dir: Directory containing seed subdirectories
        output_dir: Directory to save plots
        seeds: List of seed values
        smooth_window: Window size for smoothing
        show_individual_seeds: Whether to show individual seed runs (light lines)
    """
    # Aggregate results for both algorithms
    nmarl_results = aggregate_results(results_dir, seeds, 'nmarl')
    iql_results = aggregate_results(results_dir, seeds, 'iql')
    
    if nmarl_results is None:
        print("Warning: No NMARL results found!")
        return
    
    # Create figure with single plot (top conference style - clean and focused)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    episodes = np.arange(1, len(nmarl_results['mean_rewards']) + 1)
    
    # Color scheme for top conference style
    nmarl_color = '#2E86AB'  # Professional blue
    iql_color = '#A23B72'    # Professional purple/magenta
    
    # Plot individual seed runs (if requested) - very light for transparency
    if show_individual_seeds:
        # NMARL individual seeds
        for seed_rewards in nmarl_results['all_rewards_raw']:
            smoothed = smooth_curve(np.array(seed_rewards), smooth_window)
            ax.plot(episodes, smoothed, color=nmarl_color, alpha=0.15, linewidth=0.8, zorder=1)
        
        # IQL individual seeds
        if iql_results is not None:
            for seed_rewards in iql_results['all_rewards_raw']:
                smoothed = smooth_curve(np.array(seed_rewards), smooth_window)
                ax.plot(episodes, smoothed, color=iql_color, alpha=0.15, linewidth=0.8, zorder=1)
    
    # Plot NMARL: Mean curve with std shadow
    nmarl_mean = smooth_curve(nmarl_results['mean_rewards'], smooth_window)
    nmarl_std = smooth_curve(nmarl_results['std_rewards'], smooth_window)
    
    # Shadow area (std)
    ax.fill_between(episodes, nmarl_mean - nmarl_std, nmarl_mean + nmarl_std,
                    color=nmarl_color, alpha=0.25, zorder=2, label=f'NMARL ±1 std (n={nmarl_results["n_seeds"]})')
    
    # Mean curve (prominent)
    ax.plot(episodes, nmarl_mean, color=nmarl_color, linewidth=2.5, 
            zorder=3, label='NMARL (Ours)')
    
    # Plot IQL baseline: Mean curve with std shadow
    if iql_results is not None:
        iql_mean = smooth_curve(iql_results['mean_rewards'], smooth_window)
        iql_std = smooth_curve(iql_results['std_rewards'], smooth_window)
        
        # Shadow area (std)
        ax.fill_between(episodes, iql_mean - iql_std, iql_mean + iql_std,
                        color=iql_color, alpha=0.25, zorder=2, 
                        label=f'IQL ±1 std (n={iql_results["n_seeds"]})')
        
        # Mean curve (prominent)
        ax.plot(episodes, iql_mean, color=iql_color, linewidth=2.5, 
                zorder=3, label='IQL Baseline')
    
    # Styling for top conference
    ax.set_xlabel('Episode', fontsize=12, fontweight='normal')
    ax.set_ylabel('Total Reward', fontsize=12, fontweight='normal')
    ax.legend(loc='best', frameon=True, fancybox=False, shadow=False, 
              edgecolor='black', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    ax.set_xlim(left=0)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'training_curves_with_std.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {output_path}")
    
    # Also save as PDF for publications
    pdf_path = os.path.join(output_dir, 'training_curves_with_std.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to {pdf_path}")
    
    plt.close()
    
    # Create separate plot for standard deviation over time
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    
    # Plot std curves
    ax2.plot(episodes, nmarl_std, color=nmarl_color, linewidth=2.5, 
             label='NMARL Std', zorder=2)
    
    if iql_results is not None:
        ax2.plot(episodes, iql_std, color=iql_color, linewidth=2.5, 
                 label='IQL Std', zorder=2)
    
    ax2.set_xlabel('Episode', fontsize=12, fontweight='normal')
    ax2.set_ylabel('Standard Deviation', fontsize=12, fontweight='normal')
    ax2.set_title('Standard Deviation Over Training', fontsize=13, fontweight='normal')
    ax2.legend(loc='best', frameon=True, fancybox=False, shadow=False,
               edgecolor='black', framealpha=0.9, fontsize=10)
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    ax2.set_xlim(left=0)
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save std plot
    std_output_path = os.path.join(output_dir, 'std_over_training.png')
    plt.savefig(std_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Std plot saved to {std_output_path}")
    
    plt.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Plot multi-seed training results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing seed subdirectories')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots (default: same as results_dir)')
    parser.add_argument('--seeds', type=int, nargs='+', required=True,
                       help='List of seed values')
    parser.add_argument('--smooth_window', type=int, default=50,
                       help='Window size for smoothing (default: 50)')
    parser.add_argument('--hide_individual', action='store_true',
                       help='Hide individual seed runs (only show mean and std)')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    plot_training_curves_with_std(
        args.results_dir,
        output_dir,
        args.seeds,
        args.smooth_window,
        show_individual_seeds=not args.hide_individual
    )


if __name__ == "__main__":
    main()
