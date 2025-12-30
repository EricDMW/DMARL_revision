# ============================================================================
# File: plot_algorithm_comparison.py
# Description: Compare multiple algorithms with multiple seeds each
# ============================================================================
"""
Algorithm Comparison Plotting

This script compares multiple algorithms (e.g., DMARL vs VDN) by aggregating
results from multiple seeds for each algorithm and creating publication-quality
comparison plots with confidence intervals.
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
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def load_results_from_seed(seed_dir: str) -> Dict:
    """Load training results from a seed directory"""
    results_file = os.path.join(seed_dir, "training_results.json")
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    
    return None


def aggregate_algorithm_results(results_dir: str, seeds: List[int]) -> Dict:
    """
    Aggregate results from multiple seeds for one algorithm
    
    Args:
        results_dir: Directory containing seed subdirectories
        seeds: List of seed values
        
    Returns:
        Dictionary with aggregated statistics
    """
    all_rewards = []
    valid_seeds = []
    
    for seed in seeds:
        seed_dir = os.path.join(results_dir, f"seed_{seed}")
        results = load_results_from_seed(seed_dir)
        
        if results and 'episode_rewards' in results:
            all_rewards.append(results['episode_rewards'])
            valid_seeds.append(seed)
    
    if len(all_rewards) == 0:
        return None
    
    # Find minimum length to align all runs
    min_length = min(len(r) for r in all_rewards)
    all_rewards = [r[:min_length] for r in all_rewards]
    
    # Convert to numpy arrays
    rewards_array = np.array(all_rewards)  # [n_seeds, n_episodes]
    
    # Compute statistics
    mean_rewards = np.mean(rewards_array, axis=0)
    std_rewards = np.std(rewards_array, axis=0)
    
    return {
        'episodes': np.arange(min_length),
        'mean_rewards': mean_rewards,
        'std_rewards': std_rewards,
        'all_rewards': rewards_array,
        'n_seeds': len(valid_seeds),
        'valid_seeds': valid_seeds
    }


def plot_algorithm_comparison(
    algorithm_results: Dict[str, Dict],
    output_path: str,
    window: int = 100,
    ylabel: str = 'Episode Reward',
    title: str = 'Algorithm Comparison'
):
    """
    Plot comparison of multiple algorithms with confidence intervals
    
    Args:
        algorithm_results: Dictionary mapping algorithm names to aggregated results
        output_path: Path to save the figure
        window: Moving average window size
        ylabel: Y-axis label
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color palette (colorblind-friendly, academic style)
    colors = [
        '#2E86AB',  # Blue
        '#A23B72',  # Purple
        '#06A77D',  # Green
        '#F18F01',  # Orange
        '#D62828',  # Red
        '#6A4C93',  # Indigo
    ]
    
    linestyles = ['-', '--', '-.', ':']
    
    for idx, (algo_name, results) in enumerate(algorithm_results.items()):
        episodes = results['episodes']
        mean_rewards = results['mean_rewards']
        std_rewards = results['std_rewards']
        n_seeds = results['n_seeds']
        
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        
        # Apply moving average if window is specified
        if window > 1 and len(mean_rewards) >= window:
            kernel = np.ones(window) / window
            mean_smooth = np.convolve(mean_rewards, kernel, mode='valid')
            std_smooth = np.convolve(std_rewards, kernel, mode='valid')
            episodes_smooth = episodes[window-1:]
            
            # Plot mean with confidence interval
            ax.plot(episodes_smooth, mean_smooth, 
                   label=f'{algo_name} (n={n_seeds})', 
                   color=color, 
                   linestyle=linestyle,
                   linewidth=2.5)
            ax.fill_between(episodes_smooth, 
                           mean_smooth - std_smooth,
                           mean_smooth + std_smooth,
                           alpha=0.2, 
                           color=color)
        else:
            # Plot without smoothing
            ax.plot(episodes, mean_rewards, 
                   label=f'{algo_name} (n={n_seeds})', 
                   color=color,
                   linestyle=linestyle,
                   linewidth=2.5)
            ax.fill_between(episodes, 
                           mean_rewards - std_rewards,
                           mean_rewards + std_rewards,
                           alpha=0.2, 
                           color=color)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved comparison plot to: {output_path}")
    plt.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Compare multiple algorithms with multiple seeds'
    )
    parser.add_argument('--algorithm_dirs', type=str, nargs='+', required=True,
                       help='List of result directories for each algorithm')
    parser.add_argument('--algorithm_names', type=str, nargs='+', required=True,
                       help='List of algorithm names (same order as dirs)')
    parser.add_argument('--seeds', type=int, nargs='+', required=True,
                       help='List of seed values used (same for all algorithms)')
    parser.add_argument('--output_path', type=str, default='algorithm_comparison.png',
                       help='Path to save comparison plot')
    parser.add_argument('--window', type=int, default=100,
                       help='Moving average window size')
    parser.add_argument('--ylabel', type=str, default='Episode Reward',
                       help='Y-axis label')
    parser.add_argument('--title', type=str, default='Algorithm Comparison',
                       help='Plot title')
    
    args = parser.parse_args()
    
    if len(args.algorithm_dirs) != len(args.algorithm_names):
        raise ValueError("Number of algorithm directories must match number of algorithm names")
    
    # Aggregate results for each algorithm
    algorithm_results = {}
    
    for algo_name, algo_dir in zip(args.algorithm_names, args.algorithm_dirs):
        print(f"Aggregating results for {algo_name}...")
        results = aggregate_algorithm_results(algo_dir, args.seeds)
        
        if results is None:
            print(f"Warning: No valid results found for {algo_name} in {algo_dir}")
            continue
        
        algorithm_results[algo_name] = results
        print(f"  Found {results['n_seeds']} valid seeds: {results['valid_seeds']}")
    
    if len(algorithm_results) == 0:
        raise ValueError("No valid results found for any algorithm")
    
    # Generate comparison plot
    plot_algorithm_comparison(
        algorithm_results,
        args.output_path,
        window=args.window,
        ylabel=args.ylabel,
        title=args.title
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
