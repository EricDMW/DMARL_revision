# ============================================================================
# File: plot_ablation_results.py
# Description: Plot comparison results from ablation studies
# ============================================================================
"""
Plot Ablation Study Results

This script loads training results from ablation experiments and creates
comparison plots showing the effect of different κ_o and κ_r values.
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import glob

from config import Config


def load_training_results(results_dir: str) -> Dict:
    """
    Load training results from a results directory
    
    Args:
        results_dir: Directory containing training results
        
    Returns:
        Dictionary with training statistics
    """
    # Try to load from checkpoint directory
    checkpoint_dir = os.path.join(results_dir, "best_model")
    
    # For now, we'll need to extract from training logs or saved files
    # This is a placeholder - you may need to adjust based on how results are saved
    training_curves_path = os.path.join(results_dir, "training_curves.png")
    
    # Extract parameters from directory name
    # Format: kappa_o_{obs}_kappa_r_{reward}
    dir_name = os.path.basename(results_dir)
    parts = dir_name.split("_")
    
    kappa_o = None
    kappa_r = None
    
    for i, part in enumerate(parts):
        if part == "o" and i > 0:
            try:
                kappa_o = int(parts[i-1].split("kappa")[-1] if "kappa" in parts[i-1] else parts[i+1])
            except:
                pass
        if part == "r" and i > 0:
            try:
                kappa_r = int(parts[i+1])
            except:
                pass
    
    # Alternative: try to parse from directory name more directly
    if "kappa_o_" in dir_name and "kappa_r_" in dir_name:
        try:
            kappa_o = int(dir_name.split("kappa_o_")[1].split("_")[0])
            kappa_r = int(dir_name.split("kappa_r_")[1].split("/")[0])
        except:
            pass
    
    return {
        'kappa_o': kappa_o,
        'kappa_r': kappa_r,
        'dir': results_dir
    }


def load_episode_rewards_from_log(log_file: str, max_episodes: int) -> List[float]:
    """
    Extract episode rewards from training log file
    
    Args:
        log_file: Path to training log file
        max_episodes: Maximum number of episodes
        
    Returns:
        List of episode rewards
    """
    rewards = []
    
    if not os.path.exists(log_file):
        return rewards
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for reward information in progress bar or log lines
                if 'Reward:' in line or 'Average Reward:' in line:
                    # Try to extract reward value
                    parts = line.split()
                    for i, part in enumerate(parts):
                        try:
                            reward = float(part)
                            if -1000 < reward < 10000:  # Reasonable range
                                rewards.append(reward)
                                break
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Warning: Could not parse log file {log_file}: {e}")
    
    return rewards


def load_results_from_checkpoint(checkpoint_dir: str) -> Dict:
    """
    Load training results from checkpoint directory
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        Dictionary with training statistics
    """
    # Check if there's a saved results file
    results_file = os.path.join(checkpoint_dir, "training_results.json")
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    
    return {}


def plot_ablation1_comparison(ablation1_dir: str, output_dir: str):
    """
    Plot comparison for Ablation 1: Varying observation neighbors
    
    Args:
        ablation1_dir: Directory containing ablation 1 results
        output_dir: Directory to save plots
    """
    experiments = {}
    
    # Find all experiment directories
    for exp_dir in glob.glob(os.path.join(ablation1_dir, "kappa_o_*")):
        dir_name = os.path.basename(exp_dir)
        
        # Extract kappa_o and kappa_r from directory name
        try:
            kappa_o = int(dir_name.split("kappa_o_")[1].split("_")[0])
            kappa_r = int(dir_name.split("kappa_r_")[1])
            
            # Try to load results from JSON file first
            results = load_results_from_checkpoint(exp_dir)
            
            if results and 'episode_rewards' in results:
                rewards = results['episode_rewards']
                experiments[kappa_o] = {
                    'rewards': rewards,
                    'kappa_r': kappa_r
                }
            else:
                # Fallback to log file parsing
                log_file = os.path.join(exp_dir, "training.log")
                rewards = load_episode_rewards_from_log(log_file, 5000)
                
                if len(rewards) > 0:
                    experiments[kappa_o] = {
                        'rewards': rewards,
                        'kappa_r': kappa_r
                    }
        except Exception as e:
            print(f"Warning: Could not process {exp_dir}: {e}")
    
    if len(experiments) == 0:
        print(f"Warning: No valid experiments found in {ablation1_dir}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    window = 100
    
    for i, (kappa_o, data) in enumerate(sorted(experiments.items())):
        rewards = data['rewards']
        kappa_r = data['kappa_r']
        
        if len(rewards) >= window:
            # Moving average
            moving_avg = np.convolve(
                rewards,
                np.ones(window) / window,
                mode='valid'
            )
            episodes = range(window - 1, len(rewards))
            ax.plot(episodes, moving_avg, 
                   label=f'κ_o={kappa_o}, κ_r={kappa_r}',
                   color=colors[i % len(colors)],
                   linewidth=2)
        else:
            ax.plot(rewards, 
                   label=f'κ_o={kappa_o}, κ_r={kappa_r}',
                   color=colors[i % len(colors)],
                   alpha=0.5,
                   linewidth=1)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward (Moving Average)', fontsize=12)
    ax.set_title('Ablation 1: Effect of Observation Neighborhood (κ_o)\nFixed κ_r=1', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ablation1_obs_neighbors_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def plot_ablation2_comparison(ablation2_dir: str, output_dir: str):
    """
    Plot comparison for Ablation 2: Varying reward neighbors
    
    Args:
        ablation2_dir: Directory containing ablation 2 results
        output_dir: Directory to save plots
    """
    experiments = {}
    
    # Find all experiment directories
    for exp_dir in glob.glob(os.path.join(ablation2_dir, "kappa_o_*")):
        dir_name = os.path.basename(exp_dir)
        
        # Extract kappa_o and kappa_r from directory name
        try:
            kappa_o = int(dir_name.split("kappa_o_")[1].split("_")[0])
            kappa_r = int(dir_name.split("kappa_r_")[1])
            
            # Try to load results from JSON file first
            results = load_results_from_checkpoint(exp_dir)
            
            if results and 'episode_rewards' in results:
                rewards = results['episode_rewards']
                experiments[kappa_r] = {
                    'rewards': rewards,
                    'kappa_o': kappa_o
                }
            else:
                # Fallback to log file parsing
                log_file = os.path.join(exp_dir, "training.log")
                rewards = load_episode_rewards_from_log(log_file, 5000)
                
                if len(rewards) > 0:
                    experiments[kappa_r] = {
                        'rewards': rewards,
                        'kappa_o': kappa_o
                    }
        except Exception as e:
            print(f"Warning: Could not process {exp_dir}: {e}")
    
    if len(experiments) == 0:
        print(f"Warning: No valid experiments found in {ablation2_dir}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    window = 100
    
    for i, (kappa_r, data) in enumerate(sorted(experiments.items())):
        rewards = data['rewards']
        kappa_o = data['kappa_o']
        
        if len(rewards) >= window:
            # Moving average
            moving_avg = np.convolve(
                rewards,
                np.ones(window) / window,
                mode='valid'
            )
            episodes = range(window - 1, len(rewards))
            ax.plot(episodes, moving_avg, 
                   label=f'κ_o={kappa_o}, κ_r={kappa_r}',
                   color=colors[i % len(colors)],
                   linewidth=2)
        else:
            ax.plot(rewards, 
                   label=f'κ_o={kappa_o}, κ_r={kappa_r}',
                   color=colors[i % len(colors)],
                   alpha=0.5,
                   linewidth=1)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward (Moving Average)', fontsize=12)
    ax.set_title('Ablation 2: Effect of Reward Neighborhood (κ_r)\nFixed κ_o=2', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ablation2_reward_neighbors_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Plot ablation study results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing ablation results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots (default: results_dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot ablation 1 results
    ablation1_dir = os.path.join(args.results_dir, "ablation1_obs_neighbors")
    if os.path.exists(ablation1_dir):
        print("Plotting Ablation 1 results...")
        plot_ablation1_comparison(ablation1_dir, args.output_dir)
    else:
        print(f"Warning: Ablation 1 directory not found: {ablation1_dir}")
    
    # Plot ablation 2 results
    ablation2_dir = os.path.join(args.results_dir, "ablation2_reward_neighbors")
    if os.path.exists(ablation2_dir):
        print("Plotting Ablation 2 results...")
        plot_ablation2_comparison(ablation2_dir, args.output_dir)
    else:
        print(f"Warning: Ablation 2 directory not found: {ablation2_dir}")
    
    print("\nPlotting completed!")


if __name__ == "__main__":
    main()

