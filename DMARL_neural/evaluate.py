# ============================================================================
# File: evaluate.py
# Description: Evaluation functions and utilities
# ============================================================================
"""
Evaluation Module for DMARL

This module provides:
- Model evaluation
- Performance metrics computation
- Visualization utilities
- Result analysis
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os
from tqdm import tqdm

from config import Config
from dmarl import DMARLSystem
import env_lib


class Evaluator:
    """
    Evaluator class for DMARL models
    
    Provides evaluation and analysis tools for trained models.
    
    Attributes:
        config: Configuration object
        env: Environment instance
        system: DMARLSystem instance (loaded model)
    """
    
    def __init__(self, config: Config, model_path: Optional[str] = None):
        """
        Initialize evaluator
        
        Args:
            config: Configuration object
            model_path: Path to saved model (optional)
        """
        self.config = config
        
        # Create environment
        self.env = env_lib.WirelessCommEnv(
            grid_x=config.grid_x,
            grid_y=config.grid_y,
            ddl=config.ddl,
            n_obs_neighbors=config.n_obs_neighbors,
            max_iter=config.max_steps,
            render_mode="rgb_array"
        )
        
        # Create system
        self.system = DMARLSystem(self.env, config)
        
        # Load model if path provided
        if model_path is not None:
            self.system.load(model_path)
    
    def _extract_local_obs(self, info: Dict, obs: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Extract local observations from environment info or observation
        
        The actor network needs the full neighborhood observation (ddl * (2*κ_o+1)^2),
        not just the agent's own state. The global observation 'obs' contains the
        full observations, while info['local_obs'] may only contain agent's own state.
        
        Args:
            info: Info dictionary from environment
            obs: Global observation array (preferred source for full observations)
            
        Returns:
            local_obs_list: List of local observations (full neighborhood for each agent)
        """
        expected_obs_dim = self.config.local_obs_dim
        
        # Prioritize using global observation 'obs' as it contains full neighborhood info
        if obs is not None:
            local_obs_list = self._extract_local_obs_from_global(obs)
            # Verify dimensions match
            if len(local_obs_list) > 0 and len(local_obs_list[0]) == expected_obs_dim:
                return local_obs_list
        
        # Fallback to info['local_obs'] if obs is not available
        if 'local_obs' in info:
            local_obs_list = [
                np.array(info['local_obs'][i])
                for i in range(self.config.n_agents)
            ]
            # Check if dimensions match expected (full neighborhood)
            if len(local_obs_list) > 0 and len(local_obs_list[0]) == expected_obs_dim:
                return local_obs_list
            # If info['local_obs'] has wrong dimension, it's just agent's own state
        
        # Last resort: create zero observations with correct dimension
        return [
            np.zeros(expected_obs_dim, dtype=np.float32)
            for _ in range(self.config.n_agents)
        ]
    
    def _extract_local_obs_from_global(self, global_obs: np.ndarray) -> List[np.ndarray]:
        """
        Extract local observations from global observation array
        
        Args:
            global_obs: Global observation from environment
            
        Returns:
            List of local observations for each agent (full neighborhood observations)
        """
        obs_dim = self.config.local_obs_dim
        
        if isinstance(global_obs, dict):
            return [np.array(global_obs.get(i, np.zeros(obs_dim, dtype=np.float32))) 
                    for i in range(self.config.n_agents)]
        elif isinstance(global_obs, np.ndarray):
            if global_obs.ndim == 1:
                # Assume concatenated observations
                if len(global_obs) == self.config.n_agents * obs_dim:
                    return [global_obs[i*obs_dim:(i+1)*obs_dim] 
                            for i in range(self.config.n_agents)]
                else:
                    # Try to split evenly
                    obs_dim_per_agent = len(global_obs) // self.config.n_agents
                    return [global_obs[i*obs_dim_per_agent:(i+1)*obs_dim_per_agent] 
                            for i in range(self.config.n_agents)]
            else:
                # 2D array: [n_agents, obs_dim] or similar
                return [global_obs[i] for i in range(self.config.n_agents)]
        else:
            # Fallback: create zero observations
            return [np.zeros(obs_dim, dtype=np.float32) 
                    for _ in range(self.config.n_agents)]
    
    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False,
        save_gif: bool = False,
        gif_path: str = "evaluation.gif"
    ) -> Dict:
        """
        Evaluate trained model
        
        Args:
            n_episodes: Number of evaluation episodes
            render: Whether to render during evaluation
            save_gif: Whether to save evaluation as GIF
            gif_path: Path to save GIF
            
        Returns:
            results: Dictionary of evaluation metrics
        """
        # Disable exploration
        self.system.epsilon = 0.0
        
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        
        if save_gif:
            self.env.start_frame_collection()
        
        # Create progress bar for evaluation
        pbar = tqdm(range(n_episodes), desc="Evaluating", unit="episode",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for episode in pbar:
            obs, info = self.env.reset()
            local_obs_list = self._extract_local_obs(info, obs)
            
            episode_reward = 0
            step = 0
            done = False
            
            while not done and step < self.config.max_steps:
                # Select greedy action
                actions = self.system.select_actions(local_obs_list, deterministic=True)
                
                # Execute action
                next_obs, reward, terminated, truncated, next_info = self.env.step(actions)
                done = terminated or truncated
                
                if render or save_gif:
                    self.env.render()
                
                # Update state
                local_obs_list = self._extract_local_obs(next_info, next_obs)
                episode_reward += reward
                step += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            episode_successes.append(episode_reward > 0)
            
            # Update progress bar
            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards)
                pbar.set_postfix({
                    'Avg Reward': f'{avg_reward:.2f}',
                    'Current': f'{episode_reward:.2f}'
                })
        
        pbar.close()
        
        if save_gif:
            self.env.stop_frame_collection()
            self.env.save_gif(gif_path, fps=3)
            print(f"Evaluation GIF saved to {gif_path}")
        
        # Compute statistics
        results = {
            'n_episodes': n_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': np.mean(episode_successes),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        # Print results
        print("\n" + "=" * 50)
        print(f"Evaluation Results ({n_episodes} episodes)")
        print("=" * 50)
        print(f"Mean Reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Min/Max:      {results['min_reward']:.2f} / {results['max_reward']:.2f}")
        print(f"Mean Length:  {results['mean_length']:.1f}")
        print(f"Success Rate: {results['success_rate']*100:.1f}%")
        print("=" * 50)
        
        return results
    
    def close(self) -> None:
        """Clean up resources"""
        self.env.close()


def plot_training_curves(
    episode_rewards: List[float],
    critic_losses: List[float] = None,
    actor_losses: List[float] = None,
    window: int = 100,
    save_path: str = "training_curves.png"
) -> None:
    """
    Plot training curves
    
    Args:
        episode_rewards: List of episode rewards
        critic_losses: List of critic losses (optional)
        actor_losses: List of actor losses (optional)
        window: Moving average window size
        save_path: Path to save the figure
    """
    n_plots = 1 + (critic_losses is not None) + (actor_losses is not None)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot rewards
    ax = axes[0]
    ax.plot(episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
    
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(
            episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )
        ax.plot(
            range(window - 1, len(episode_rewards)),
            moving_avg,
            label=f'{window}-Episode Moving Average',
            color='red',
            linewidth=2
        )
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot critic loss
    plot_idx = 1
    if critic_losses is not None:
        ax = axes[plot_idx]
        ax.plot(critic_losses, alpha=0.5, color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Critic Loss')
        ax.set_title('Critic Loss')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot actor loss
    if actor_losses is not None:
        ax = axes[plot_idx]
        ax.plot(actor_losses, alpha=0.5, color='orange')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Actor Loss')
        ax.set_title('Actor Loss')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Changed from plt.show() to plt.close() for headless execution
    print(f"Training curves saved to {save_path}")


def compare_models(
    model_paths: List[str],
    model_names: List[str],
    config: Config,
    n_episodes: int = 20
) -> Dict:
    """
    Compare multiple trained models
    
    Args:
        model_paths: List of paths to saved models
        model_names: Names for each model
        config: Configuration object
        n_episodes: Number of evaluation episodes
        
    Returns:
        comparison: Dictionary of comparison results
    """
    results = {}
    
    for path, name in zip(model_paths, model_names):
        print(f"\nEvaluating {name}...")
        evaluator = Evaluator(config, model_path=path)
        results[name] = evaluator.evaluate(n_episodes=n_episodes)
        evaluator.close()
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(results.keys())
    means = [results[name]['mean_reward'] for name in names]
    stds = [results[name]['std_reward'] for name in names]
    
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    
    ax.set_ylabel('Average Reward')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results