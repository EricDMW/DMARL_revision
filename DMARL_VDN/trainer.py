# ============================================================================
# File: trainer.py
# Description: Training loop and utilities for VDN
# ============================================================================
"""
Training Module for VDN

This module provides:
- Main training loop
- Episode execution
- Logging and checkpointing
- Training statistics tracking
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import os
import time
import json
from datetime import datetime
from tqdm import tqdm

from config import Config
from vdn import VDNSystem
import env_lib


class Trainer:
    """
    Trainer class for VDN on Wireless Communication Environment
    
    Handles:
    - Environment interaction
    - Training loop execution
    - Logging and checkpointing
    - Statistics tracking
    
    Attributes:
        config: Configuration object
        env: Environment instance
        system: VDNSystem instance
        episode_rewards: List of episode rewards
        best_reward: Best average reward achieved
    """
    
    def __init__(self, config: Config):
        """
        Initialize trainer
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Create environment
        self.env = env_lib.WirelessCommEnv(
            grid_x=config.grid_x,
            grid_y=config.grid_y,
            ddl=config.ddl,
            n_obs_neighbors=config.n_obs_neighbors,
            max_iter=config.max_steps,
            render_mode=None
        )
        
        # Set environment seed if available
        if hasattr(config, 'seed') and config.seed is not None:
            if hasattr(self.env, 'seed'):
                self.env.seed(config.seed)
        
        # Create multi-agent system
        self.system = VDNSystem(self.env, config)
        
        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.critic_losses: List[float] = []
        self.actor_losses: List[float] = []
        self.best_reward = float('-inf')
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
    
    def _extract_local_obs(self, info: Dict, obs: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], np.ndarray]:
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
            local_obs_array: Stacked array [n_agents, local_obs_dim]
        """
        expected_obs_dim = self.config.local_obs_dim
        
        # Prioritize using global observation 'obs' as it contains full neighborhood info
        if obs is not None:
            local_obs_list = self._extract_local_obs_from_global(obs)
            # Verify dimensions match
            if len(local_obs_list) > 0 and len(local_obs_list[0]) == expected_obs_dim:
                local_obs_array = np.stack(local_obs_list)
                return local_obs_list, local_obs_array
        
        # Fallback to info['local_obs'] if obs is not available
        if 'local_obs' in info:
            local_obs_list = [
                np.array(info['local_obs'][i])
                for i in range(self.config.n_agents)
            ]
            # Check if dimensions match expected (full neighborhood)
            if len(local_obs_list) > 0 and len(local_obs_list[0]) == expected_obs_dim:
                local_obs_array = np.stack(local_obs_list)
                return local_obs_list, local_obs_array
        
        # Last resort: create zero observations with correct dimension
        local_obs_list = [
            np.zeros(expected_obs_dim, dtype=np.float32)
            for _ in range(self.config.n_agents)
        ]
        local_obs_array = np.stack(local_obs_list)
        return local_obs_list, local_obs_array
    
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
    
    def run_episode(self, training: bool = True) -> Tuple[float, int, Dict]:
        """
        Run a single episode
        
        Args:
            training: If True, store experiences and update networks
            
        Returns:
            episode_reward: Total reward for the episode
            episode_length: Number of steps taken
            episode_stats: Dictionary of episode statistics
        """
        obs, info = self.env.reset()
        local_obs_list, local_obs_array = self._extract_local_obs(info, obs)
        
        episode_reward = 0.0
        episode_critic_losses = []
        episode_actor_losses = []
        step = 0
        
        done = False
        while not done and step < self.config.max_steps:
            # Select actions
            actions = self.system.select_actions(
                local_obs_list,
                deterministic=not training
            )
            
            # Execute actions
            next_obs, reward, terminated, truncated, next_info = self.env.step(actions)
            done = terminated or truncated
            
            # Extract next local observations
            next_local_obs_list, next_local_obs_array = self._extract_local_obs(next_info, next_obs)
            
            # Get local rewards
            if 'local_rewards' in next_info:
                local_rewards = np.array(next_info['local_rewards'])
            else:
                # Fallback: distribute total reward equally among agents
                local_rewards = np.full(self.config.n_agents, reward / self.config.n_agents)
            
            if training:
                # Store transition
                dones = np.array([float(done)] * self.config.n_agents)
                self.system.store_transition(
                    obs, actions, local_rewards, next_obs, dones,
                    local_obs_array, next_local_obs_array
                )
                
                # Update networks
                self.system.total_steps += 1
                if (self.system.total_steps > self.config.warmup_steps and
                    self.system.total_steps % self.config.update_freq == 0):
                    result = self.system.update()
                    if result is not None:
                        q_losses, actor_losses = result
                        episode_critic_losses.extend(q_losses)
                        episode_actor_losses.extend(actor_losses)
            
            # Update state
            obs = next_obs
            local_obs_list = next_local_obs_list
            local_obs_array = next_local_obs_array
            episode_reward += reward
            step += 1
        
        # Compute episode statistics
        episode_stats = {
            'reward': episode_reward,
            'length': step,
            'critic_loss': np.mean(episode_critic_losses) if episode_critic_losses else 0.0,
            'actor_loss': np.mean(episode_actor_losses) if episode_actor_losses else 0.0
        }
        
        return episode_reward, step, episode_stats
    
    def train(self) -> Dict:
        """
        Main training loop
        
        Returns:
            training_stats: Dictionary of training statistics
        """
        print("=" * 60)
        print("Starting VDN Training for Wireless Communication")
        print("=" * 60)
        print(self.config)
        print("=" * 60)
        
        start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(range(self.config.max_episodes), desc="Training", unit="episode",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for episode in pbar:
            # Run episode
            episode_reward, episode_length, episode_stats = self.run_episode(training=True)
            
            # Decay exploration
            self.system.decay_epsilon()
            
            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.critic_losses.append(episode_stats['critic_loss'])
            self.actor_losses.append(episode_stats['actor_loss'])
            
            # Update progress bar with current statistics
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-min(self.config.log_freq, len(self.episode_rewards)):]
                avg_reward = np.mean(recent_rewards)
                pbar.set_postfix({
                    'Reward': f'{avg_reward:.2f}',
                    'Epsilon': f'{self.system.epsilon:.3f}',
                    'Buffer': f'{len(self.system.replay_buffer)}'
                })
            
            # Logging (less frequent now since we have progress bar)
            if (episode + 1) % self.config.log_freq == 0:
                self._log_progress(episode + 1, start_time)
            
            # Save checkpoint
            if (episode + 1) % self.config.save_freq == 0:
                self._save_checkpoint(episode + 1)
        
        pbar.close()
        
        # Final save
        self._save_checkpoint(self.config.max_episodes, final=True)
        
        # Training summary
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Total time: {total_time / 3600:.2f} hours")
        print(f"Best average reward: {self.best_reward:.2f}")
        print("=" * 60)
        
        results = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'critic_losses': self.critic_losses,
            'actor_losses': self.actor_losses,
            'best_reward': self.best_reward,
            'total_time': total_time,
            'config': {
                'n_obs_neighbors': self.config.n_obs_neighbors,
                'n_reward_neighbors': self.config.n_reward_neighbors,
                'grid_x': self.config.grid_x,
                'grid_y': self.config.grid_y,
                'ddl': self.config.ddl,
                'max_episodes': self.config.max_episodes,
                'seed': getattr(self.config, 'seed', None)
            }
        }
        
        # Save results to JSON file for easy loading
        results_file = os.path.join(self.config.save_dir, 'training_results.json')
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'episode_rewards': [float(r) for r in self.episode_rewards],
            'episode_lengths': [int(l) for l in self.episode_lengths],
            'critic_losses': [float(l) for l in self.critic_losses],
            'actor_losses': [float(l) for l in self.actor_losses],
            'best_reward': float(self.best_reward),
            'total_time': float(total_time),
            'config': results['config']
        }
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        return results
    
    def _log_progress(self, episode: int, start_time: float) -> None:
        """Log training progress"""
        avg_reward = np.mean(self.episode_rewards[-self.config.log_freq:])
        avg_length = np.mean(self.episode_lengths[-self.config.log_freq:])
        avg_critic_loss = np.mean(self.critic_losses[-self.config.log_freq:])
        avg_actor_loss = np.mean(self.actor_losses[-self.config.log_freq:])
        
        elapsed_time = time.time() - start_time
        eps_per_sec = episode / elapsed_time
        
        # Use tqdm.write to avoid interfering with progress bar
        tqdm.write(f"\nEpisode {episode}/{self.config.max_episodes}")
        tqdm.write(f"  Average Reward:     {avg_reward:.2f}")
        tqdm.write(f"  Average Length:     {avg_length:.1f}")
        tqdm.write(f"  Q-Network Loss:     {avg_critic_loss:.4f}")
        tqdm.write(f"  Actor Loss:         {avg_actor_loss:.4f}")
        tqdm.write(f"  Epsilon:            {self.system.epsilon:.4f}")
        tqdm.write(f"  Buffer Size:        {len(self.system.replay_buffer)}")
        tqdm.write(f"  Episodes/sec:       {eps_per_sec:.2f}")
        tqdm.write("-" * 40)
        
        # Update best reward and save best model
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            best_path = os.path.join(self.config.save_dir, "best_model")
            self.system.save(best_path)
            tqdm.write(f"  New best model saved! (reward: {avg_reward:.2f})")
    
    def _save_checkpoint(self, episode: int, final: bool = False) -> None:
        """Save training checkpoint"""
        if final:
            checkpoint_path = os.path.join(self.config.save_dir, "final_model")
        else:
            checkpoint_path = os.path.join(self.config.save_dir, f"checkpoint_{episode}")
        
        self.system.save(checkpoint_path)
    
    def close(self) -> None:
        """Clean up resources"""
        self.env.close()
