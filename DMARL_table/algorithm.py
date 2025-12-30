"""
Algorithm Implementation for NMARL.

This module implements Algorithm 1: Distributed and Scalable Algorithm
from the paper "Distributed and Scalable Algorithm for Networked
Multi-agent Reinforcement Learning under Asymmetric Information".
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time
from tqdm import tqdm

from config import Config, AlgorithmConfig
from network_utils import CommunicationNetwork, StateAggregator
from agent import Agent, MultiAgentSystem, SoftmaxPolicy
from q_function import TruncatedQFunction, QFunctionManager, PolicyGradientEstimator


class NMARLAlgorithm:
    """
    Implementation of Algorithm 1: Distributed and Scalable Algorithm.
    
    This algorithm operates without requiring symmetric information structures
    or Q-values exchange among agents.
    """
    
    def __init__(self, config: Config, env):
        """
        Initialize the NMARL algorithm.
        
        Args:
            config: Configuration object
            env: Gymnasium environment (WirelessCommEnv)
        """
        self.config = config
        self.env = env
        self.algo_config = config.algo
        
        # Environment properties
        self.n_agents = config.env.n_agents
        self.n_actions = 5  # As per wireless comm env documentation
        
        # Set up communication network
        self.network = CommunicationNetwork(
            grid_x=config.env.grid_x,
            grid_y=config.env.grid_y,
            topology="grid"
        )
        
        # Set up state aggregator
        self.state_aggregator = StateAggregator(
            network=self.network,
            kappa_o=self.algo_config.kappa_o,
            kappa_r=self.algo_config.kappa_r
        )
        
        # Initialize agents
        self._init_agents()
        
        # Initialize Q-function manager
        self.q_manager = QFunctionManager(
            network=self.network,
            state_aggregator=self.state_aggregator,
            kappa_o=self.algo_config.kappa_o,
            kappa_r=self.algo_config.kappa_r,
            gamma=self.algo_config.gamma
        )
        
        # Policy gradient estimator
        self.pg_estimator = PolicyGradientEstimator(gamma=self.algo_config.gamma)
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'q_values': [],
            'gradient_norms': [],
            'policy_entropy': [],
            'local_rewards': defaultdict(list)
        }
    
    def _init_agents(self):
        """Initialize all agents with their policies."""
        # Compute state dimension based on environment
        # Each agent observes ddl * ((2*n_obs_neighbors+1)^2) values
        obs_grid_size = (2 * self.config.env.n_obs_neighbors + 1) ** 2
        state_dim = self.config.env.ddl * obs_grid_size
        
        self.agents: List[Agent] = []
        for i in range(self.n_agents):
            agent = Agent(
                agent_id=i,
                state_dim=state_dim,
                n_actions=self.n_actions,
                kappa_o=self.algo_config.kappa_o[i],
                kappa_r=self.algo_config.kappa_r[i],
                temperature=self.algo_config.temperature,
                device=self.config.env.device
            )
            self.agents.append(agent)
    
    def _extract_local_states(self, global_obs: np.ndarray) -> List[np.ndarray]:
        """
        Extract local state observations for each agent.
        
        Args:
            global_obs: Global observation from environment
            
        Returns:
            List of local observations for each agent
        """
        # The environment returns observations for each agent
        # We need to handle the observation format
        if isinstance(global_obs, dict):
            return [global_obs.get(i, np.zeros(self.config.env.ddl)) 
                    for i in range(self.n_agents)]
        elif isinstance(global_obs, np.ndarray):
            if global_obs.ndim == 1:
                # Assume concatenated observations
                obs_dim = len(global_obs) // self.n_agents
                return [global_obs[i*obs_dim:(i+1)*obs_dim] 
                        for i in range(self.n_agents)]
            else:
                return [global_obs[i] for i in range(self.n_agents)]
        else:
            # Try to use info['local_obs'] if available
            return [np.zeros(self.config.env.ddl) for _ in range(self.n_agents)]
    
    def _select_joint_action(self, local_states: List[np.ndarray]) -> np.ndarray:
        """
        Select joint action using current policies.
        
        Args:
            local_states: Local observations for each agent
            
        Returns:
            Joint action array
        """
        actions = []
        for i, (agent, state) in enumerate(zip(self.agents, local_states)):
            action = agent.select_action(state)
            actions.append(action)
        return np.array(actions)
    
    def _compute_actor_learning_rate(self, m: int) -> float:
        """
        Compute actor step learning rate η_m.
        
        η_m = η / sqrt(m+1)
        But use a minimum learning rate to prevent it from becoming too small
        
        Args:
            m: Current outer iteration
            
        Returns:
            Learning rate η_m
        """
        lr = self.algo_config.eta / np.sqrt(m + 1)
        # Use minimum learning rate to ensure continued learning
        min_lr = self.algo_config.eta * 0.01
        return max(lr, min_lr)
    
    def _compute_critic_learning_rate(self, t: int) -> float:
        """
        Compute critic step learning rate α_{t-1}.
        
        α_{t-1} = h / (t-1 + t0)
        
        Args:
            t: Current inner iteration (1-indexed)
            
        Returns:
            Learning rate α_{t-1}
        """
        return self.algo_config.h / (t - 1 + self.algo_config.t0)
    
    def run_episode(self, m: int, collect_stats: bool = True) -> Dict:
        """
        Run one episode of the algorithm (one outer iteration).
        
        Implements lines 2-12 of Algorithm 1.
        
        Args:
            m: Episode number
            collect_stats: Whether to collect training statistics
            
        Returns:
            Episode statistics
        """
        T = self.algo_config.T
        gamma = self.algo_config.gamma
        
        # NOTE: Q-functions should NOT be reset every episode for convergence
        # Only reset on first episode
        if m == 0:
            self.q_manager.reset_all()
            for agent in self.agents:
                agent.reset_q_estimates()
        
        # Sample initial state (Line 3)
        obs, info = self.env.reset()
        
        # Extract local states
        if 'local_obs' in info:
            local_states = [info['local_obs'][i] for i in range(self.n_agents)]
        else:
            local_states = self._extract_local_states(obs)
        
        # Initial action sampling (Line 3)
        actions = self._select_joint_action(local_states)
        
        # Store trajectories for gradient computation
        trajectories = [[] for _ in range(self.n_agents)]
        
        # Episode statistics
        episode_reward = 0.0
        agent_rewards = [0.0] * self.n_agents
        
        # Create initial state array for Q-function
        # Use a more robust state representation
        global_state = np.array([hash(tuple(s.flatten().astype(int))) % 10000 
                                 if len(s) > 0 else 0 
                                 for s in local_states])
        
        # ============================================
        # Distributed Critic Step (Lines 5-9)
        # ============================================
        for t in range(1, T + 1):
            # Store current info for trajectory
            for i in range(self.n_agents):
                state_action_key = self.q_manager.create_state_action_key(
                    global_state, actions, i
                )
                trajectories[i].append({
                    'local_state': local_states[i].copy(),
                    'action': actions[i],
                    'state_action_key': state_action_key
                })
            
            # Take action in environment (Line 6)
            next_obs, reward, terminated, truncated, info = self.env.step(actions)
            
            # Extract rewards (Line 6-7)
            if 'local_rewards' in info:
                global_rewards = np.array(info['local_rewards'])
            else:
                # Distribute total reward equally
                global_rewards = np.full(self.n_agents, reward / self.n_agents)
            
            # Extract next local states
            if 'local_obs' in info:
                next_local_states = [info['local_obs'][i] for i in range(self.n_agents)]
            else:
                next_local_states = self._extract_local_states(next_obs)
            
            # Select next actions
            next_actions = self._select_joint_action(next_local_states)
            
            # Create next state array
            next_global_state = np.array([hash(tuple(s.flatten().astype(int))) % 10000 
                                          if len(s) > 0 else 0 
                                          for s in next_local_states])
            
            # Compute learning rate (Line 7)
            alpha = self._compute_critic_learning_rate(t)
            
            # Update Q-function estimates for all agents (Line 8)
            self.q_manager.update_all_q_functions(
                global_state=global_state,
                global_action=actions,
                global_rewards=global_rewards,
                next_global_state=next_global_state,
                next_global_action=next_actions,
                t=t,
                h=self.algo_config.h,
                t0=self.algo_config.t0
            )
            
            # Also update agent's internal Q-estimates
            for i in range(self.n_agents):
                current_key = self.q_manager.create_state_action_key(
                    global_state, actions, i
                )
                next_key = self.q_manager.create_state_action_key(
                    next_global_state, next_actions, i
                )
                truncated_reward = self.q_manager.compute_truncated_reward(
                    global_rewards, i
                )
                self.agents[i].update_q_estimate(
                    current_key, truncated_reward, next_key, alpha, gamma
                )
            
            # Accumulate rewards
            episode_reward += reward
            for i in range(self.n_agents):
                agent_rewards[i] += global_rewards[i]
            
            # Update state for next iteration
            local_states = next_local_states
            global_state = next_global_state
            actions = next_actions
            
            if terminated or truncated:
                break
        
        # ============================================
        # Decentralized Actor Step (Lines 10-11)
        # ============================================
        eta_m = self._compute_actor_learning_rate(m)
        
        gradient_norms = []
        for i, agent in enumerate(self.agents):
            # Calculate gradient estimate (Line 10, equation 12)
            gradient_updates = agent.compute_policy_gradient_estimate(
                trajectories[i], gamma
            )
            
            # Compute gradient norm for logging
            total_grad_norm = sum(np.linalg.norm(g) for g in gradient_updates.values())
            gradient_norms.append(total_grad_norm)
            
            # Update policy parameters (Line 11, equation 13)
            agent.update_policy(
                gradient_updates, 
                eta_m, 
                self.algo_config.max_grad_norm
            )
        
        # Collect statistics
        episode_stats = {
            'episode': m,
            'total_reward': episode_reward,
            'episode_length': t,
            'agent_rewards': agent_rewards,
            'mean_gradient_norm': np.mean(gradient_norms),
            'learning_rate_actor': eta_m
        }
        
        if collect_stats:
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(t)
            self.training_stats['gradient_norms'].append(np.mean(gradient_norms))
            
            for i, r in enumerate(agent_rewards):
                self.training_stats['local_rewards'][i].append(r)
            
            # Q-value statistics
            q_stats = self.q_manager.get_statistics()
            self.training_stats['q_values'].append({
                'mean': np.mean(q_stats['mean_q_values']),
                'max': np.max(q_stats['max_q_values']),
                'min': np.min(q_stats['min_q_values'])
            })
        
        return episode_stats
    
    def train(self, callback=None) -> Dict:
        """
        Train the agents using Algorithm 1.
        
        Args:
            callback: Optional callback function called after each episode
                      with signature callback(episode, stats)
                      
        Returns:
            Training statistics
        """
        M = self.algo_config.M
        
        print(f"Starting training for {M} episodes...")
        print(f"Config: {self.n_agents} agents, kappa_o={self.algo_config.kappa_o}, "
              f"kappa_r={self.algo_config.kappa_r}")
        
        start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(range(M), desc="Training", unit="episode", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for m in pbar:
            # Run one episode
            episode_stats = self.run_episode(m)
            
            # Update progress bar with current statistics
            if len(self.training_stats['episode_rewards']) > 0:
                recent_rewards = self.training_stats['episode_rewards'][-min(self.config.logging.log_interval, len(self.training_stats['episode_rewards'])):]
                mean_reward = np.mean(recent_rewards)
                pbar.set_postfix({
                    'Reward': f'{mean_reward:.4f}',
                    'Grad': f'{episode_stats["mean_gradient_norm"]:.4f}',
                    'LR': f'{episode_stats["learning_rate_actor"]:.6f}'
                })
            
            # Logging (less frequent now since we have progress bar)
            if (m + 1) % self.config.logging.log_interval == 0:
                recent_rewards = self.training_stats['episode_rewards'][-self.config.logging.log_interval:]
                mean_reward = np.mean(recent_rewards)
                tqdm.write(f"Episode {m+1}/{M} | Mean Reward: {mean_reward:.4f} | "
                      f"Grad Norm: {episode_stats['mean_gradient_norm']:.4f} | "
                      f"LR: {episode_stats['learning_rate_actor']:.6f}")
            
            # Callback
            if callback is not None:
                callback(m, episode_stats)
        
        pbar.close()
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        
        return self.training_stats
    
    def evaluate(self, n_episodes: int = 10) -> Dict:
        """
        Evaluate the trained agents.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation statistics
        """
        eval_rewards = []
        
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            
            if 'local_obs' in info:
                local_states = [info['local_obs'][i] for i in range(self.n_agents)]
            else:
                local_states = self._extract_local_states(obs)
            
            episode_reward = 0.0
            
            for _ in range(self.algo_config.T):
                # Greedy action selection (no exploration)
                actions = []
                for agent, state in zip(self.agents, local_states):
                    probs = agent.policy.get_action_probs(state)
                    actions.append(np.argmax(probs))
                actions = np.array(actions)
                
                obs, reward, terminated, truncated, info = self.env.step(actions)
                episode_reward += reward
                
                if 'local_obs' in info:
                    local_states = [info['local_obs'][i] for i in range(self.n_agents)]
                else:
                    local_states = self._extract_local_states(obs)
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'all_rewards': eval_rewards
        }
    
    def get_policy_parameters(self) -> List[Dict]:
        """Get policy parameters for all agents."""
        return [agent.policy.get_all_parameters() for agent in self.agents]
    
    def save_model(self, filepath: str):
        """Save model parameters to file."""
        import pickle
        
        model_data = {
            'policy_params': self.get_policy_parameters(),
            'config': self.config.to_dict(),
            'training_stats': self.training_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model parameters from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore policy parameters
        for agent, params in zip(self.agents, model_data['policy_params']):
            agent.policy.set_parameters(params)
        
        # Restore training stats
        if 'training_stats' in model_data:
            self.training_stats = model_data['training_stats']
        
        print(f"Model loaded from {filepath}")


class IQLBaseline:
    """
    Independent Q-Learning baseline for comparison.
    
    Each agent uses only its local state-action and reward information.
    """
    
    def __init__(self, config: Config, env):
        """Initialize IQL baseline."""
        self.config = config
        self.env = env
        self.n_agents = config.env.n_agents
        self.n_actions = 5
        
        # Each agent has its own Q-table and policy
        self.agents: List[Agent] = []
        
        obs_grid_size = (2 * config.env.n_obs_neighbors + 1) ** 2
        state_dim = config.env.ddl * obs_grid_size
        
        for i in range(self.n_agents):
            agent = Agent(
                agent_id=i,
                state_dim=state_dim,
                n_actions=self.n_actions,
                kappa_o=0,  # Only local observation
                kappa_r=0,  # Only local reward
                temperature=config.algo.temperature,
                device=config.env.device
            )
            self.agents.append(agent)
        
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'local_rewards': defaultdict(list)
        }
    
    def train(self, M: int = 1000, T: int = 50) -> Dict:
        """Train using IQL."""
        gamma = self.config.algo.gamma
        
        # Create progress bar
        pbar = tqdm(range(M), desc="IQL Training", unit="episode",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for m in pbar:
            # Reset Q-estimates
            for agent in self.agents:
                agent.reset_q_estimates()
            
            obs, info = self.env.reset()
            
            if 'local_obs' in info:
                local_states = [info['local_obs'][i] for i in range(self.n_agents)]
            else:
                local_states = [np.zeros(self.config.env.ddl) 
                               for _ in range(self.n_agents)]
            
            episode_reward = 0.0
            trajectories = [[] for _ in range(self.n_agents)]
            
            for t in range(1, T + 1):
                # Select actions
                actions = np.array([agent.select_action(state) 
                                   for agent, state in zip(self.agents, local_states)])
                
                # Store trajectories
                for i in range(self.n_agents):
                    key = (tuple(local_states[i].flatten().astype(int)), 
                           (actions[i],))
                    trajectories[i].append({
                        'local_state': local_states[i].copy(),
                        'action': actions[i],
                        'state_action_key': key
                    })
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(actions)
                
                if 'local_rewards' in info:
                    local_rewards = info['local_rewards']
                else:
                    local_rewards = [reward / self.n_agents] * self.n_agents
                
                if 'local_obs' in info:
                    next_local_states = [info['local_obs'][i] 
                                        for i in range(self.n_agents)]
                else:
                    next_local_states = [np.zeros(self.config.env.ddl) 
                                        for _ in range(self.n_agents)]
                
                next_actions = np.array([agent.select_action(state) 
                                        for agent, state in zip(self.agents, next_local_states)])
                
                # Update Q-estimates (local only)
                alpha = self.config.algo.h / (t + self.config.algo.t0)
                for i in range(self.n_agents):
                    current_key = (tuple(local_states[i].flatten().astype(int)), 
                                  (actions[i],))
                    next_key = (tuple(next_local_states[i].flatten().astype(int)), 
                               (next_actions[i],))
                    
                    self.agents[i].update_q_estimate(
                        current_key, 
                        local_rewards[i] / self.n_agents,  # Local reward only
                        next_key, 
                        alpha, 
                        gamma
                    )
                
                episode_reward += reward
                local_states = next_local_states
                
                if terminated or truncated:
                    break
            
            # Actor update
            eta_m = self.config.algo.eta / np.sqrt(m + 1)
            for i, agent in enumerate(self.agents):
                gradient_updates = agent.compute_policy_gradient_estimate(
                    trajectories[i], gamma
                )
                agent.update_policy(gradient_updates, eta_m)
            
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(t)
            
            # Update progress bar
            if len(self.training_stats['episode_rewards']) > 0:
                recent = self.training_stats['episode_rewards'][-min(self.config.logging.log_interval, len(self.training_stats['episode_rewards'])):]
                pbar.set_postfix({'Reward': f'{np.mean(recent):.4f}'})
            
            if (m + 1) % self.config.logging.log_interval == 0:
                recent = self.training_stats['episode_rewards'][-self.config.logging.log_interval:]
                tqdm.write(f"IQL Episode {m+1}/{M} | Mean Reward: {np.mean(recent):.4f}")
        
        pbar.close()
        return self.training_stats