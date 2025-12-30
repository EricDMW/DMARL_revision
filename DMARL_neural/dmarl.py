# ============================================================================
# File: DMARL.py
# Description: Multi-Agent System Manager
# ============================================================================
"""
DMARL System Module

This module implements the multi-agent system manager that:
- Initializes and manages all agents
- Coordinates action selection
- Handles experience storage and sampling
- Orchestrates training updates

Key responsibilities:
- Create agents with appropriate network sizes based on topology
- Extract neighbor information for centralized critic training
- Compute truncated rewards for each agent
- Update all agents in a coordinated manner
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import os

from config import Config
from buffer import ReplayBuffer
from agent import DMARLAgent
from utils import NeighborExtractor


class DMARLSystem:
    """
    Multi-Agent DDPG System Manager
    
    Coordinates training of multiple DMARL agents on the
    wireless communication environment.
    
    Attributes:
        env: Environment instance
        config: Configuration
        n_agents: Number of agents
        agents: List of DMARLAgent instances
        replay_buffer: Shared experience buffer
        neighbor_extractor: Topology-aware neighbor extraction
        epsilon: Current exploration rate
        total_steps: Total training steps
    """
    
    def __init__(self, env, config: Config):
        """
        Initialize Multi-Agent System
        
        Args:
            env: Wireless communication environment
            config: Configuration object
        """
        self.env = env
        self.config = config
        self.device = config.device
        self.n_agents = config.n_agents
        self.n_actions = config.n_actions
        self.local_obs_dim = config.local_obs_dim
        
        # Initialize neighbor extractor for topology-aware operations
        self.neighbor_extractor = NeighborExtractor(
            grid_x=config.grid_x,
            grid_y=config.grid_y,
            kappa_o=config.n_obs_neighbors,
            kappa_r=config.n_reward_neighbors,
            device=self.device
        )
        
        # Initialize agents
        self.agents = self._create_agents()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_size,
            n_agents=self.n_agents,
            device=self.device
        )
        
        # Training state
        self.epsilon = config.epsilon_start
        self.total_steps = 0
        
        # Statistics
        self.critic_losses = []
        self.actor_losses = []
    
    def _create_agents(self) -> List[DMARLAgent]:
        """
        Create all agents with appropriate network sizes
        
        Returns:
            List of initialized DMARLAgent instances
        """
        agents = []
        
        for agent_id in range(self.n_agents):
            # Get neighbor information for this agent
            neighbor_info = self.neighbor_extractor.get_neighbor_info(agent_id)
            n_neighbors = neighbor_info['n_obs_neighbors']
            neighbor_obs_dim = n_neighbors * self.local_obs_dim
            
            # Create agent
            agent = DMARLAgent(
                agent_id=agent_id,
                local_obs_dim=self.local_obs_dim,
                neighbor_obs_dim=neighbor_obs_dim,
                n_neighbors=n_neighbors,
                n_actions=self.n_actions,
                config=self.config,
                neighbor_extractor=self.neighbor_extractor
            )
            agents.append(agent)
        
        return agents
    
    def select_actions(
        self,
        local_obs_list: List[np.ndarray],
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select actions for all agents
        
        Args:
            local_obs_list: List of local observations for each agent
            deterministic: If True, use greedy actions (no exploration)
            
        Returns:
            actions: Array of actions [n_agents]
        """
        actions = []
        epsilon = 0.0 if deterministic else self.epsilon
        
        for agent_id, agent in enumerate(self.agents):
            action = agent.select_action(
                local_obs_list[agent_id],
                epsilon=epsilon,
                deterministic=deterministic
            )
            actions.append(action)
        
        return np.array(actions)
    
    def store_transition(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        local_obs: np.ndarray,
        next_local_obs: np.ndarray
    ) -> None:
        """
        Store transition in replay buffer
        
        Args:
            obs: Global observation
            actions: Actions for all agents
            rewards: Local rewards for all agents
            next_obs: Next global observation
            dones: Done flags
            local_obs: Local observations [n_agents, local_obs_dim]
            next_local_obs: Next local observations
        """
        self.replay_buffer.push(
            obs, actions, rewards, next_obs, dones, local_obs, next_local_obs
        )
    
    def update(self) -> Optional[Tuple[List[float], List[float]]]:
        """
        Update all agents using sampled batch
        
        Returns:
            (critic_losses, actor_losses): Lists of losses for each agent,
            or None if buffer not ready
        """
        if not self.replay_buffer.is_ready(self.config.batch_size):
            return None
        
        # Sample batch
        (obs, actions, rewards, next_obs, dones,
         local_obs, next_local_obs) = self.replay_buffer.sample(self.config.batch_size)
        
        critic_losses = []
        actor_losses = []
        
        # Update each agent
        for agent_id, agent in enumerate(self.agents):
            # Get neighbor indices
            neighbor_ids = self.neighbor_extractor.get_obs_neighbor_indices(agent_id)
            
            # Extract neighbor observations
            neighbor_obs = local_obs[:, neighbor_ids, :].reshape(
                self.config.batch_size, -1
            )
            next_neighbor_obs = next_local_obs[:, neighbor_ids, :].reshape(
                self.config.batch_size, -1
            )
            
            # Extract neighbor actions (one-hot encoded)
            neighbor_actions = actions[:, neighbor_ids]
            neighbor_actions_onehot = F.one_hot(
                neighbor_actions, num_classes=self.n_actions
            ).float().reshape(self.config.batch_size, -1)
            
            # Get next actions from target policies
            next_neighbor_actions = []
            for nid in neighbor_ids:
                next_action = self.agents[nid].get_target_action(
                    next_local_obs[:, nid, :]
                )
                next_neighbor_actions.append(next_action)
            
            next_neighbor_actions = torch.stack(next_neighbor_actions, dim=1)
            next_neighbor_actions_onehot = F.one_hot(
                next_neighbor_actions, num_classes=self.n_actions
            ).float().reshape(self.config.batch_size, -1)
            
            # Compute truncated reward (averaged from κ_r-hop neighbors)
            truncated_rewards = self.neighbor_extractor.compute_truncated_reward(
                rewards, agent_id
            )
            
            # Update critic
            critic_loss = agent.update_critic(
                neighbor_obs=neighbor_obs,
                neighbor_actions_onehot=neighbor_actions_onehot,
                rewards=truncated_rewards,
                next_neighbor_obs=next_neighbor_obs,
                next_neighbor_actions_onehot=next_neighbor_actions_onehot,
                dones=dones[:, 0]
            )
            critic_losses.append(critic_loss)
            
            # Update actor
            agent_local_obs = local_obs[:, agent_id, :]
            actor_loss = agent.update_actor(
                local_obs=agent_local_obs,
                neighbor_obs=neighbor_obs,
                neighbor_actions_onehot=neighbor_actions_onehot
            )
            actor_losses.append(actor_loss)
            
            # Soft update target networks
            agent.soft_update()
        
        return critic_losses, actor_losses
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate"""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
    
    def save(self, path: str) -> None:
        """
        Save all agents and training state
        
        Args:
            path: Directory path to save models
        """
        os.makedirs(path, exist_ok=True)
        
        # Save each agent
        for agent_id, agent in enumerate(self.agents):
            agent_path = os.path.join(path, f"agent_{agent_id}.pt")
            agent.save(agent_path)
        
        # Save training state
        state = {
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }
        state_path = os.path.join(path, "training_state.pt")
        torch.save(state, state_path)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load all agents and training state
        
        Args:
            path: Directory path to load models from
        """
        # Load each agent
        for agent_id, agent in enumerate(self.agents):
            agent_path = os.path.join(path, f"agent_{agent_id}.pt")
            agent.load(agent_path)
        
        # Load training state
        state_path = os.path.join(path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.epsilon = state['epsilon']
            self.total_steps = state['total_steps']
        
        print(f"Model loaded from {path}")
    
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        return {
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'buffer_size': len(self.replay_buffer)
        }