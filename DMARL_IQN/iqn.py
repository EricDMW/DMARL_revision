# ============================================================================
# File: iqn.py
# Description: IQN Multi-Agent System Manager
# ============================================================================
"""
IQN System Module

This module implements the multi-agent system manager for IQN (Independent Q-Network).
Key features:
- Each agent has an individual Q-network
- Each agent learns completely independently
- No joint value computation - fully decentralized learning
- Coordinates action selection and training updates
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import os

from config import Config
from buffer import ReplayBuffer
from agent import IQNAgent


class IQNSystem:
    """
    Multi-Agent IQN System Manager
    
    Coordinates training of multiple IQN agents on the
    wireless communication environment.
    
    In IQN, each agent learns completely independently:
    - Each agent has its own Q-network Q_i(s_i, a_i)
    - No joint value computation
    - Each agent updates based only on its own experience
    
    Attributes:
        env: Environment instance
        config: Configuration
        n_agents: Number of agents
        agents: List of IQNAgent instances
        replay_buffer: Shared experience buffer
        epsilon: Current exploration rate
        total_steps: Total training steps
    """
    
    def __init__(self, env, config: Config):
        """
        Initialize Multi-Agent IQN System
        
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
    
    def _create_agents(self) -> List[IQNAgent]:
        """
        Create all agents with individual Q-networks
        
        Returns:
            List of initialized IQNAgent instances
        """
        agents = []
        
        for agent_id in range(self.n_agents):
            # Create agent with individual Q-network
            agent = IQNAgent(
                agent_id=agent_id,
                local_obs_dim=self.local_obs_dim,
                n_actions=self.n_actions,
                config=self.config
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
        Update all agents using IQN algorithm
        
        In IQN, each agent learns completely independently:
        1. Each agent updates its Q-network: Q_i(s_i, a_i)
        2. Target: r_i + γ * Q_i_target(s'_i, a'_i)
        3. Loss: L_i = (Q_i - target_Q_i)^2
        4. No joint value computation - fully independent learning
        
        Returns:
            (q_losses, actor_losses): Lists of losses for each agent,
            or None if buffer not ready
        """
        if not self.replay_buffer.is_ready(self.config.batch_size):
            return None
        
        # Sample batch
        (obs, actions, rewards, next_obs, dones,
         local_obs, next_local_obs) = self.replay_buffer.sample(self.config.batch_size)
        
        q_losses = []
        actor_losses = []
        
        # Get next actions from target policies
        next_actions = []
        for agent_id, agent in enumerate(self.agents):
            next_action = agent.get_target_action(next_local_obs[:, agent_id, :])
            next_actions.append(next_action)
        next_actions = torch.stack(next_actions, dim=1)  # [batch, n_agents]
        
        # Update each agent independently
        for agent_id, agent in enumerate(self.agents):
            agent_local_obs = local_obs[:, agent_id, :]
            agent_action = actions[:, agent_id]
            agent_reward = rewards[:, agent_id]
            agent_next_local_obs = next_local_obs[:, agent_id, :]
            agent_next_action = next_actions[:, agent_id]
            agent_done = dones[:, agent_id]
            
            # Update Q-network independently
            q_loss = agent.update_q_network(
                local_obs=agent_local_obs,
                actions=agent_action,
                rewards=agent_reward,
                next_local_obs=agent_next_local_obs,
                next_actions=agent_next_action,
                dones=agent_done
            )
            q_losses.append(q_loss)
            
            # Update actor independently
            actor_loss = agent.update_actor(
                local_obs=agent_local_obs
            )
            actor_losses.append(actor_loss)
            
            # Soft update target networks
            agent.soft_update()
        
        return q_losses, actor_losses
    
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
