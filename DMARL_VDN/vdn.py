# ============================================================================
# File: vdn.py
# Description: VDN Multi-Agent System Manager
# ============================================================================
"""
VDN System Module

This module implements the multi-agent system manager for VDN (Value Decomposition Networks).
Key features:
- Each agent has an individual Q-network
- Joint Q-value: Q_tot(s, a) = Σ_i Q_i(s_i, a_i)
- Centralized training with decentralized execution
- Coordinates action selection and training updates
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import os

from config import Config
from buffer import ReplayBuffer
from agent import VDNAgent


class VDNSystem:
    """
    Multi-Agent VDN System Manager
    
    Coordinates training of multiple VDN agents on the
    wireless communication environment.
    
    Attributes:
        env: Environment instance
        config: Configuration
        n_agents: Number of agents
        agents: List of VDNAgent instances
        replay_buffer: Shared experience buffer
        epsilon: Current exploration rate
        total_steps: Total training steps
    """
    
    def __init__(self, env, config: Config):
        """
        Initialize Multi-Agent VDN System
        
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
    
    def _create_agents(self) -> List[VDNAgent]:
        """
        Create all agents with individual Q-networks
        
        Returns:
            List of initialized VDNAgent instances
        """
        agents = []
        
        for agent_id in range(self.n_agents):
            # Create agent with individual Q-network
            agent = VDNAgent(
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
        Update all agents using VDN algorithm
        
        In VDN:
        1. Compute joint Q-value: Q_tot = Σ_i Q_i(s_i, a_i)
        2. Compute joint target: Q_tot_target = Σ_i (r_i + γ * max_{a'_i} Q_i_target(s'_i, a'_i))
        3. Loss: L = (Q_tot - Q_tot_target)^2
        4. Gradients flow to individual Q-networks through backpropagation
        
        Returns:
            (q_losses, actor_losses): Lists of losses for each agent,
            or None if buffer not ready
        """
        if not self.replay_buffer.is_ready(self.config.batch_size):
            return None
        
        # Sample batch
        (obs, actions, rewards, next_obs, dones,
         local_obs, next_local_obs) = self.replay_buffer.sample(self.config.batch_size)
        
        batch_size = local_obs.size(0)
        
        # ============================================
        # Step 1: Compute joint Q-value Q_tot
        # ============================================
        # Q_tot = Σ_i Q_i(s_i, a_i)
        individual_q_values = []
        for agent_id, agent in enumerate(self.agents):
            agent_local_obs = local_obs[:, agent_id, :]
            agent_action = actions[:, agent_id]
            q_val = agent.get_q_value(agent_local_obs, agent_action)
            individual_q_values.append(q_val)
        
        # Sum individual Q-values to get joint Q-value
        q_tot = torch.stack(individual_q_values, dim=1).sum(dim=1)  # [batch]
        
        # ============================================
        # Step 2: Compute joint target Q_tot_target
        # ============================================
        # Q_tot_target = Σ_i (r_i + γ * max_{a'_i} Q_i_target(s'_i, a'_i))
        # In VDN, we use the total reward (sum of all agent rewards)
        total_rewards = rewards.sum(dim=1)  # [batch] - sum of all agent rewards
        
        individual_next_q_max = []
        for agent_id, agent in enumerate(self.agents):
            agent_next_local_obs = next_local_obs[:, agent_id, :]
            agent_done = dones[:, agent_id]
            
            # Compute max Q-value over all actions for next state
            next_q_max = agent.get_max_target_q_value(agent_next_local_obs)
            # Apply discount and done mask
            next_q_max_masked = (1 - agent_done.float()).unsqueeze(1) * next_q_max
            individual_next_q_max.append(next_q_max_masked)
        
        # Sum individual max Q-values
        sum_next_q_max = torch.stack(individual_next_q_max, dim=1).sum(dim=1).squeeze()  # [batch]
        
        # Joint target: total_reward + γ * sum of max Q-values
        q_tot_target = total_rewards + self.config.gamma * sum_next_q_max
        
        # ============================================
        # Step 3: Compute joint loss and backpropagate
        # ============================================
        # Loss: L = (Q_tot - Q_tot_target)^2
        q_loss = F.mse_loss(q_tot, q_tot_target)
        
        # Backpropagate - gradients will flow to each individual Q-network
        # Zero gradients for all agents
        for agent in self.agents:
            agent.q_optimizer.zero_grad()
        
        # Backward pass - gradients flow through the sum operation to each Q_i
        q_loss.backward()
        
        # Update each Q-network
        q_losses = []
        for agent in self.agents:
            torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), self.config.grad_clip)
            agent.q_optimizer.step()
            q_losses.append(q_loss.item())  # Same loss for all agents (joint loss)
        
        # ============================================
        # Step 4: Update actors independently
        # ============================================
        actor_losses = []
        for agent_id, agent in enumerate(self.agents):
            agent_local_obs = local_obs[:, agent_id, :]
            actor_loss = agent.update_actor(agent_local_obs)
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
