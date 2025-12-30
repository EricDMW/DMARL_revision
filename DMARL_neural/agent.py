# ============================================================================
# File: agent.py
# Description: Single DMARL Agent implementation
# ============================================================================
"""
DMARL Agent Module

This module implements a single agent in the DMARL framework.
Each agent has:
- Actor network for decentralized policy
- Critic network for centralized value estimation
- Target networks for stable learning
- Update methods for actor and critic

Key features:
- Asymmetric information: Actor uses local obs, Critic uses neighbor info
- No Q-value exchange between agents
- Soft target updates for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from typing import Tuple, Optional

from networks import ActorNetwork, CriticNetwork
from utils import NeighborExtractor
from config import Config


class DMARLAgent:
    """
    Multi-Agent Deep Deterministic Policy Gradient Agent
    
    Implements actor-critic with:
    - Decentralized actor: π_i(a_i | s_i; θ_i)
    - Centralized critic: Q_i(s_{N_i^κ_o}, a_{N_i^κ_o}; ω_i)
    - Target networks with soft updates
    
    Attributes:
        agent_id: Unique identifier for this agent
        actor: Policy network
        critic: Q-value network
        actor_target, critic_target: Target networks
        actor_optimizer, critic_optimizer: Optimizers
    """
    
    def __init__(
        self,
        agent_id: int,
        local_obs_dim: int,
        neighbor_obs_dim: int,
        n_neighbors: int,
        n_actions: int,
        config: Config,
        neighbor_extractor: NeighborExtractor
    ):
        """
        Initialize DMARL Agent
        
        Args:
            agent_id: Agent index
            local_obs_dim: Dimension of local observation (for actor)
            neighbor_obs_dim: Dimension of neighbor observations (for critic)
            n_neighbors: Number of neighbors in observation range
            n_actions: Number of possible actions
            config: Configuration object
            neighbor_extractor: NeighborExtractor instance for topology info
        """
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.config = config
        self.device = config.device
        self.neighbor_extractor = neighbor_extractor
        
        # Get self index in neighbor list (for actor update)
        self.self_idx_in_neighbors = neighbor_extractor.get_self_index_in_neighbors(agent_id)
        
        # ==================
        # Actor Network
        # ==================
        # Decentralized policy - uses only local observation
        self.actor = ActorNetwork(
            local_obs_dim=local_obs_dim,
            n_actions=n_actions,
            hidden_dims=(config.hidden_dim_1, config.hidden_dim_2)
        ).to(self.device)
        
        self.actor_target = deepcopy(self.actor)
        self._freeze_target(self.actor_target)
        
        # ==================
        # Critic Network
        # ==================
        # Centralized Q-function - uses neighbor information
        self.critic = CriticNetwork(
            neighbor_obs_dim=neighbor_obs_dim,
            n_neighbors=n_neighbors,
            n_actions=n_actions,
            hidden_dims=(config.hidden_dim_1, config.hidden_dim_2, config.hidden_dim_3)
        ).to(self.device)
        
        self.critic_target = deepcopy(self.critic)
        self._freeze_target(self.critic_target)
        
        # ==================
        # Optimizers
        # ==================
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.lr_actor
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.lr_critic
        )
    
    def _freeze_target(self, network: nn.Module) -> None:
        """Freeze target network parameters"""
        for param in network.parameters():
            param.requires_grad = False
    
    def select_action(
        self,
        local_obs: np.ndarray,
        epsilon: float = 0.0,
        deterministic: bool = False
    ) -> int:
        """
        Select action based on local observation (decentralized execution)
        
        Args:
            local_obs: Local observation for this agent
            epsilon: Exploration rate
            deterministic: If True, always select greedy action
            
        Returns:
            action: Selected action (integer)
        """
        with torch.no_grad():
            if isinstance(local_obs, np.ndarray):
                local_obs = torch.FloatTensor(local_obs).to(self.device)
            
            action = self.actor.get_action(local_obs, epsilon, deterministic)
            
            if isinstance(action, torch.Tensor):
                return action.cpu().numpy().item()
            return action
    
    def update_critic(
        self,
        neighbor_obs: torch.Tensor,
        neighbor_actions_onehot: torch.Tensor,
        rewards: torch.Tensor,
        next_neighbor_obs: torch.Tensor,
        next_neighbor_actions_onehot: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """
        Update critic network using TD learning
        
        Implements:
            δ_i = r̄_i + γ × Q_i^target(s'_{N_i}, a'_{N_i}) - Q_i(s_{N_i}, a_{N_i})
            ω_i ← ω_i + α × δ_i × ∇_{ω_i} Q_i
        
        Args:
            neighbor_obs: Neighbor observations [batch, neighbor_obs_dim]
            neighbor_actions_onehot: One-hot actions [batch, n_neighbors × n_actions]
            rewards: Truncated rewards [batch]
            next_neighbor_obs: Next neighbor observations [batch, neighbor_obs_dim]
            next_neighbor_actions_onehot: Next one-hot actions [batch, n_neighbors × n_actions]
            dones: Done flags [batch]
            
        Returns:
            critic_loss: Scalar loss value
        """
        # Current Q-value
        current_q = self.critic(neighbor_obs, neighbor_actions_onehot)
        
        # Target Q-value
        with torch.no_grad():
            target_q = self.critic_target(next_neighbor_obs, next_neighbor_actions_onehot)
            target_q = rewards.unsqueeze(1) + self.config.gamma * (1 - dones.unsqueeze(1)) * target_q
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_actor(
        self,
        local_obs: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_actions_onehot: torch.Tensor
    ) -> float:
        """
        Update actor network using policy gradient
        
        Implements:
            θ_i ← θ_i + η × E[Q_i(s_{N_i}, a_{N_i}) × ∇_{θ_i} log π_i(a_i | s_i)]
        
        Args:
            local_obs: Local observations [batch, local_obs_dim]
            neighbor_obs: Neighbor observations [batch, neighbor_obs_dim]
            neighbor_actions_onehot: Current neighbor actions [batch, n_neighbors × n_actions]
            
        Returns:
            actor_loss: Scalar loss value
        """
        batch_size = local_obs.size(0)
        n_neighbors = neighbor_actions_onehot.size(1) // self.n_actions
        
        # Get action probabilities from current policy
        action_probs = self.actor(local_obs)
        
        # Reshape neighbor actions to [batch, n_neighbors, n_actions]
        neighbor_actions_reshaped = neighbor_actions_onehot.reshape(
            batch_size, n_neighbors, self.n_actions
        )
        
        # Compute Q-value for each possible action of this agent
        q_values = []
        for action in range(self.n_actions):
            # Replace this agent's action with the current action
            modified_actions = neighbor_actions_reshaped.clone()
            action_onehot = torch.zeros(batch_size, self.n_actions).to(self.device)
            action_onehot[:, action] = 1.0
            modified_actions[:, self.self_idx_in_neighbors, :] = action_onehot
            modified_actions_flat = modified_actions.reshape(batch_size, -1)
            
            q = self.critic(neighbor_obs, modified_actions_flat)
            q_values.append(q)
        
        q_values = torch.cat(q_values, dim=1)  # [batch, n_actions]
        
        # Policy gradient: maximize expected Q-value
        expected_q = (action_probs * q_values).sum(dim=1)
        actor_loss = -expected_q.mean()
        
        # Add entropy regularization for exploration
        entropy = self.actor.get_entropy(local_obs).mean()
        actor_loss = actor_loss - self.config.entropy_coef * entropy
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def soft_update(self) -> None:
        """
        Soft update of target networks
        
        Implements:
            θ^- ← τ × θ + (1 - τ) × θ^-
            ω^- ← τ × ω + (1 - τ) × ω^-
        """
        tau = self.config.tau
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def get_target_action(self, local_obs: torch.Tensor) -> torch.Tensor:
        """
        Get action from target policy (for computing next Q-values)
        
        Args:
            local_obs: Local observation [batch, local_obs_dim]
            
        Returns:
            action: Greedy action from target policy [batch]
        """
        with torch.no_grad():
            action_probs = self.actor_target(local_obs)
            return torch.argmax(action_probs, dim=-1)
    
    def save(self, path: str) -> None:
        """Save agent state to file"""
        state = {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
        torch.save(state, path)
    
    def load(self, path: str) -> None:
        """Load agent state from file"""
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state['actor'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])