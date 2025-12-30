# ============================================================================
# File: agent.py
# Description: VDN Agent Implementation
# ============================================================================
"""
VDN Agent Module

This module implements a single agent for VDN (Value Decomposition Networks).
Each agent has:
- Actor network: π_i(a_i | s_i; θ_i)
- Individual Q-network: Q_i(s_i, a_i; ω_i)
- Target networks for stable learning

Key difference from DMARL:
- Uses individual Q-networks (not centralized critics)
- Joint Q-value: Q_tot = Σ_i Q_i(s_i, a_i)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from typing import Tuple, Optional

from networks import ActorNetwork, IndividualQNetwork
from config import Config


class VDNAgent:
    """
    VDN Agent for Multi-Agent Reinforcement Learning
    
    Implements actor-critic with:
    - Decentralized actor: π_i(a_i | s_i; θ_i)
    - Individual Q-network: Q_i(s_i, a_i; ω_i)
    - Target networks with soft updates
    
    Attributes:
        agent_id: Unique identifier for this agent
        actor: Policy network
        q_network: Individual Q-value network
        actor_target, q_target: Target networks
        actor_optimizer, q_optimizer: Optimizers
    """
    
    def __init__(
        self,
        agent_id: int,
        local_obs_dim: int,
        n_actions: int,
        config: Config
    ):
        """
        Initialize VDN Agent
        
        Args:
            agent_id: Agent index
            local_obs_dim: Dimension of local observation
            n_actions: Number of possible actions
            config: Configuration object
        """
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.config = config
        self.device = config.device
        
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
        
        # ==================
        # Individual Q-Network
        # ==================
        # Individual Q-network for VDN
        self.q_network = IndividualQNetwork(
            local_obs_dim=local_obs_dim,
            n_actions=n_actions,
            hidden_dims=(config.hidden_dim_1, config.hidden_dim_2, config.hidden_dim_3)
        ).to(self.device)
        
        self.q_target = deepcopy(self.q_network)
        
        # ==================
        # Optimizers
        # ==================
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.lr_actor
        )
        
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.lr_critic
        )
    
    def select_action(
        self,
        local_obs: np.ndarray,
        epsilon: float = 0.0,
        deterministic: bool = False
    ) -> int:
        """
        Select action using current policy
        
        Args:
            local_obs: Local observation
            epsilon: Exploration rate
            deterministic: If True, use greedy action
            
        Returns:
            action: Selected action
        """
        local_obs_tensor = torch.FloatTensor(local_obs).to(self.device)
        
        with torch.no_grad():
            action = self.actor.get_action(
                local_obs_tensor,
                epsilon=epsilon,
                deterministic=deterministic
            )
        
        if isinstance(action, torch.Tensor):
            if action.dim() == 0:
                return action.cpu().numpy().item()
            return action
        return action
    
    def update_q_network(
        self,
        local_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_local_obs: torch.Tensor,
        next_actions: torch.Tensor,
        dones: torch.Tensor,
        joint_target_q: torch.Tensor
    ) -> float:
        """
        Update individual Q-network using VDN loss
        
        In VDN, the loss is computed on the joint Q-value:
            L = (Q_tot - Q_tot_target)^2
        where Q_tot = Σ_i Q_i(s_i, a_i)
        
        The gradient flows to each individual Q-network.
        
        Args:
            local_obs: Local observations [batch, local_obs_dim]
            actions: Actions taken [batch]
            rewards: Individual rewards [batch] (for this agent)
            next_local_obs: Next local observations [batch, local_obs_dim]
            next_actions: Next actions [batch]
            dones: Done flags [batch]
            joint_target_q: Joint target Q-value [batch] (computed externally)
            
        Returns:
            q_loss: Scalar loss value
        """
        # One-hot encode actions
        actions_onehot = F.one_hot(actions, num_classes=self.n_actions).float()
        next_actions_onehot = F.one_hot(next_actions, num_classes=self.n_actions).float()
        
        # Current Q-value for this agent
        current_q = self.q_network(local_obs, actions_onehot)
        
        # Target Q-value for this agent (from next state)
        with torch.no_grad():
            next_q = self.q_target(next_local_obs, next_actions_onehot)
            # Individual target: reward + gamma * next_q (if not done)
            target_q = rewards.unsqueeze(1) + self.config.gamma * (1 - dones.unsqueeze(1)) * next_q
        
        # Q-network loss (MSE)
        q_loss = F.mse_loss(current_q, target_q)
        
        # Update Q-network
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip)
        self.q_optimizer.step()
        
        return q_loss.item()
    
    def update_actor(
        self,
        local_obs: torch.Tensor
    ) -> float:
        """
        Update actor network using policy gradient
        
        The actor is updated to maximize the individual Q-value:
            θ_i ← θ_i + η × E[Q_i(s_i, a_i) × ∇_{θ_i} log π_i(a_i | s_i)]
        
        Args:
            local_obs: Local observations [batch, local_obs_dim]
            
        Returns:
            actor_loss: Scalar loss value
        """
        batch_size = local_obs.size(0)
        
        # Get action probabilities from current policy
        action_probs = self.actor(local_obs)
        
        # Compute Q-value for each possible action
        q_values = []
        for action in range(self.n_actions):
            action_onehot = torch.zeros(batch_size, self.n_actions).to(self.device)
            action_onehot[:, action] = 1.0
            
            q = self.q_network(local_obs, action_onehot)
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
    
    def get_q_value(
        self,
        local_obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Get Q-value for given observation and action
        
        Args:
            local_obs: Local observation [batch, local_obs_dim]
            action: Action [batch]
            
        Returns:
            q_value: Q-value [batch, 1]
        """
        action_onehot = F.one_hot(action, num_classes=self.n_actions).float()
        return self.q_network(local_obs, action_onehot)
    
    def get_target_q_value(
        self,
        local_obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Get target Q-value for given observation and action
        
        Args:
            local_obs: Local observation [batch, local_obs_dim]
            action: Action [batch]
            
        Returns:
            q_value: Target Q-value [batch, 1]
        """
        action_onehot = F.one_hot(action, num_classes=self.n_actions).float()
        with torch.no_grad():
            return self.q_target(local_obs, action_onehot)
    
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
        
        for param, target_param in zip(self.q_network.parameters(), self.q_target.parameters()):
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
            'q_network': self.q_network.state_dict(),
            'q_target': self.q_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict()
        }
        torch.save(state, path)
    
    def load(self, path: str) -> None:
        """Load agent state from file"""
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state['actor'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.q_network.load_state_dict(state['q_network'])
        self.q_target.load_state_dict(state['q_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.q_optimizer.load_state_dict(state['q_optimizer'])
