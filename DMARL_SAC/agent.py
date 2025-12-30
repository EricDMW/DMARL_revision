# ============================================================================
# File: agent.py
# Description: SAC Agent Implementation
# ============================================================================
"""
SAC Agent Module

This module implements a single agent for SAC (Soft Actor-Critic).
Each agent has:
- Actor network: π_i(a_i | s_i; θ_i)
- Two Q-networks: Q1_i(s_i, a_i; ω1_i) and Q2_i(s_i, a_i; ω2_i)
- Target Q-networks for stable learning
- Temperature parameter α for entropy regularization

Key features:
- Soft Q-learning with entropy regularization
- Automatic temperature tuning (optional)
- Two Q-networks for stability (take minimum)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from typing import Tuple, Optional

from networks import ActorNetwork, IndividualQNetwork
from config import Config


class SACAgent:
    """
    SAC Agent for Multi-Agent Reinforcement Learning
    
    Implements soft actor-critic with:
    - Decentralized actor: π_i(a_i | s_i; θ_i)
    - Two Q-networks: Q1_i(s_i, a_i) and Q2_i(s_i, a_i)
    - Target networks with soft updates
    - Entropy regularization with temperature α
    
    Attributes:
        agent_id: Unique identifier for this agent
        actor: Policy network
        q1_network, q2_network: Two Q-value networks
        q1_target, q2_target: Target Q-networks
        actor_optimizer, q1_optimizer, q2_optimizer: Optimizers
        alpha: Temperature parameter for entropy regularization
    """
    
    def __init__(
        self,
        agent_id: int,
        local_obs_dim: int,
        n_actions: int,
        config: Config
    ):
        """
        Initialize SAC Agent
        
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
        
        # Temperature parameter (entropy regularization coefficient)
        self.use_automatic_entropy_tuning = getattr(config, 'use_automatic_entropy_tuning', False)
        
        if self.use_automatic_entropy_tuning:
            # Learnable temperature parameter
            self.target_entropy = -np.log(1.0 / n_actions) * getattr(config, 'target_entropy_coef', 0.98)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=getattr(config, 'lr_alpha', 3e-4))
        else:
            # Fixed temperature
            self.alpha = getattr(config, 'alpha', 0.2)
        
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
        # Q-Networks (Two for stability)
        # ==================
        # First Q-network
        self.q1_network = IndividualQNetwork(
            local_obs_dim=local_obs_dim,
            n_actions=n_actions,
            hidden_dims=(config.hidden_dim_1, config.hidden_dim_2, config.hidden_dim_3)
        ).to(self.device)
        
        self.q1_target = deepcopy(self.q1_network)
        
        # Second Q-network
        self.q2_network = IndividualQNetwork(
            local_obs_dim=local_obs_dim,
            n_actions=n_actions,
            hidden_dims=(config.hidden_dim_1, config.hidden_dim_2, config.hidden_dim_3)
        ).to(self.device)
        
        self.q2_target = deepcopy(self.q2_network)
        
        # ==================
        # Optimizers
        # ==================
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.lr_actor
        )
        
        self.q1_optimizer = torch.optim.Adam(
            self.q1_network.parameters(),
            lr=config.lr_critic
        )
        
        self.q2_optimizer = torch.optim.Adam(
            self.q2_network.parameters(),
            lr=config.lr_critic
        )
    
    @property
    def alpha_value(self) -> float:
        """Get current temperature value"""
        if self.use_automatic_entropy_tuning:
            return self.log_alpha.exp().item()
        return self.alpha
    
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
            epsilon: Exploration rate (for epsilon-greedy fallback)
            deterministic: If True, use greedy action
            
        Returns:
            action: Selected action
        """
        local_obs_tensor = torch.FloatTensor(local_obs).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action = self.actor.get_action(
                    local_obs_tensor,
                    epsilon=0.0,
                    deterministic=True
                )
            else:
                # SAC uses stochastic policy, but for discrete actions we can use epsilon-greedy
                action = self.actor.get_action(
                    local_obs_tensor,
                    epsilon=epsilon,
                        deterministic=False
            )
        
        if isinstance(action, torch.Tensor):
            if action.dim() == 0:
                return action.cpu().numpy().item()
            return action
        return action
    
    def update_q_networks(
        self,
        local_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_local_obs: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """
        Update Q-networks using soft Q-learning
        
        Soft Q-learning update:
            Q(s,a) = r + γ * E[min(Q1(s',a'), Q2(s',a')) - α * log π(a'|s')]
        
        where expectation E[·] is over a' ~ π(·|s').
        Uses two Q-networks and takes minimum for stability.
        
        Args:
            local_obs: Local observations [batch, local_obs_dim]
            actions: Actions taken [batch]
            rewards: Individual rewards [batch]
            next_local_obs: Next local observations [batch, local_obs_dim]
            dones: Done flags [batch]
            
        Returns:
            q_loss: Average Q-loss value
        """
        actions_onehot = F.one_hot(actions, num_classes=self.n_actions).float()
        
        # Current Q-values
        current_q1 = self.q1_network(local_obs, actions_onehot)
        current_q2 = self.q2_network(local_obs, actions_onehot)
        
        # Compute target Q-value using soft Q-learning
        with torch.no_grad():
            # Get next action probabilities from target policy
            next_action_probs = self.actor_target(next_local_obs)  # [batch, n_actions]
            
            # Compute Q-values for all next actions
            next_q1_values = []
            next_q2_values = []
            for action in range(self.n_actions):
                action_onehot = torch.zeros(actions.size(0), self.n_actions).to(self.device)
                action_onehot[:, action] = 1.0
                
                q1_val = self.q1_target(next_local_obs, action_onehot)
                q2_val = self.q2_target(next_local_obs, action_onehot)
                next_q1_values.append(q1_val)
                next_q2_values.append(q2_val)
            
            next_q1_all = torch.cat(next_q1_values, dim=1)  # [batch, n_actions]
            next_q2_all = torch.cat(next_q2_values, dim=1)  # [batch, n_actions]
            
            # Take minimum of two Q-networks
            next_q_min = torch.min(next_q1_all, next_q2_all)  # [batch, n_actions]
            
            # Compute soft Q-value: E[Q(s',a') - α * log π(a'|s')]
            # where expectation is over a' ~ π(·|s')
            log_probs = torch.log(next_action_probs + 1e-8)
            soft_q = (next_action_probs * (next_q_min - self.alpha_value * log_probs)).sum(dim=1, keepdim=True)
            
            # Target: r + γ * soft_q (if not done)
            target_q = rewards.unsqueeze(1) + self.config.gamma * (1 - dones.unsqueeze(1)) * soft_q
        
        # Q-network losses (MSE)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1_network.parameters(), self.config.grad_clip)
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2_network.parameters(), self.config.grad_clip)
        self.q2_optimizer.step()
        
        return (q1_loss.item() + q2_loss.item()) / 2.0
    
    def update_actor(
        self,
        local_obs: torch.Tensor
    ) -> float:
        """
        Update actor network using soft policy gradient
        
        The actor is updated to maximize:
            E[Q(s,a) - α * log π(a|s)]
        where a ~ π(·|s) and Q = min(Q1, Q2)
        
        Args:
            local_obs: Local observations [batch, local_obs_dim]
            
        Returns:
            actor_loss: Scalar loss value
        """
        batch_size = local_obs.size(0)
        
        # Get action probabilities from current policy
        action_probs = self.actor(local_obs)  # [batch, n_actions]
        log_probs = torch.log(action_probs + 1e-8)
        
        # Compute Q-values for all actions
        q1_values = []
        q2_values = []
        for action in range(self.n_actions):
            action_onehot = torch.zeros(batch_size, self.n_actions).to(self.device)
            action_onehot[:, action] = 1.0
            
            q1_val = self.q1_network(local_obs, action_onehot)
            q2_val = self.q2_network(local_obs, action_onehot)
            q1_values.append(q1_val)
            q2_values.append(q2_val)
        
        q1_all = torch.cat(q1_values, dim=1)  # [batch, n_actions]
        q2_all = torch.cat(q2_values, dim=1)  # [batch, n_actions]
        
        # Take minimum of two Q-networks
        q_min = torch.min(q1_all, q2_all)  # [batch, n_actions]
        
        # Compute expected value: E[Q(s,a) - α * log π(a|s)]
        expected_value = (action_probs * (q_min - self.alpha_value * log_probs)).sum(dim=1)
        
        # Actor loss: minimize negative expected value (maximize expected value)
        actor_loss = -expected_value.mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def update_temperature(self, local_obs: torch.Tensor) -> None:
        """
        Update temperature parameter α (if using automatic tuning)
        
        Minimizes: α * (log π(a|s) + target_entropy)
        
        Args:
            local_obs: Local observations [batch, local_obs_dim]
        """
        if not self.use_automatic_entropy_tuning:
            return
        
        with torch.no_grad():
            action_probs = self.actor(local_obs)
            log_probs = torch.log(action_probs + 1e-8)
            current_entropy = -(action_probs * log_probs).sum(dim=1).mean()
        
        # Temperature loss
        alpha_loss = -(self.log_alpha * (current_entropy + self.target_entropy).detach())
        
        # Update temperature
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
    
    def get_q_value(
        self,
        local_obs: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Get Q-value for given observation and action (minimum of Q1 and Q2)
        
        Args:
            local_obs: Local observation [batch, local_obs_dim]
            action: Action [batch]
            
        Returns:
            q_value: Q-value [batch, 1]
        """
        action_onehot = F.one_hot(action, num_classes=self.n_actions).float()
        q1 = self.q1_network(local_obs, action_onehot)
        q2 = self.q2_network(local_obs, action_onehot)
        return torch.min(q1, q2)
    
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
        
        for param, target_param in zip(self.q1_network.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(self.q2_network.parameters(), self.q2_target.parameters()):
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
            'q1_network': self.q1_network.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_network': self.q2_network.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict()
        }
        
        if self.use_automatic_entropy_tuning:
            state['log_alpha'] = self.log_alpha
            state['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        else:
            state['alpha'] = self.alpha
        
        torch.save(state, path)
    
    def load(self, path: str) -> None:
        """Load agent state from file"""
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state['actor'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.q1_network.load_state_dict(state['q1_network'])
        self.q1_target.load_state_dict(state['q1_target'])
        self.q2_network.load_state_dict(state['q2_network'])
        self.q2_target.load_state_dict(state['q2_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.q1_optimizer.load_state_dict(state['q1_optimizer'])
        self.q2_optimizer.load_state_dict(state['q2_optimizer'])
        
        if self.use_automatic_entropy_tuning and 'log_alpha' in state:
            self.log_alpha = state['log_alpha']
            self.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
        elif 'alpha' in state:
            self.alpha = state['alpha']
