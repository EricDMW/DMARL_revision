# ============================================================================
# File: networks.py
# Description: Neural network architectures for Actor and Critic
# ============================================================================
"""
Neural Network Architectures for DMARL

This module defines:
- ActorNetwork: Decentralized policy π_i(a_i | s_i; θ_i)
- CriticNetwork: Centralized Q-function Q_i(s_{N_i}, a_{N_i}; ω_i)

Key design principles:
- Actor uses only local observation (decentralized execution)
- Critic uses neighborhood information (centralized training)
- Layer normalization for training stability
- Orthogonal weight initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class ActorNetwork(nn.Module):
    """
    Actor Network for Decentralized Policy
    
    Implements: π_i(a_i | s_i; θ_i)
    
    Input: Local observation s_i (only agent's own state)
    Output: Action probabilities over discrete action space
    
    This preserves decentralized execution - each agent's policy
    depends only on its local observation.
    
    Architecture:
        Input -> FC(hidden_1) -> LayerNorm -> ReLU
              -> FC(hidden_2) -> LayerNorm -> ReLU
              -> FC(n_actions) -> Softmax
    """
    
    def __init__(
        self,
        local_obs_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, int] = (256, 128)
    ):
        """
        Initialize Actor Network
        
        Args:
            local_obs_dim: Dimension of local observation
            n_actions: Number of discrete actions
            hidden_dims: Tuple of hidden layer dimensions
        """
        super(ActorNetwork, self).__init__()
        
        self.local_obs_dim = local_obs_dim
        self.n_actions = n_actions
        
        # Network layers
        self.fc1 = nn.Linear(local_obs_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], n_actions)
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        self.ln2 = nn.LayerNorm(hidden_dims[1])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights using orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Output layer with smaller weights for stable initial policy
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
    
    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            local_obs: Local observation [batch, local_obs_dim]
            
        Returns:
            action_probs: Action probabilities [batch, n_actions]
        """
        x = F.relu(self.ln1(self.fc1(local_obs)))
        x = F.relu(self.ln2(self.fc2(x)))
        logits = self.fc3(x)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs
    
    def get_action(
        self,
        local_obs: torch.Tensor,
        epsilon: float = 0.0,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Select action with epsilon-greedy exploration
        
        Args:
            local_obs: Local observation [local_obs_dim] or [batch, local_obs_dim]
            epsilon: Exploration rate (probability of random action)
            deterministic: If True, always select greedy action
            
        Returns:
            action: Selected action(s)
        """
        if local_obs.dim() == 1:
            local_obs = local_obs.unsqueeze(0)
        
        action_probs = self.forward(local_obs)
        
        if deterministic or epsilon == 0.0:
            action = torch.argmax(action_probs, dim=-1)
        elif torch.rand(1).item() < epsilon:
            # Random action
            action = torch.randint(0, self.n_actions, (local_obs.size(0),))
            action = action.to(local_obs.device)
        else:
            # Greedy action
            action = torch.argmax(action_probs, dim=-1)
        
        return action.squeeze()
    
    def get_log_prob(
        self,
        local_obs: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of actions
        
        Args:
            local_obs: Local observations [batch, local_obs_dim]
            actions: Actions [batch]
            
        Returns:
            log_probs: Log probabilities [batch]
        """
        action_probs = self.forward(local_obs)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
        return log_probs.squeeze()
    
    def get_entropy(self, local_obs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of action distribution
        
        Args:
            local_obs: Local observations [batch, local_obs_dim]
            
        Returns:
            entropy: Entropy values [batch]
        """
        action_probs = self.forward(local_obs)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
        return entropy


class CriticNetwork(nn.Module):
    """
    Critic Network for Centralized Q-function
    
    Implements: Q_i(s_{N_i^κ_o}, a_{N_i^κ_o}; ω_i)
    
    Input: Concatenated neighbor observations and one-hot encoded actions
    Output: Q-value estimate
    
    This implements the truncated neighbors-averaged Q-function where
    each agent's critic observes κ_o-hop neighborhood information.
    
    Architecture:
        Input (neighbor_obs + neighbor_actions_onehot)
              -> FC(hidden_1) -> LayerNorm -> ReLU
              -> FC(hidden_2) -> LayerNorm -> ReLU
              -> FC(hidden_3) -> LayerNorm -> ReLU
              -> FC(1) -> Q-value
    """
    
    def __init__(
        self,
        neighbor_obs_dim: int,
        n_neighbors: int,
        n_actions: int,
        hidden_dims: Tuple[int, int, int] = (256, 128, 64)
    ):
        """
        Initialize Critic Network
        
        Args:
            neighbor_obs_dim: Dimension of concatenated neighbor observations
            n_neighbors: Number of neighbors (including self)
            n_actions: Number of possible actions per agent
            hidden_dims: Tuple of hidden layer dimensions
        """
        super(CriticNetwork, self).__init__()
        
        self.neighbor_obs_dim = neighbor_obs_dim
        self.n_neighbors = n_neighbors
        self.n_actions = n_actions
        
        # Input: observations + one-hot actions for all neighbors
        input_dim = neighbor_obs_dim + n_neighbors * n_actions
        
        # Network layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], 1)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        self.ln2 = nn.LayerNorm(hidden_dims[1])
        self.ln3 = nn.LayerNorm(hidden_dims[2])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Output layer
        nn.init.orthogonal_(self.fc4.weight, gain=1.0)
    
    def forward(
        self,
        neighbor_obs: torch.Tensor,
        neighbor_actions_onehot: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            neighbor_obs: Neighbor observations [batch, neighbor_obs_dim]
            neighbor_actions_onehot: One-hot neighbor actions [batch, n_neighbors * n_actions]
            
        Returns:
            q_value: Q-value estimate [batch, 1]
        """
        x = torch.cat([neighbor_obs, neighbor_actions_onehot], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        q_value = self.fc4(x)
        return q_value