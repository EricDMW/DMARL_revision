# ============================================================================
# File: buffer.py
# Description: Replay buffer for experience storage
# ============================================================================
"""
Replay Buffer module for DMARL

Implements experience replay buffer for off-policy multi-agent learning.
Stores transitions with support for:
- Global observations
- Local observations per agent
- Actions, rewards, and done flags
"""

import numpy as np
import torch
from collections import deque
from typing import Tuple, Dict, Any
import random


class ReplayBuffer:
    """
    Experience replay buffer for multi-agent reinforcement learning
    
    Stores transitions: (obs, actions, rewards, next_obs, dones, local_obs, next_local_obs)
    
    Attributes:
        capacity: Maximum buffer size
        n_agents: Number of agents
        buffer: Deque storing experiences
    """
    
    def __init__(self, capacity: int, n_agents: int, device: torch.device):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
            n_agents: Number of agents in the environment
            device: Torch device for tensor operations
        """
        self.capacity = capacity
        self.n_agents = n_agents
        self.device = device
        self.buffer = deque(maxlen=capacity)
    
    def push(
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
        Store a transition in the buffer
        
        Args:
            obs: Global observation
            actions: Actions for all agents [n_agents]
            rewards: Local rewards for all agents [n_agents]
            next_obs: Next global observation
            dones: Done flags [n_agents]
            local_obs: Local observations [n_agents, local_obs_dim]
            next_local_obs: Next local observations [n_agents, local_obs_dim]
        """
        experience = {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obs,
            'dones': dones,
            'local_obs': local_obs,
            'next_local_obs': next_local_obs
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of tensors: (obs, actions, rewards, next_obs, dones, local_obs, next_local_obs)
        """
        batch = random.sample(self.buffer, batch_size)
        
        obs = torch.FloatTensor(
            np.array([e['obs'] for e in batch])
        ).to(self.device)
        
        actions = torch.LongTensor(
            np.array([e['actions'] for e in batch])
        ).to(self.device)
        
        rewards = torch.FloatTensor(
            np.array([e['rewards'] for e in batch])
        ).to(self.device)
        
        next_obs = torch.FloatTensor(
            np.array([e['next_obs'] for e in batch])
        ).to(self.device)
        
        dones = torch.FloatTensor(
            np.array([e['dones'] for e in batch])
        ).to(self.device)
        
        local_obs = torch.FloatTensor(
            np.array([e['local_obs'] for e in batch])
        ).to(self.device)
        
        next_local_obs = torch.FloatTensor(
            np.array([e['next_local_obs'] for e in batch])
        ).to(self.device)
        
        return obs, actions, rewards, next_obs, dones, local_obs, next_local_obs
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training"""
        return len(self.buffer) >= batch_size
    
    def clear(self) -> None:
        """Clear all experiences from buffer"""
        self.buffer.clear()