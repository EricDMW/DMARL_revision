# ============================================================================
# File: utils.py
# Description: Utility functions for neighbor extraction and network topology
# ============================================================================
"""
Utility Module for DMARL

This module provides:
- NeighborExtractor: Handles κ_o-hop and κ_r-hop neighborhood extraction
- Helper functions for network topology operations

Key concepts:
- κ_o (kappa_o): Observation neighborhood radius
- κ_r (kappa_r): Reward neighborhood radius (κ_r ≤ κ_o)
- Asymmetric information structure from the NMARL paper
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


class NeighborExtractor:
    """
    Utility class for extracting neighbor information based on network topology
    
    Implements the asymmetric information structure where:
    - κ_o: observation neighborhood (agents can observe this range)
    - κ_r: reward neighborhood (rewards averaged from this range, κ_r ≤ κ_o)
    
    Attributes:
        grid_x, grid_y: Grid dimensions
        n_agents: Total number of agents
        kappa_o: Observation neighborhood radius
        kappa_r: Reward neighborhood radius
        obs_neighbors: Precomputed observation neighbors for each agent
        reward_neighbors: Precomputed reward neighbors for each agent
    """
    
    def __init__(
        self,
        grid_x: int,
        grid_y: int,
        kappa_o: int = 1,
        kappa_r: int = 1,
        device: torch.device = None
    ):
        """
        Initialize neighbor extractor
        
        Args:
            grid_x: Grid width (number of columns)
            grid_y: Grid height (number of rows)
            kappa_o: Observation neighborhood radius
            kappa_r: Reward neighborhood radius (must be ≤ kappa_o)
            device: Torch device
        """
        assert kappa_r <= kappa_o, "Reward neighborhood must be ≤ observation neighborhood"
        
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.n_agents = grid_x * grid_y
        self.kappa_o = kappa_o
        self.kappa_r = kappa_r
        self.device = device or torch.device('cpu')
        
        # Precompute neighbor indices for each agent
        self.obs_neighbors = self._compute_neighbors(kappa_o)
        self.reward_neighbors = self._compute_neighbors(kappa_r)
        
        # Precompute neighbor counts
        self.obs_neighbor_counts = {
            i: len(neighbors) for i, neighbors in self.obs_neighbors.items()
        }
        self.reward_neighbor_counts = {
            i: len(neighbors) for i, neighbors in self.reward_neighbors.items()
        }
    
    def _get_position(self, agent_id: int) -> Tuple[int, int]:
        """
        Convert agent ID to (row, col) position
        
        Args:
            agent_id: Agent index
            
        Returns:
            (row, col): Grid position
        """
        row = agent_id // self.grid_x
        col = agent_id % self.grid_x
        return row, col
    
    def _get_agent_id(self, row: int, col: int) -> int:
        """
        Convert (row, col) position to agent ID
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            agent_id: Agent index
        """
        return row * self.grid_x + col
    
    def _compute_neighbors(self, kappa: int) -> Dict[int, List[int]]:
        """
        Compute κ-hop neighbors for each agent
        
        Uses Chebyshev distance (max of row/col distance)
        
        Args:
            kappa: Neighborhood radius
            
        Returns:
            neighbors: Dict mapping agent_id -> list of neighbor agent_ids
        """
        neighbors = {}
        
        for agent_id in range(self.n_agents):
            row, col = self._get_position(agent_id)
            neighbor_list = []
            
            # Get all agents within κ-hop distance
            for dr in range(-kappa, kappa + 1):
                for dc in range(-kappa, kappa + 1):
                    nr, nc = row + dr, col + dc
                    # Check bounds
                    if 0 <= nr < self.grid_y and 0 <= nc < self.grid_x:
                        neighbor_id = self._get_agent_id(nr, nc)
                        neighbor_list.append(neighbor_id)
            
            neighbors[agent_id] = sorted(neighbor_list)
        
        return neighbors
    
    def get_obs_neighbor_indices(self, agent_id: int) -> List[int]:
        """
        Get indices of agents in observation neighborhood (κ_o-hop)
        
        Args:
            agent_id: Agent index
            
        Returns:
            List of neighbor agent indices
        """
        return self.obs_neighbors[agent_id]
    
    def get_reward_neighbor_indices(self, agent_id: int) -> List[int]:
        """
        Get indices of agents in reward neighborhood (κ_r-hop)
        
        Args:
            agent_id: Agent index
            
        Returns:
            List of neighbor agent indices
        """
        return self.reward_neighbors[agent_id]
    
    def get_self_index_in_neighbors(self, agent_id: int) -> int:
        """
        Get the index of the agent itself within its neighbor list
        
        Args:
            agent_id: Agent index
            
        Returns:
            Index of agent in its own neighbor list
        """
        return self.obs_neighbors[agent_id].index(agent_id)
    
    def extract_neighbor_obs(
        self,
        local_obs_all: torch.Tensor,
        agent_id: int
    ) -> torch.Tensor:
        """
        Extract neighbor observations for agent i
        
        Args:
            local_obs_all: Local observations for all agents
                          [n_agents, local_obs_dim] or [batch, n_agents, local_obs_dim]
            agent_id: Agent index
            
        Returns:
            neighbor_obs: Concatenated neighbor observations
                         [neighbor_obs_dim] or [batch, neighbor_obs_dim]
        """
        neighbor_ids = self.obs_neighbors[agent_id]
        
        if local_obs_all.dim() == 2:
            # Single observation: [n_agents, local_obs_dim]
            neighbor_obs = local_obs_all[neighbor_ids].flatten()
        else:
            # Batched: [batch, n_agents, local_obs_dim]
            neighbor_obs = local_obs_all[:, neighbor_ids, :].reshape(
                local_obs_all.size(0), -1
            )
        
        return neighbor_obs
    
    def extract_neighbor_actions(
        self,
        actions: torch.Tensor,
        agent_id: int,
        n_actions: int
    ) -> torch.Tensor:
        """
        Extract neighbor actions for agent i (one-hot encoded)
        
        Args:
            actions: All agent actions [n_agents] or [batch, n_agents]
            agent_id: Agent index
            n_actions: Number of possible actions
            
        Returns:
            neighbor_actions_onehot: One-hot encoded neighbor actions
                                    [n_neighbors * n_actions] or [batch, n_neighbors * n_actions]
        """
        neighbor_ids = self.obs_neighbors[agent_id]
        
        if actions.dim() == 1:
            # Single action set
            neighbor_actions = actions[neighbor_ids]
            onehot = F.one_hot(neighbor_actions, num_classes=n_actions).float()
            return onehot.flatten()
        else:
            # Batched actions
            batch_size = actions.size(0)
            neighbor_actions = actions[:, neighbor_ids]
            onehot = F.one_hot(neighbor_actions, num_classes=n_actions).float()
            return onehot.reshape(batch_size, -1)
    
    def compute_truncated_reward(
        self,
        local_rewards: torch.Tensor,
        agent_id: int
    ) -> torch.Tensor:
        """
        Compute truncated neighbors-averaged reward for agent i
        
        Implements: r̄_i = (1/|N_i^κ_r|) × Σ_{j ∈ N_i^κ_r} r_j
        
        Args:
            local_rewards: Local rewards for all agents
                          [n_agents] or [batch, n_agents]
            agent_id: Agent index
            
        Returns:
            truncated_reward: Averaged reward from κ_r-hop neighbors
                             scalar or [batch]
        """
        neighbor_ids = self.reward_neighbors[agent_id]
        
        if isinstance(local_rewards, np.ndarray):
            local_rewards = torch.FloatTensor(local_rewards).to(self.device)
        
        if local_rewards.dim() == 1:
            neighbor_rewards = local_rewards[neighbor_ids]
        else:
            neighbor_rewards = local_rewards[:, neighbor_ids]
        
        truncated_reward = neighbor_rewards.mean(dim=-1)
        return truncated_reward
    
    def get_neighbor_info(self, agent_id: int) -> Dict:
        """
        Get complete neighbor information for an agent
        
        Args:
            agent_id: Agent index
            
        Returns:
            Dict containing neighbor indices and counts
        """
        return {
            'obs_neighbors': self.obs_neighbors[agent_id],
            'reward_neighbors': self.reward_neighbors[agent_id],
            'n_obs_neighbors': self.obs_neighbor_counts[agent_id],
            'n_reward_neighbors': self.reward_neighbor_counts[agent_id],
            'self_index': self.get_self_index_in_neighbors(agent_id)
        }


def create_action_mask(grid_x: int, grid_y: int) -> np.ndarray:
    """
    Create action mask for each agent based on grid position
    
    Corner agents: 2 legal actions (Idle + 1 transmission)
    Edge agents: 3 legal actions (Idle + 2 transmissions)
    Center agents: 5 legal actions (all)
    
    Args:
        grid_x: Grid width
        grid_y: Grid height
        
    Returns:
        action_mask: Boolean mask [n_agents, n_actions]
    """
    n_agents = grid_x * grid_y
    n_actions = 5
    mask = np.ones((n_agents, n_actions), dtype=bool)
    
    for agent_id in range(n_agents):
        row = agent_id // grid_x
        col = agent_id % grid_x
        
        # Action 1: Upper-Left (illegal if on top or left edge)
        if row == 0 or col == 0:
            mask[agent_id, 1] = False
        
        # Action 2: Lower-Left (illegal if on left edge)
        if col == 0:
            mask[agent_id, 2] = False
        
        # Action 3: Upper-Right (illegal if on top edge)
        if row == 0:
            mask[agent_id, 3] = False
        
        # Action 4: Lower-Right (illegal if on bottom or right edge)
        if row == grid_y - 1 or col == grid_x - 1:
            mask[agent_id, 4] = False
    
    return mask