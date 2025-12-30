"""
Network Utilities for NMARL Algorithm.

This module provides utilities for handling the communication network graph,
including neighborhood computation and κ-hop neighbor finding.
"""

import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict


class CommunicationNetwork:
    """
    Handles the communication network topology among agents.
    
    In the wireless communication environment, agents are arranged in a grid.
    The communication graph defines which agents can share information.
    """
    
    def __init__(self, grid_x: int, grid_y: int, topology: str = "grid"):
        """
        Initialize the communication network.
        
        Args:
            grid_x: Number of agents in x-direction
            grid_y: Number of agents in y-direction
            topology: Network topology type ("grid", "ring", "full")
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.n_agents = grid_x * grid_y
        self.topology = topology
        
        # Build adjacency list
        self.adjacency: Dict[int, Set[int]] = self._build_adjacency()
        
        # Cache for κ-hop neighborhoods
        self._neighborhood_cache: Dict[Tuple[int, int], Set[int]] = {}
    
    def _build_adjacency(self) -> Dict[int, Set[int]]:
        """Build the adjacency list based on topology."""
        adjacency = defaultdict(set)
        
        if self.topology == "grid":
            # 4-connected grid topology
            for i in range(self.n_agents):
                row = i // self.grid_x
                col = i % self.grid_x
                
                # Up neighbor
                if row > 0:
                    adjacency[i].add(i - self.grid_x)
                # Down neighbor
                if row < self.grid_y - 1:
                    adjacency[i].add(i + self.grid_x)
                # Left neighbor
                if col > 0:
                    adjacency[i].add(i - 1)
                # Right neighbor
                if col < self.grid_x - 1:
                    adjacency[i].add(i + 1)
        
        elif self.topology == "ring":
            # Ring topology (circular)
            for i in range(self.n_agents):
                adjacency[i].add((i - 1) % self.n_agents)
                adjacency[i].add((i + 1) % self.n_agents)
        
        elif self.topology == "full":
            # Fully connected
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if i != j:
                        adjacency[i].add(j)
        
        return dict(adjacency)
    
    def get_neighbors(self, agent_id: int) -> Set[int]:
        """Get immediate (1-hop) neighbors of an agent."""
        return self.adjacency.get(agent_id, set())
    
    def get_kappa_hop_neighbors(self, agent_id: int, kappa: int, 
                                include_self: bool = True) -> Set[int]:
        """
        Get κ-hop neighborhood of an agent.
        
        This implements N_i^κ from the paper: agents whose graph distance
        to i is less than or equal to κ.
        
        Args:
            agent_id: The agent index
            kappa: Number of hops
            include_self: Whether to include the agent itself
            
        Returns:
            Set of agent indices in the κ-hop neighborhood
        """
        cache_key = (agent_id, kappa)
        if cache_key in self._neighborhood_cache:
            neighbors = self._neighborhood_cache[cache_key].copy()
            if not include_self:
                neighbors.discard(agent_id)
            return neighbors
        
        # BFS to find all agents within κ hops
        visited = {agent_id}
        frontier = {agent_id}
        
        for _ in range(kappa):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.adjacency.get(node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        
        # Cache the result
        self._neighborhood_cache[cache_key] = visited.copy()
        
        if not include_self:
            visited.discard(agent_id)
        
        return visited
    
    def get_neighborhood_excluding(self, agent_id: int, kappa: int, 
                                   exclude: int) -> Set[int]:
        """
        Get κ-hop neighborhood excluding a specific agent.
        
        Implements N_{i,-j}^κ from the paper.
        """
        neighbors = self.get_kappa_hop_neighbors(agent_id, kappa)
        neighbors.discard(exclude)
        return neighbors
    
    def get_agents_outside_neighborhood(self, agent_id: int, kappa: int) -> Set[int]:
        """
        Get agents outside the κ-hop neighborhood.
        
        Implements N_{-i}^κ from the paper.
        """
        in_neighborhood = self.get_kappa_hop_neighbors(agent_id, kappa)
        all_agents = set(range(self.n_agents))
        return all_agents - in_neighborhood
    
    def get_graph_distance(self, agent_i: int, agent_j: int) -> int:
        """Compute the graph distance between two agents."""
        if agent_i == agent_j:
            return 0
        
        visited = {agent_i}
        frontier = {agent_i}
        distance = 0
        
        while frontier:
            distance += 1
            next_frontier = set()
            for node in frontier:
                for neighbor in self.adjacency.get(node, set()):
                    if neighbor == agent_j:
                        return distance
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        
        return float('inf')  # Not connected
    
    def agent_position(self, agent_id: int) -> Tuple[int, int]:
        """Get the (row, col) position of an agent in the grid."""
        row = agent_id // self.grid_x
        col = agent_id % self.grid_x
        return (row, col)
    
    def agent_from_position(self, row: int, col: int) -> int:
        """Get agent index from grid position."""
        return row * self.grid_x + col
    
    def get_access_points_for_agent(self, agent_id: int) -> List[int]:
        """
        Get access point indices that an agent can transmit to.
        
        In the wireless environment, access points are at grid intersections.
        """
        row, col = self.agent_position(agent_id)
        aps = []
        
        # Access points are indexed based on their position
        # AP at (ap_x, ap_y) has index: ap_y * (grid_x - 1) + ap_x + 1
        
        # Upper-left AP (if exists)
        if row > 0 and col > 0:
            aps.append((row - 1) * (self.grid_x - 1) + (col - 1) + 1)
        # Lower-left AP (if exists)
        if row < self.grid_y - 1 and col > 0:
            aps.append(row * (self.grid_x - 1) + (col - 1) + 1)
        # Upper-right AP (if exists)
        if row > 0 and col < self.grid_x - 1:
            aps.append((row - 1) * (self.grid_x - 1) + col + 1)
        # Lower-right AP (if exists)
        if row < self.grid_y - 1 and col < self.grid_x - 1:
            aps.append(row * (self.grid_x - 1) + col + 1)
        
        return aps


class StateAggregator:
    """
    Handles state aggregation for truncated Q-function computation.
    
    This implements the surjection functions h_1^i and h_2^i from the paper,
    which project global states/actions to local neighborhoods.
    """
    
    def __init__(self, network: CommunicationNetwork, 
                 kappa_o: List[int], kappa_r: List[int]):
        """
        Initialize the state aggregator.
        
        Args:
            network: Communication network
            kappa_o: Observation range for each agent
            kappa_r: Reward communication range for each agent
        """
        self.network = network
        self.kappa_o = kappa_o
        self.kappa_r = kappa_r
        self.n_agents = network.n_agents
        
        # Precompute neighborhoods for efficiency
        self.neighborhoods_o = [
            network.get_kappa_hop_neighbors(i, kappa_o[i])
            for i in range(self.n_agents)
        ]
        self.neighborhoods_r = [
            network.get_kappa_hop_neighbors(i, kappa_r[i])
            for i in range(self.n_agents)
        ]
    
    def extract_local_state(self, global_state: np.ndarray, 
                           agent_id: int) -> np.ndarray:
        """
        Extract local state for agent's κ_o-hop neighborhood.
        
        Args:
            global_state: Shape (n_agents, state_dim) or (n_agents,)
            agent_id: Agent index
            
        Returns:
            Local state for the neighborhood
        """
        neighbors = sorted(self.neighborhoods_o[agent_id])
        if global_state.ndim == 1:
            return global_state[neighbors]
        return global_state[neighbors]
    
    def extract_local_action(self, global_action: np.ndarray,
                            agent_id: int) -> np.ndarray:
        """
        Extract local actions for agent's κ_o-hop neighborhood.
        """
        neighbors = sorted(self.neighborhoods_o[agent_id])
        if global_action.ndim == 1:
            return global_action[neighbors]
        return global_action[neighbors]
    
    def extract_local_rewards(self, global_rewards: np.ndarray,
                             agent_id: int) -> np.ndarray:
        """
        Extract rewards from agent's κ_r-hop neighborhood.
        
        Returns the sum/average of rewards from κ_r-hop neighbors.
        """
        neighbors = sorted(self.neighborhoods_r[agent_id])
        return global_rewards[neighbors]
    
    def compute_truncated_reward(self, global_rewards: np.ndarray,
                                agent_id: int) -> float:
        """
        Compute the truncated reward for agent i.
        
        This is (1/N) * sum_{j in N_i^{κ_r}} r_j as per equation (8).
        """
        local_rewards = self.extract_local_rewards(global_rewards, agent_id)
        return local_rewards.sum() / self.n_agents
    
    def get_local_state_action_key(self, global_state: np.ndarray,
                                   global_action: np.ndarray,
                                   agent_id: int) -> Tuple:
        """
        Create a hashable key for the local state-action pair.
        
        This is used for the tabular Q-function.
        """
        local_state = self.extract_local_state(global_state, agent_id)
        local_action = self.extract_local_action(global_action, agent_id)
        return (tuple(local_state.flatten()), tuple(local_action.flatten()))