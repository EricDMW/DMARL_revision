# ============================================================================
# File: config.py
# Description: Configuration and hyperparameters for VDN
# ============================================================================
"""
Configuration module for VDN on Wireless Communication Environment

This file contains all hyperparameters and configuration settings.
Modify this file to adjust training parameters.
"""

import torch


class Config:
    """
    Hyperparameters for VDN algorithm
    
    Attributes:
        Environment Parameters:
            grid_x, grid_y: Grid dimensions
            n_agents: Total number of agents
            n_obs_neighbors: Observation neighborhood radius (κ_o)
            ddl: Deadline horizon
            
        Network Architecture:
            hidden_dim_1, hidden_dim_2, hidden_dim_3: Hidden layer dimensions
            
        Training Parameters:
            lr_actor, lr_critic: Learning rates
            gamma: Discount factor
            tau: Soft update coefficient
            
        Replay Buffer:
            buffer_size: Maximum buffer capacity
            batch_size: Training batch size
            
        Training Schedule:
            max_episodes: Total training episodes
            max_steps: Maximum steps per episode
            update_freq: Network update frequency
            warmup_steps: Steps before training starts
            
        Exploration:
            epsilon_start, epsilon_end: Exploration bounds
            epsilon_decay: Decay rate
            
        Logging:
            log_freq: Logging frequency
            save_freq: Model saving frequency
    """
    
    # ====================
    # Environment Settings
    # ====================
    grid_x: int = 5
    grid_y: int = 5
    n_obs_neighbors: int = 1  # κ_o: observation neighborhood radius
    n_reward_neighbors: int = 1  # κ_r: reward neighborhood radius (κ_r ≤ κ_o)
    ddl: int = 2  # deadline horizon
    max_steps: int = 50
    
    # Computed
    @property
    def n_agents(self) -> int:
        return self.grid_x * self.grid_y
    
    @property
    def n_actions(self) -> int:
        return 5  # Idle + 4 transmission directions
    
    @property
    def local_obs_dim(self) -> int:
        """
        Local observation dimension.
        
        Each agent observes a (2*κ_o+1) x (2*κ_o+1) grid of neighbors,
        with ddl values per cell.
        """
        obs_grid_size = (2 * self.n_obs_neighbors + 1) ** 2
        return self.ddl * obs_grid_size
    
    # ====================
    # Network Architecture
    # ====================
    hidden_dim_1: int = 256
    hidden_dim_2: int = 128
    hidden_dim_3: int = 64
    
    # ====================
    # Training Parameters
    # ====================
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    gamma: float = 0.95  # discount factor
    tau: float = 0.005  # soft update coefficient
    grad_clip: float = 1.0  # gradient clipping
    
    # ====================
    # Replay Buffer
    # ====================
    buffer_size: int = 100000
    batch_size: int = 256
    
    # ====================
    # Training Schedule
    # ====================
    max_episodes: int = 5000
    update_freq: int = 1  # update every N steps
    warmup_steps: int = 1000  # steps before training starts
    
    # ====================
    # Exploration
    # ====================
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995
    
    # ====================
    # Regularization
    # ====================
    entropy_coef: float = 0.01  # entropy regularization coefficient
    
    # ====================
    # Logging & Saving
    # ====================
    log_freq: int = 100
    save_freq: int = 500
    save_dir: str = "./checkpoints"
    
    # ====================
    # Device
    # ====================
    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  Algorithm: VDN (Value Decomposition Networks)\n"
            f"  Environment: {self.grid_x}x{self.grid_y} grid, {self.n_agents} agents\n"
            f"  Neighborhoods: κ_o={self.n_obs_neighbors}, κ_r={self.n_reward_neighbors}\n"
            f"  Training: {self.max_episodes} episodes, batch_size={self.batch_size}\n"
            f"  Learning rates: actor={self.lr_actor}, critic={self.lr_critic}\n"
            f"  Device: {self.device}\n"
            f")"
        )


# Default configuration instance
default_config = Config()
