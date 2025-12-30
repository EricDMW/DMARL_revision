"""
Configuration module for NMARL Algorithm.

This module contains all hyperparameters and configuration settings
for the Distributed and Scalable NMARL Algorithm under Asymmetric Information.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch


@dataclass
class EnvironmentConfig:
    """Configuration for the wireless communication environment."""
    grid_x: int = 4
    grid_y: int = 4
    ddl: int = 2  # Deadline horizon
    packet_arrival_probability: float = 0.8
    success_transmission_probability: float = 0.8
    n_obs_neighbors: int = 1
    max_iter: int = 50
    render_mode: Optional[str] = None
    device: str = "cpu"
    
    @property
    def n_agents(self) -> int:
        return self.grid_x * self.grid_y


@dataclass
class AlgorithmConfig:
    """Configuration for Algorithm 1 (Distributed and Scalable Algorithm)."""
    # Episode and iteration settings
    M: int = 1000  # Number of outer-loop episodes
    T: int = 50    # Length of each episode (critic step iterations)
    
    # Discount factor
    gamma: float = 0.9
    
    # Learning rates
    eta: float = 0.1     # Base learning rate for actor step
    h: float = 2.0       # Learning rate parameter for critic step
    t0: float = 10.0     # Learning rate parameter for critic step
    
    # Neighborhood ranges (asymmetric information)
    # κ_o^i: observation range (state-action pairs)
    # κ_r^i: reward communication range
    # If None, will be set uniformly for all agents
    kappa_o: Optional[List[int]] = None  # Default: 1-hop for all
    kappa_r: Optional[List[int]] = None  # Default: 0-hop (local) for all
    
    # Default uniform values if not specified per-agent
    default_kappa_o: int = 1
    default_kappa_r: int = 0
    
    # Softmax policy temperature
    temperature: float = 1.0
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Seed for reproducibility
    seed: int = 42


@dataclass
class LoggingConfig:
    """Configuration for logging and visualization."""
    log_interval: int = 10      # Log every N episodes
    eval_interval: int = 50     # Evaluate every N episodes
    save_interval: int = 100    # Save model every N episodes
    
    # Plotting settings
    plot_rewards: bool = True
    plot_gradients: bool = True
    plot_q_values: bool = True
    
    # Output directory
    output_dir: str = "./results"
    experiment_name: str = "nmarl_experiment"


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    algo: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Initialize device and validate configuration."""
        # Set device
        if self.env.device == "auto":
            self.env.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize kappa values if not provided
        n_agents = self.env.n_agents
        if self.algo.kappa_o is None:
            self.algo.kappa_o = [self.algo.default_kappa_o] * n_agents
        if self.algo.kappa_r is None:
            self.algo.kappa_r = [self.algo.default_kappa_r] * n_agents
        
        # Validate kappa constraints: κ_r <= κ_o for all agents
        for i, (ko, kr) in enumerate(zip(self.algo.kappa_o, self.algo.kappa_r)):
            assert kr <= ko, f"Agent {i}: κ_r ({kr}) must be <= κ_o ({ko})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving."""
        return {
            "env": {
                "grid_x": self.env.grid_x,
                "grid_y": self.env.grid_y,
                "ddl": self.env.ddl,
                "packet_arrival_probability": self.env.packet_arrival_probability,
                "success_transmission_probability": self.env.success_transmission_probability,
                "n_obs_neighbors": self.env.n_obs_neighbors,
                "max_iter": self.env.max_iter,
                "device": self.env.device,
            },
            "algo": {
                "M": self.algo.M,
                "T": self.algo.T,
                "gamma": self.algo.gamma,
                "eta": self.algo.eta,
                "h": self.algo.h,
                "t0": self.algo.t0,
                "kappa_o": self.algo.kappa_o,
                "kappa_r": self.algo.kappa_r,
                "temperature": self.algo.temperature,
                "seed": self.algo.seed,
            },
            "logging": {
                "log_interval": self.logging.log_interval,
                "eval_interval": self.logging.eval_interval,
                "output_dir": self.logging.output_dir,
                "experiment_name": self.logging.experiment_name,
            }
        }


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_small_test_config() -> Config:
    """Get a small configuration for testing."""
    return Config(
        env=EnvironmentConfig(grid_x=3, grid_y=3, max_iter=20),
        algo=AlgorithmConfig(M=100, T=20, default_kappa_o=1, default_kappa_r=0),
        logging=LoggingConfig(log_interval=5, eval_interval=10)
    )


def get_asymmetric_config() -> Config:
    """Get configuration with asymmetric information structure."""
    env_config = EnvironmentConfig(grid_x=4, grid_y=4, max_iter=50)
    n_agents = env_config.n_agents
    
    # Create asymmetric kappa values
    # Agents at different positions have different observation/reward ranges
    kappa_o = []
    kappa_r = []
    for i in range(n_agents):
        row = i // env_config.grid_x
        col = i % env_config.grid_x
        # Center agents have larger ranges
        is_center = (1 <= row < env_config.grid_y - 1) and (1 <= col < env_config.grid_x - 1)
        kappa_o.append(2 if is_center else 1)
        kappa_r.append(1 if is_center else 0)
    
    return Config(
        env=env_config,
        algo=AlgorithmConfig(
            M=500,
            T=50,
            kappa_o=kappa_o,
            kappa_r=kappa_r
        ),
        logging=LoggingConfig(log_interval=10, eval_interval=25)
    )