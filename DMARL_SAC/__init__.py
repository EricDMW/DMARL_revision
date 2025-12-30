# ============================================================================
# File: __init__.py
# Description: Package initialization for SAC
# ============================================================================
"""
SAC (Soft Actor-Critic) for Wireless Communication Environment

This package implements SAC algorithm adapted for wireless communication 
environments with asymmetric information.

Modules:
    config: Configuration and hyperparameters
    buffer: Replay buffer for experience storage
    networks: Actor and Individual Q-networks
    utils: Neighbor extraction and topology utilities
    agent: Single SAC agent implementation
    sac: Multi-agent system manager
    trainer: Training loop and utilities
    evaluate: Evaluation and visualization tools
    main: Entry point

Example:
    from DMARL_SAC import Config, Trainer
    
    config = Config()
    config.grid_x = 5
    config.grid_y = 5
    config.max_episodes = 5000
    
    trainer = Trainer(config)
    results = trainer.train()
"""

from config import Config, default_config
from buffer import ReplayBuffer
from networks import ActorNetwork, IndividualQNetwork
from utils import NeighborExtractor, create_action_mask
from agent import SACAgent
from sac import SACSystem
from trainer import Trainer
from evaluate import Evaluator, plot_training_curves, compare_models

__version__ = '1.0.0'
__author__ = 'SAC Implementation'

__all__ = [
    # Configuration
    'Config',
    'default_config',
    
    # Core components
    'ReplayBuffer',
    'ActorNetwork',
    'IndividualQNetwork',
    'NeighborExtractor',
    'create_action_mask',
    'SACAgent',
    'SACSystem',
    
    # Training and evaluation
    'Trainer',
    'Evaluator',
    'plot_training_curves',
    'compare_models',
]
