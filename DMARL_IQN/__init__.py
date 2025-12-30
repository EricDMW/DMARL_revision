# ============================================================================
# File: __init__.py
# Description: Package initialization for IQN
# ============================================================================
"""
IQN (Independent Q-Network) for Wireless Communication Environment

This package implements IQN algorithm adapted for wireless communication 
environments with asymmetric information.

Modules:
    config: Configuration and hyperparameters
    buffer: Replay buffer for experience storage
    networks: Actor and Individual Q-networks
    utils: Neighbor extraction and topology utilities
    agent: Single IQN agent implementation
    iqn: Multi-agent system manager
    trainer: Training loop and utilities
    evaluate: Evaluation and visualization tools
    main: Entry point

Example:
    from DMARL_IQN import Config, Trainer
    
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
from agent import IQNAgent
from iqn import IQNSystem
from trainer import Trainer
from evaluate import Evaluator, plot_training_curves, compare_models

__version__ = '1.0.0'
__author__ = 'IQN Implementation'

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
    'IQNAgent',
    'IQNSystem',
    
    # Training and evaluation
    'Trainer',
    'Evaluator',
    'plot_training_curves',
    'compare_models',
]
