# ============================================================================
# File: __init__.py
# Description: Package initialization for VDN
# ============================================================================
"""
VDN (Value Decomposition Networks) for Wireless Communication Environment

This package implements VDN algorithm adapted for wireless communication 
environments with asymmetric information.

Modules:
    config: Configuration and hyperparameters
    buffer: Replay buffer for experience storage
    networks: Actor and Individual Q-networks
    utils: Neighbor extraction and topology utilities
    agent: Single VDN agent implementation
    vdn: Multi-agent system manager
    trainer: Training loop and utilities
    evaluate: Evaluation and visualization tools
    main: Entry point

Example:
    from DMARL_VDN import Config, Trainer
    
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
from agent import VDNAgent
from vdn import VDNSystem
from trainer import Trainer
from evaluate import Evaluator, plot_training_curves, compare_models

__version__ = '1.0.0'
__author__ = 'VDN Implementation'

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
    'VDNAgent',
    'VDNSystem',
    
    # Training and evaluation
    'Trainer',
    'Evaluator',
    'plot_training_curves',
    'compare_models',
]
