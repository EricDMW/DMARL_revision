# ============================================================================
# File: __init__.py
# Description: Package initialization
# ============================================================================
"""
DMARL for Wireless Communication Environment

This package implements Multi-Agent Deep Deterministic Policy Gradient (DMARL)
adapted for wireless communication environments with asymmetric information.

Modules:
    config: Configuration and hyperparameters
    buffer: Replay buffer for experience storage
    networks: Actor and Critic neural networks
    utils: Neighbor extraction and topology utilities
    agent: Single DMARL agent implementation
    DMARL: Multi-agent system manager
    trainer: Training loop and utilities
    evaluate: Evaluation and visualization tools
    main: Entry point

Example:
    from DMARL_wireless import Config, Trainer
    
    config = Config()
    config.grid_x = 5
    config.grid_y = 5
    config.max_episodes = 5000
    
    trainer = Trainer(config)
    results = trainer.train()
"""

from config import Config, default_config
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork
from utils import NeighborExtractor, create_action_mask
from agent import DMARLAgent
from dmarl import DMARLSystem
from trainer import Trainer
from evaluate import Evaluator, plot_training_curves, compare_models

__version__ = '1.0.0'
__author__ = 'DMARL Implementation'

__all__ = [
    # Configuration
    'Config',
    'default_config',
    
    # Core components
    'ReplayBuffer',
    'ActorNetwork',
    'CriticNetwork',
    'NeighborExtractor',
    'create_action_mask',
    'DMARLAgent',
    'DMARLSystem',
    
    # Training and evaluation
    'Trainer',
    'Evaluator',
    'plot_training_curves',
    'compare_models',
]