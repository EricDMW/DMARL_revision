"""
NMARL Algorithm Implementation

A distributed and scalable algorithm for networked multi-agent 
reinforcement learning under asymmetric information.

Based on the paper:
"Distributed and Scalable Algorithm for Networked Multi-agent 
Reinforcement Learning under Asymmetric Information"
by Pengcheng Dai, Dongming Wang, He Wang, Wenwu Yu, and Wei Ren
"""

from .config import Config, EnvironmentConfig, AlgorithmConfig, LoggingConfig
from .config import get_default_config, get_small_test_config, get_asymmetric_config
from .network_utils import CommunicationNetwork, StateAggregator
from .agent import Agent, SoftmaxPolicy, MultiAgentSystem
from .q_function import TruncatedQFunction, QFunctionManager, PolicyGradientEstimator
from .algorithm import NMARLAlgorithm, IQLBaseline

__version__ = "1.0.0"
__author__ = "NMARL Implementation"

__all__ = [
    # Config
    'Config',
    'EnvironmentConfig', 
    'AlgorithmConfig',
    'LoggingConfig',
    'get_default_config',
    'get_small_test_config',
    'get_asymmetric_config',
    # Network utilities
    'CommunicationNetwork',
    'StateAggregator',
    # Agent
    'Agent',
    'SoftmaxPolicy',
    'MultiAgentSystem',
    # Q-function
    'TruncatedQFunction',
    'QFunctionManager',
    'PolicyGradientEstimator',
    # Algorithm
    'NMARLAlgorithm',
    'IQLBaseline',
]