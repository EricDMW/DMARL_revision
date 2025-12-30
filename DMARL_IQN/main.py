# ============================================================================
# File: main.py
# Description: Entry point for training and evaluation
# ============================================================================
"""
Main Entry Point for IQN on Wireless Communication

This script provides command-line interface for:
- Training new models
- Evaluating trained models
- Visualizing results

Usage:
    # Train a new model
    python main.py --mode train
    
    # Evaluate a trained model
    python main.py --mode eval --model_path ./checkpoints/best_model
    
    # Train with custom parameters
    python main.py --mode train --grid_x 6 --grid_y 6 --max_episodes 10000
"""

import argparse
import os
import sys
import numpy as np
import torch
import random

from config import Config
from trainer import Trainer
from evaluate import Evaluator, plot_training_curves


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='IQN for Wireless Communication Environment'
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'eval', 'both'],
        help='Running mode: train, eval, or both'
    )
    
    # Environment parameters
    parser.add_argument('--grid_x', type=int, default=5, help='Grid width')
    parser.add_argument('--grid_y', type=int, default=5, help='Grid height')
    parser.add_argument('--n_obs_neighbors', type=int, default=1, help='Observation neighborhood (κ_o)')
    parser.add_argument('--n_reward_neighbors', type=int, default=1, help='Reward neighborhood (κ_r)')
    parser.add_argument('--ddl', type=int, default=2, help='Deadline horizon')
    
    # Training parameters
    parser.add_argument('--max_episodes', type=int, default=5000, help='Maximum training episodes')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum steps per episode')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--buffer_size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=3e-4, help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient')
    
    # Exploration parameters
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.05, help='Final exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.9995, help='Exploration decay rate')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    # Logging and saving
    parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency')
    parser.add_argument('--save_freq', type=int, default=500, help='Checkpoint frequency')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    
    # Evaluation parameters
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model')
    parser.add_argument('--n_eval_episodes', type=int, default=20, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render during evaluation')
    parser.add_argument('--save_gif', action='store_true', help='Save evaluation as GIF')
    
    return parser.parse_args()


def create_config_from_args(args) -> Config:
    """Create config from command line arguments"""
    config = Config()
    
    # Environment
    config.grid_x = args.grid_x
    config.grid_y = args.grid_y
    config.n_obs_neighbors = args.n_obs_neighbors
    config.n_reward_neighbors = args.n_reward_neighbors
    config.ddl = args.ddl
    config.max_steps = args.max_steps
    
    # Training
    config.max_episodes = args.max_episodes
    config.batch_size = args.batch_size
    config.buffer_size = args.buffer_size
    config.lr_actor = args.lr_actor
    config.lr_critic = args.lr_critic
    config.gamma = args.gamma
    config.tau = args.tau
    
    # Exploration
    config.epsilon_start = args.epsilon_start
    config.epsilon_end = args.epsilon_end
    config.epsilon_decay = args.epsilon_decay
    
    # Random seed
    config.seed = args.seed
    
    # Logging
    config.log_freq = args.log_freq
    config.save_freq = args.save_freq
    config.save_dir = args.save_dir
    
    return config


def train_model(config: Config) -> dict:
    """Train a new model"""
    print("\n" + "=" * 60)
    print("TRAINING MODE")
    print("=" * 60)
    
    trainer = Trainer(config)
    
    try:
        results = trainer.train()
    finally:
        trainer.close()
    
    # Plot training curves
    plot_training_curves(
        episode_rewards=results['episode_rewards'],
        critic_losses=results['critic_losses'],
        actor_losses=results['actor_losses'],
        save_path=os.path.join(config.save_dir, 'training_curves.png')
    )
    
    return results


def evaluate_model(config: Config, model_path: str, args) -> dict:
    """Evaluate a trained model"""
    print("\n" + "=" * 60)
    print("EVALUATION MODE")
    print("=" * 60)
    
    if model_path is None:
        model_path = os.path.join(config.save_dir, 'best_model')
    
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist!")
        sys.exit(1)
    
    evaluator = Evaluator(config, model_path=model_path)
    
    try:
        results = evaluator.evaluate(
            n_episodes=args.n_eval_episodes,
            render=args.render,
            save_gif=args.save_gif,
            gif_path=os.path.join(config.save_dir, 'evaluation.gif')
        )
    finally:
        evaluator.close()
    
    return results


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """Main entry point"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    config = create_config_from_args(args)
    
    print("\n" + "=" * 60)
    print("IQN (Independent Q-Network) for Wireless Communication Environment")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Grid: {config.grid_x} x {config.grid_y}")
    print(f"Agents: {config.n_agents}")
    print(f"Device: {config.device}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print("=" * 60)
    
    if args.mode == 'train':
        train_model(config)
        
    elif args.mode == 'eval':
        evaluate_model(config, args.model_path, args)
        
    elif args.mode == 'both':
        # Train then evaluate
        train_results = train_model(config)
        
        print("\nProceeding to evaluation...")
        eval_results = evaluate_model(
            config,
            os.path.join(config.save_dir, 'best_model'),
            args
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()