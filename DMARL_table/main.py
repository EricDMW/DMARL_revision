"""
Main Entry Point for NMARL Algorithm.

This script demonstrates how to use the NMARL algorithm implementation
with the Wireless Communication Environment.
"""

import os
import sys
import argparse
import numpy as np
import time
from typing import Dict, Optional
from tqdm import tqdm

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, EnvironmentConfig, AlgorithmConfig, LoggingConfig
from config import get_default_config, get_small_test_config, get_asymmetric_config
from algorithm import NMARLAlgorithm, IQLBaseline
from plotting import (
    plot_training_rewards, 
    plot_comparison, 
    plot_comprehensive_summary,
    create_all_plots,
    plot_kappa_analysis
)


def create_mock_environment(config: Config):
    """
    Create a mock environment for testing when the real env is not available.
    
    This simulates the basic interface of WirelessCommEnv.
    """
    class MockWirelessEnv:
        def __init__(self, grid_x, grid_y, ddl=2, n_obs_neighbors=1, 
                     max_iter=50, **kwargs):
            self.grid_x = grid_x
            self.grid_y = grid_y
            self.n_agents = grid_x * grid_y
            self.ddl = ddl
            self.n_obs_neighbors = n_obs_neighbors
            self.max_iter = max_iter
            self.n_actions = 5
            self.current_step = 0
            
            # State dimension
            self.obs_dim = ddl * ((2 * n_obs_neighbors + 1) ** 2)
            
            # Internal state
            self.agent_states = None
            
        def reset(self, seed=None):
            self.current_step = 0
            # Initialize random states for each agent
            self.agent_states = np.random.randint(0, 2, size=(self.n_agents, self.ddl))
            
            # Create observation
            obs = self._get_observation()
            info = self._get_info()
            
            return obs, info
        
        def _get_observation(self):
            """Get flattened observation."""
            obs_list = []
            for i in range(self.n_agents):
                # Create local observation with neighborhood
                local_obs = np.zeros(self.obs_dim)
                # Fill with agent's state
                local_obs[:self.ddl] = self.agent_states[i]
                obs_list.append(local_obs)
            return np.concatenate(obs_list)
        
        def _get_info(self):
            """Get info dict with local observations and rewards."""
            return {
                'local_obs': [self.agent_states[i].astype(np.float32) 
                             for i in range(self.n_agents)],
                'local_rewards': np.zeros(self.n_agents)
            }
        
        def step(self, actions):
            self.current_step += 1
            
            # Simple dynamics: packet transmission simulation
            rewards = np.zeros(self.n_agents)
            
            for i in range(self.n_agents):
                action = actions[i] if isinstance(actions, np.ndarray) else actions
                
                # Action 0: idle
                # Actions 1-4: transmit to access point
                if action > 0:
                    # Transmission attempt
                    if np.any(self.agent_states[i] > 0):
                        # Has packet to send
                        if np.random.random() < 0.8:  # Success probability
                            rewards[i] = 1.0
                            # Clear oldest packet
                            for d in range(self.ddl):
                                if self.agent_states[i, d] > 0:
                                    self.agent_states[i, d] = 0
                                    break
                        else:
                            rewards[i] = -0.1  # Collision/failure
                
                # Packet arrival
                if np.random.random() < 0.8:  # Arrival probability
                    # Add new packet at latest deadline
                    if self.agent_states[i, -1] == 0:
                        self.agent_states[i, -1] = 1
                
                # Deadline shift
                if self.agent_states[i, 0] > 0:
                    rewards[i] -= 0.5  # Penalty for expired packet
                
                self.agent_states[i] = np.roll(self.agent_states[i], -1)
                self.agent_states[i, -1] = 0
            
            total_reward = rewards.sum()
            terminated = False
            truncated = self.current_step >= self.max_iter
            
            obs = self._get_observation()
            info = {
                'local_obs': [self.agent_states[i].astype(np.float32) 
                             for i in range(self.n_agents)],
                'local_rewards': rewards
            }
            
            return obs, total_reward, terminated, truncated, info
        
        def close(self):
            pass
        
        @property
        def action_space(self):
            class ActionSpace:
                def __init__(self, n):
                    self.n = n
                def sample(self):
                    return np.random.randint(0, self.n)
            return ActionSpace(self.n_actions)
    
    return MockWirelessEnv(
        grid_x=config.env.grid_x,
        grid_y=config.env.grid_y,
        ddl=config.env.ddl,
        n_obs_neighbors=config.env.n_obs_neighbors,
        max_iter=config.env.max_iter
    )


def try_import_env():
    """Try to import the actual environment."""
    try:
        import env_lib
        return env_lib.WirelessCommEnv
    except ImportError:
        print("Warning: env_lib not found, using mock environment")
        return None


def create_environment(config: Config, use_mock: bool = False):
    """Create the environment based on configuration."""
    if use_mock:
        return create_mock_environment(config)
    
    EnvClass = try_import_env()
    if EnvClass is None:
        return create_mock_environment(config)
    
    return EnvClass(
        grid_x=config.env.grid_x,
        grid_y=config.env.grid_y,
        ddl=config.env.ddl,
        packet_arrival_probability=config.env.packet_arrival_probability,
        success_transmission_probability=config.env.success_transmission_probability,
        n_obs_neighbors=config.env.n_obs_neighbors,
        max_iter=config.env.max_iter,
        render_mode=config.env.render_mode,
        device=config.env.device
    )


def run_experiment(config: Config, use_mock: bool = False,
                  run_baseline: bool = True, save_results: bool = True,
                  output_dir: str = None) -> Dict:
    """
    Run a complete experiment with NMARL and optionally IQL baseline.
    
    Args:
        config: Configuration object
        use_mock: Whether to use mock environment
        run_baseline: Whether to run IQL baseline for comparison
        save_results: Whether to save results to JSON
        output_dir: Directory to save results (if None, uses config.logging.output_dir)
        
    Returns:
        Dictionary containing results
    """
    import json
    
    results = {}
    output_dir = output_dir or config.logging.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    env = create_environment(config, use_mock)
    
    print("=" * 60)
    print("NMARL Algorithm Training")
    print("=" * 60)
    print(f"Grid size: {config.env.grid_x}x{config.env.grid_y}")
    print(f"Number of agents: {config.env.n_agents}")
    print(f"Episodes: {config.algo.M}, Steps per episode: {config.algo.T}")
    print(f"κ_o (observation range): {config.algo.kappa_o}")
    print(f"κ_r (reward range): {config.algo.kappa_r}")
    if hasattr(config.algo, 'seed') and config.algo.seed is not None:
        print(f"Seed: {config.algo.seed}")
    print("=" * 60)
    
    # Train NMARL
    nmarl = NMARLAlgorithm(config, env)
    nmarl_stats = nmarl.train()
    results['nmarl'] = nmarl_stats
    
    # Evaluate NMARL
    print("\nEvaluating NMARL...")
    eval_results = nmarl.evaluate(n_episodes=20)
    print(f"NMARL Evaluation: Mean={eval_results['mean_reward']:.4f}, "
          f"Std={eval_results['std_reward']:.4f}")
    results['nmarl_eval'] = eval_results
    
    # Run IQL baseline if requested
    if run_baseline:
        print("\n" + "=" * 60)
        print("IQL Baseline Training")
        print("=" * 60)
        
        # Reset environment
        env = create_environment(config, use_mock)
        
        iql = IQLBaseline(config, env)
        iql_stats = iql.train(M=config.algo.M, T=config.algo.T)
        results['iql'] = iql_stats
    
    env.close()
    
    # Save results to JSON
    if save_results:
        json_results = {
            'nmarl': {
                'episode_rewards': [float(r) for r in results['nmarl']['episode_rewards']],
                'episode_lengths': [int(l) for l in results['nmarl']['episode_lengths']],
                'gradient_norms': [float(g) for g in results['nmarl'].get('gradient_norms', [])],
            },
            'nmarl_eval': {
                'mean_reward': float(eval_results['mean_reward']),
                'std_reward': float(eval_results['std_reward']),
            },
            'config': {
                'grid_x': config.env.grid_x,
                'grid_y': config.env.grid_y,
                'ddl': config.env.ddl,
                'n_obs_neighbors': config.env.n_obs_neighbors,
                'M': config.algo.M,
                'T': config.algo.T,
                'gamma': config.algo.gamma,
                'eta': config.algo.eta,
                'kappa_o': config.algo.kappa_o,
                'kappa_r': config.algo.kappa_r,
                'seed': getattr(config.algo, 'seed', None)
            }
        }
        
        if 'iql' in results:
            json_results['iql'] = {
                'episode_rewards': [float(r) for r in results['iql']['episode_rewards']],
                'episode_lengths': [int(l) for l in results['iql']['episode_lengths']],
            }
        
        results_file = os.path.join(output_dir, 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {results_file}")
    
    return results


def run_kappa_sweep(base_config: Config, kappa_pairs: list,
                   use_mock: bool = False) -> Dict:
    """
    Run experiments with different (κ_o, κ_r) configurations.
    
    Args:
        base_config: Base configuration
        kappa_pairs: List of (kappa_o, kappa_r) tuples
        use_mock: Whether to use mock environment
        
    Returns:
        Dictionary mapping kappa pairs to results
    """
    results = {}
    
    # Create progress bar for kappa sweep
    pbar = tqdm(kappa_pairs, desc="Kappa Sweep", unit="config",
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for kappa_o, kappa_r in pbar:
        pbar.set_description(f"Kappa Sweep (κ_o={kappa_o}, κ_r={kappa_r})")
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Running with κ_o={kappa_o}, κ_r={kappa_r}")
        tqdm.write("=" * 60)
        
        # Create config with specific kappa values
        config = Config(
            env=base_config.env,
            algo=AlgorithmConfig(
                M=base_config.algo.M,
                T=base_config.algo.T,
                gamma=base_config.algo.gamma,
                eta=base_config.algo.eta,
                h=base_config.algo.h,
                t0=base_config.algo.t0,
                kappa_o=[kappa_o] * base_config.env.n_agents,
                kappa_r=[kappa_r] * base_config.env.n_agents,
                temperature=base_config.algo.temperature,
                seed=base_config.algo.seed
            ),
            logging=base_config.logging
        )
        
        env = create_environment(config, use_mock)
        nmarl = NMARLAlgorithm(config, env)
        stats = nmarl.train()
        
        results[(kappa_o, kappa_r)] = stats
        env.close()
    
    pbar.close()
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NMARL Algorithm Training")
    parser.add_argument('--grid-x', type=int, default=3, help='Grid x dimension')
    parser.add_argument('--grid-y', type=int, default=3, help='Grid y dimension')
    parser.add_argument('--episodes', type=int, default=20000, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=30, help='Steps per episode')
    parser.add_argument('--kappa-o', type=int, default=1, help='Observation range')
    parser.add_argument('--kappa-r', type=int, default=0, help='Reward range')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--eta', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--no-baseline', action='store_true', help='Skip IQL baseline')
    parser.add_argument('--mock', action='store_true', help='Use mock environment')
    parser.add_argument('--sweep', action='store_true', help='Run kappa sweep experiment')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    seed = args.seed if args.seed is not None else 42
    np.random.seed(seed)
    
    if args.sweep:
        # Run kappa sweep experiment
        base_config = Config(
            env=EnvironmentConfig(grid_x=args.grid_x, grid_y=args.grid_y,
                                 max_iter=args.steps),
            algo=AlgorithmConfig(M=args.episodes, T=args.steps,
                                gamma=args.gamma, eta=args.eta),
            logging=LoggingConfig(log_interval=20, output_dir=args.output_dir)
        )
        
        kappa_pairs = [
            (0, 0),  # Local only
            (1, 0),  # 1-hop observation, local reward
            (1, 1),  # 1-hop for both
            (2, 1),  # 2-hop observation, 1-hop reward
        ]
        
        sweep_results = run_kappa_sweep(base_config, kappa_pairs, use_mock=args.mock)
        
        # Plot kappa analysis
        plot_kappa_analysis(sweep_results, 
                          save_path=os.path.join(args.output_dir, 'kappa_analysis.png'))
        
    else:
        # Run single experiment
        config = Config(
            env=EnvironmentConfig(
                grid_x=args.grid_x,
                grid_y=args.grid_y,
                max_iter=args.steps
            ),
            algo=AlgorithmConfig(
                M=args.episodes,
                T=args.steps,
                gamma=args.gamma,
                eta=args.eta,
                default_kappa_o=args.kappa_o,
                default_kappa_r=args.kappa_r,
                seed=seed
            ),
            logging=LoggingConfig(
                log_interval=max(1, args.episodes // 20),
                output_dir=args.output_dir
            )
        )
        
        # Set seed again after config creation
        np.random.seed(seed)
        
        results = run_experiment(config, use_mock=args.mock,
                                run_baseline=not args.no_baseline,
                                output_dir=args.output_dir)
        
        # Create plots
        print("\nGenerating plots...")
        
        # NMARL plots
        create_all_plots(results['nmarl'], args.output_dir, prefix='nmarl')
        
        # Comparison plot if baseline was run
        if 'iql' in results:
            plot_comparison(
                {'NMARL (Ours)': results['nmarl'], 'IQL': results['iql']},
                save_path=os.path.join(args.output_dir, 'algorithm_comparison.png'),
                title='NMARL vs IQL Comparison'
            )
        
        print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()