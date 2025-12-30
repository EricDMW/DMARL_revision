"""
Q-Function Module for NMARL Algorithm.

This module implements the truncated neighbors-averaged Q-function
as described in equation (8) of the paper.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

from network_utils import CommunicationNetwork, StateAggregator


class TruncatedQFunction:
    """
    Truncated Neighbors-Averaged Q-Function.
    
    Implements Q̃^{π_θ}_{tru,i}(s_{N^{κ_o}_i}, a_{N^{κ_o}_i}) from equation (8):
    
    Q̃_{tru,i} = E[1/N * Σ_{t=0}^∞ γ^t Σ_{j∈N^{κ_r}_i} r_j | s_{N^{κ_o}_i,0}, a_{N^{κ_o}_i,0}]
    
    This evaluates the value of state-action of agent i's κ_o-hop neighbors,
    defined as the cumulative rewards over its κ_r-hop neighbors.
    """
    
    def __init__(self, agent_id: int, kappa_o: int, kappa_r: int,
                 n_agents: int, gamma: float = 0.9):
        """
        Initialize the truncated Q-function.
        
        Args:
            agent_id: Agent index
            kappa_o: State-action observation range
            kappa_r: Reward communication range
            n_agents: Total number of agents
            gamma: Discount factor
        """
        self.agent_id = agent_id
        self.kappa_o = kappa_o
        self.kappa_r = kappa_r
        self.n_agents = n_agents
        self.gamma = gamma
        
        # Tabular Q-function estimates
        # Key: (local_state, local_action) tuple
        # Value: Q-estimate
        self.q_table: Dict[Tuple, float] = defaultdict(float)
        
        # Visit counts for potential UCB-style exploration
        self.visit_counts: Dict[Tuple, int] = defaultdict(int)
        
    def reset(self):
        """Reset Q-function estimates."""
        self.q_table = defaultdict(float)
        self.visit_counts = defaultdict(int)
    
    def get_q_value(self, state_action_key: Tuple) -> float:
        """
        Get Q-value estimate for a local state-action pair.
        
        Args:
            state_action_key: (local_state_tuple, local_action_tuple)
            
        Returns:
            Q-value estimate
        """
        return self.q_table[state_action_key]
    
    def update(self, state_action_key: Tuple, truncated_reward: float,
              next_state_action_key: Tuple, alpha: float):
        """
        Update Q-function using TD(0) update (equation 11).
        
        Q̂_{i,t}(s,a) = (1-α)Q̂_{i,t-1}(s,a) + α(r_tru + γQ̂_{i,t-1}(s',a'))
        
        This is only updated when (s_{N^κ_o}, a_{N^κ_o}) = (s_{N^κ_o,t-1}, a_{N^κ_o,t-1})
        
        Args:
            state_action_key: Current state-action key
            truncated_reward: (1/N) * Σ_{j∈N^κ_r} r_j
            next_state_action_key: Next state-action key
            alpha: Learning rate α_{t-1}
        """
        old_q = self.q_table[state_action_key]
        next_q = self.q_table[next_state_action_key]
        
        # TD target: truncated_reward + γ * Q(s', a')
        td_target = truncated_reward + self.gamma * next_q
        
        # TD update
        new_q = (1 - alpha) * old_q + alpha * td_target
        
        self.q_table[state_action_key] = new_q
        self.visit_counts[state_action_key] += 1
    
    def get_all_estimates(self) -> Dict[Tuple, float]:
        """Get all Q-function estimates."""
        return dict(self.q_table)


class QFunctionManager:
    """
    Manages Q-function estimation for all agents.
    
    Handles the distributed critic step of Algorithm 1.
    """
    
    def __init__(self, network: CommunicationNetwork, 
                 state_aggregator: StateAggregator,
                 kappa_o: List[int], kappa_r: List[int],
                 gamma: float = 0.9):
        """
        Initialize the Q-function manager.
        
        Args:
            network: Communication network
            state_aggregator: State aggregation utilities
            kappa_o: Observation ranges for all agents
            kappa_r: Reward ranges for all agents
            gamma: Discount factor
        """
        self.network = network
        self.state_aggregator = state_aggregator
        self.kappa_o = kappa_o
        self.kappa_r = kappa_r
        self.gamma = gamma
        self.n_agents = network.n_agents
        
        # Create Q-function for each agent
        self.q_functions: List[TruncatedQFunction] = []
        for i in range(self.n_agents):
            q_func = TruncatedQFunction(
                agent_id=i,
                kappa_o=kappa_o[i],
                kappa_r=kappa_r[i],
                n_agents=self.n_agents,
                gamma=gamma
            )
            self.q_functions.append(q_func)
    
    def reset_all(self):
        """Reset all Q-function estimates."""
        for q_func in self.q_functions:
            q_func.reset()
    
    def compute_learning_rate(self, t: int, h: float, t0: float) -> float:
        """
        Compute the learning rate α_{t-1} for critic step.
        
        α_{t-1} = h / (t-1 + t0)
        
        Args:
            t: Current time step
            h: Learning rate parameter
            t0: Learning rate offset parameter
            
        Returns:
            Learning rate α_{t-1}
        """
        return h / (t + t0)
    
    def create_state_action_key(self, global_state: np.ndarray,
                                global_action: np.ndarray,
                                agent_id: int) -> Tuple:
        """
        Create state-action key for agent's local neighborhood.
        
        Args:
            global_state: Global state (all agents)
            global_action: Global action (all agents)
            agent_id: Agent index
            
        Returns:
            Hashable key for (s_{N^κ_o}, a_{N^κ_o})
        """
        return self.state_aggregator.get_local_state_action_key(
            global_state, global_action, agent_id
        )
    
    def compute_truncated_reward(self, global_rewards: np.ndarray,
                                agent_id: int) -> float:
        """
        Compute truncated reward for an agent.
        
        r_tru = (1/N) * Σ_{j∈N^κ_r_i} r_j
        
        Args:
            global_rewards: Rewards for all agents
            agent_id: Agent index
            
        Returns:
            Truncated reward
        """
        return self.state_aggregator.compute_truncated_reward(
            global_rewards, agent_id
        )
    
    def update_all_q_functions(self, 
                               global_state: np.ndarray,
                               global_action: np.ndarray,
                               global_rewards: np.ndarray,
                               next_global_state: np.ndarray,
                               next_global_action: np.ndarray,
                               t: int, h: float, t0: float):
        """
        Update Q-function estimates for all agents (distributed critic step).
        
        Implements line 8 of Algorithm 1.
        
        Args:
            global_state: Current global state
            global_action: Current global action
            global_rewards: Rewards received
            next_global_state: Next global state
            next_global_action: Next global action
            t: Current time step
            h, t0: Learning rate parameters
        """
        alpha = self.compute_learning_rate(t, h, t0)
        
        for i in range(self.n_agents):
            # Create state-action keys
            current_key = self.create_state_action_key(
                global_state, global_action, i
            )
            next_key = self.create_state_action_key(
                next_global_state, next_global_action, i
            )
            
            # Compute truncated reward
            truncated_reward = self.compute_truncated_reward(global_rewards, i)
            
            # Update Q-function
            self.q_functions[i].update(
                state_action_key=current_key,
                truncated_reward=truncated_reward,
                next_state_action_key=next_key,
                alpha=alpha
            )
    
    def get_q_value(self, agent_id: int, state_action_key: Tuple) -> float:
        """Get Q-value for an agent's local state-action pair."""
        return self.q_functions[agent_id].get_q_value(state_action_key)
    
    def get_statistics(self) -> Dict:
        """Get statistics about Q-function estimates."""
        stats = {
            'n_states_visited': [],
            'mean_q_values': [],
            'max_q_values': [],
            'min_q_values': []
        }
        
        for i, q_func in enumerate(self.q_functions):
            q_values = list(q_func.q_table.values())
            stats['n_states_visited'].append(len(q_func.q_table))
            if q_values:
                stats['mean_q_values'].append(np.mean(q_values))
                stats['max_q_values'].append(np.max(q_values))
                stats['min_q_values'].append(np.min(q_values))
            else:
                stats['mean_q_values'].append(0.0)
                stats['max_q_values'].append(0.0)
                stats['min_q_values'].append(0.0)
        
        return stats


class PolicyGradientEstimator:
    """
    Estimates the approximated policy gradient (equation 10 and 12).
    
    g^{π_θ}_{app,i} = (1/(1-γ)) E_{s,a}[Q̃_{tru,i}(s_{N^κ_o}, a_{N^κ_o}) 
                                         ∇_{θ_i} log π_i(a_i|s_i, θ_i)]
    """
    
    def __init__(self, gamma: float = 0.9):
        """
        Initialize the policy gradient estimator.
        
        Args:
            gamma: Discount factor
        """
        self.gamma = gamma
    
    def estimate_gradient(self, trajectory: List[Dict],
                         q_function: TruncatedQFunction,
                         policy) -> Dict[Tuple, np.ndarray]:
        """
        Estimate policy gradient from a trajectory.
        
        ĝ^{π_θ_m}_i = Σ_{t=0}^T γ^t Q̂(s_{N^κ_o,t}, a_{N^κ_o,t}) 
                       ∇_{θ_i} log π_i(a_{i,t}|s_{i,t}, θ_{i,m})
        
        Args:
            trajectory: List of transition dictionaries containing:
                - 'local_state': Local state observation
                - 'action': Action taken
                - 'state_action_key': Key for Q-function lookup
            q_function: Agent's Q-function
            policy: Agent's policy
            
        Returns:
            Dictionary mapping state keys to gradient updates
        """
        gradient_updates = defaultdict(
            lambda: np.zeros(policy.n_actions, dtype=np.float32)
        )
        
        for t, step in enumerate(trajectory):
            local_state = step['local_state']
            action = step['action']
            state_action_key = step['state_action_key']
            
            # Get Q-estimate
            q_value = q_function.get_q_value(state_action_key)
            
            # Compute log probability gradient
            log_prob_grad = policy.get_log_prob_gradient(local_state, action)
            
            # Accumulate with discount
            state_key = policy._state_key(local_state)
            gradient_updates[state_key] += (self.gamma ** t) * q_value * log_prob_grad
        
        return gradient_updates
    
    def estimate_all_gradients(self, trajectories: List[List[Dict]],
                              q_functions: List[TruncatedQFunction],
                              policies: List) -> List[Dict[Tuple, np.ndarray]]:
        """
        Estimate policy gradients for all agents.
        
        Args:
            trajectories: List of trajectories for each agent
            q_functions: List of Q-functions
            policies: List of policies
            
        Returns:
            List of gradient updates for each agent
        """
        all_gradients = []
        
        for trajectory, q_func, policy in zip(trajectories, q_functions, policies):
            gradient = self.estimate_gradient(trajectory, q_func, policy)
            all_gradients.append(gradient)
        
        return all_gradients