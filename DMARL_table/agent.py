"""
Agent Module for NMARL Algorithm.

This module implements the agent with softmax policy as described in the paper.
Each agent maintains its own policy parameters θ_i.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
from collections import defaultdict


class SoftmaxPolicy:
    """
    Softmax policy for a single agent.
    
    Implements π_i(a_i|s_i, θ_i) = exp(θ_{i,s_i,a_i}) / Σ_{a'_i} exp(θ_{i,s_i,a'_i})
    as in equation (37) of the paper.
    """
    
    def __init__(self, state_dim: int, n_actions: int, 
                 temperature: float = 1.0, device: str = "cpu"):
        """
        Initialize the softmax policy.
        
        Args:
            state_dim: Dimension of the local state
            n_actions: Number of possible actions
            temperature: Softmax temperature (lower = more deterministic)
            device: Computation device
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.temperature = temperature
        self.device = device
        
        # Policy parameters: θ_{i,s,a} for each state-action pair
        # Using a dictionary for tabular representation
        self.theta: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions, dtype=np.float32)
        )
        
    def _state_key(self, state: np.ndarray) -> Tuple:
        """Convert state to hashable key."""
        return tuple(state.flatten().astype(int))
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for a given state.
        
        Args:
            state: Local state observation
            
        Returns:
            Array of action probabilities
        """
        key = self._state_key(state)
        logits = self.theta[key] / self.temperature
        
        # Numerically stable softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        
        return probs
    
    def sample_action(self, state: np.ndarray) -> int:
        """
        Sample an action from the policy.
        
        Args:
            state: Local state observation
            
        Returns:
            Sampled action index
        """
        probs = self.get_action_probs(state)
        action = np.random.choice(self.n_actions, p=probs)
        return action
    
    def get_log_prob(self, state: np.ndarray, action: int) -> float:
        """
        Get log probability of an action given state.
        
        Args:
            state: Local state
            action: Action index
            
        Returns:
            Log probability log π_i(a_i|s_i, θ_i)
        """
        probs = self.get_action_probs(state)
        return np.log(probs[action] + 1e-10)
    
    def get_log_prob_gradient(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute ∇_{θ_i} log π_i(a_i|s_i, θ_i).
        
        For softmax policy:
        ∂/∂θ_{s,a} log π(a|s) = 1_{action=a} - π(a|s)
        
        Args:
            state: Local state
            action: Action taken
            
        Returns:
            Gradient with respect to θ for this state
        """
        probs = self.get_action_probs(state)
        gradient = -probs.copy()
        gradient[action] += 1.0
        gradient /= self.temperature
        return gradient
    
    def update_theta(self, state: np.ndarray, gradient: np.ndarray, 
                    learning_rate: float):
        """
        Update policy parameters for a state.
        
        θ_i,m+1 = θ_i,m + η_m * gradient
        
        Args:
            state: State for which to update
            gradient: Gradient to apply
            learning_rate: Learning rate η_m
        """
        key = self._state_key(state)
        self.theta[key] += learning_rate * gradient
    
    def get_all_parameters(self) -> Dict[Tuple, np.ndarray]:
        """Get all policy parameters."""
        return dict(self.theta)
    
    def set_parameters(self, params: Dict[Tuple, np.ndarray]):
        """Set policy parameters."""
        self.theta = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32),
            params
        )


class Agent:
    """
    Complete agent implementation for NMARL.
    
    Each agent maintains:
    - Softmax policy π_i(a_i|s_i, θ_i)
    - Truncated neighbors-averaged Q-function estimate
    """
    
    def __init__(self, agent_id: int, state_dim: int, n_actions: int,
                 kappa_o: int, kappa_r: int, temperature: float = 1.0,
                 device: str = "cpu"):
        """
        Initialize the agent.
        
        Args:
            agent_id: Unique identifier for this agent
            state_dim: Dimension of local state observation
            n_actions: Number of possible actions
            kappa_o: State-action observation range
            kappa_r: Reward communication range
            temperature: Policy temperature
            device: Computation device
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.kappa_o = kappa_o
        self.kappa_r = kappa_r
        self.device = device
        
        # Initialize policy
        self.policy = SoftmaxPolicy(
            state_dim=state_dim,
            n_actions=n_actions,
            temperature=temperature,
            device=device
        )
        
        # Q-function estimates (tabular)
        # Q̂^{π_θ_m}_{i,T}(s_{N^{κ_o}_i}, a_{N^{κ_o}_i})
        self.q_estimates: Dict[Tuple, float] = defaultdict(float)
        
        # Accumulated gradient for actor update
        self.gradient_accumulator: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions, dtype=np.float32)
        )
        
    def reset_q_estimates(self):
        """Reset Q-function estimates for new episode."""
        self.q_estimates = defaultdict(float)
    
    def reset_gradient_accumulator(self):
        """Reset gradient accumulator."""
        self.gradient_accumulator = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32)
        )
    
    def select_action(self, local_state: np.ndarray) -> int:
        """
        Select action using current policy.
        
        Args:
            local_state: Local state observation
            
        Returns:
            Selected action
        """
        return self.policy.sample_action(local_state)
    
    def update_q_estimate(self, state_action_key: Tuple, 
                         truncated_reward: float, 
                         next_state_action_key: Tuple,
                         alpha: float, gamma: float):
        """
        Update Q-function estimate using TD update (equation 11).
        
        Q̂_{i,t} = (1-α)Q̂_{i,t-1} + α(r_truncated + γQ̂_{i,t-1}(s',a'))
        
        Args:
            state_action_key: Current (s_{N^κ_o}, a_{N^κ_o}) key
            truncated_reward: (1/N) Σ_{j∈N^κ_r} r_j
            next_state_action_key: Next state-action key
            alpha: Learning rate α_{t-1}
            gamma: Discount factor γ
        """
        old_q = self.q_estimates[state_action_key]
        next_q = self.q_estimates[next_state_action_key]
        
        td_target = truncated_reward + gamma * next_q
        new_q = (1 - alpha) * old_q + alpha * td_target
        
        self.q_estimates[state_action_key] = new_q
    
    def compute_policy_gradient_estimate(self, trajectory: List[Dict],
                                        gamma: float) -> Dict[Tuple, np.ndarray]:
        """
        Compute policy gradient estimate (equation 12).
        
        ĝ^{π_θ_m}_i = Σ_t γ^t Q̂(s,a) ∇_{θ_i} log π_i(a_i,t|s_i,t, θ_i,m)
        
        Args:
            trajectory: List of {state, action, state_action_key}
            gamma: Discount factor
            
        Returns:
            Gradient dictionary mapping states to gradient updates
        """
        gradient_updates = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32)
        )
        
        for t, step in enumerate(trajectory):
            local_state = step['local_state']
            action = step['action']
            state_action_key = step['state_action_key']
            
            # Get Q-estimate for this state-action
            q_value = self.q_estimates[state_action_key]
            
            # Compute log probability gradient
            log_prob_grad = self.policy.get_log_prob_gradient(local_state, action)
            
            # Accumulate gradient with discount
            state_key = self.policy._state_key(local_state)
            gradient_updates[state_key] += (gamma ** t) * q_value * log_prob_grad
        
        return gradient_updates
    
    def update_policy(self, gradient_updates: Dict[Tuple, np.ndarray],
                     learning_rate: float, max_grad_norm: float = 1.0):
        """
        Update policy parameters (equation 13).
        
        θ_{i,m+1} = θ_i,m + η_m * ĝ^{π_θ_m}_i
        
        Args:
            gradient_updates: Gradient dictionary from compute_policy_gradient_estimate
            learning_rate: η_m
            max_grad_norm: Maximum gradient norm for clipping
        """
        for state_key, gradient in gradient_updates.items():
            # Gradient clipping
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > max_grad_norm:
                gradient = gradient * (max_grad_norm / grad_norm)
            
            # Update parameters
            self.policy.theta[state_key] += learning_rate * gradient


class MultiAgentSystem:
    """
    Container for all agents in the NMARL system.
    """
    
    def __init__(self, n_agents: int, state_dims: List[int], 
                 n_actions: int, kappa_o: List[int], kappa_r: List[int],
                 temperature: float = 1.0, device: str = "cpu"):
        """
        Initialize the multi-agent system.
        
        Args:
            n_agents: Number of agents
            state_dims: State dimension for each agent
            n_actions: Number of actions (same for all agents)
            kappa_o: Observation range for each agent
            kappa_r: Reward range for each agent
            temperature: Policy temperature
            device: Computation device
        """
        self.n_agents = n_agents
        self.agents: List[Agent] = []
        
        for i in range(n_agents):
            agent = Agent(
                agent_id=i,
                state_dim=state_dims[i] if isinstance(state_dims, list) else state_dims,
                n_actions=n_actions,
                kappa_o=kappa_o[i],
                kappa_r=kappa_r[i],
                temperature=temperature,
                device=device
            )
            self.agents.append(agent)
    
    def __getitem__(self, idx: int) -> Agent:
        return self.agents[idx]
    
    def __iter__(self):
        return iter(self.agents)
    
    def reset_all_q_estimates(self):
        """Reset Q-estimates for all agents."""
        for agent in self.agents:
            agent.reset_q_estimates()
    
    def select_actions(self, local_states: List[np.ndarray]) -> List[int]:
        """
        Select actions for all agents.
        
        Args:
            local_states: List of local state observations
            
        Returns:
            List of selected actions
        """
        return [agent.select_action(state) 
                for agent, state in zip(self.agents, local_states)]
    
    def get_joint_action_probs(self, local_states: List[np.ndarray]) -> np.ndarray:
        """
        Get action probability distributions for all agents.
        
        Returns:
            Array of shape (n_agents, n_actions)
        """
        probs = []
        for agent, state in zip(self.agents, local_states):
            probs.append(agent.policy.get_action_probs(state))
        return np.array(probs)