# VDN (Value Decomposition Networks) Implementation

This directory contains an implementation of VDN (Value Decomposition Networks) for multi-agent reinforcement learning on wireless communication environments.

## Overview

VDN is a value-based multi-agent RL algorithm that decomposes the joint Q-function into a sum of individual Q-functions:

**Q_tot(s, a) = Σ_i Q_i(s_i, a_i)**

### Key Differences from DMARL

1. **Individual Q-Networks**: Each agent has its own Q-network that takes only its own observation and action (not neighbor information)
2. **Joint Q-Value**: The joint Q-value is computed as the sum of individual Q-values
3. **Decentralized Execution**: Each agent's policy depends only on its local observation
4. **Centralized Training**: Training uses the joint Q-value, but gradients flow to individual networks

## File Structure

```
DMARL_VDN/
├── __init__.py          # Package initialization
├── config.py            # Configuration and hyperparameters
├── buffer.py            # Experience replay buffer
├── networks.py          # Actor and Individual Q-network architectures
├── agent.py             # VDN agent implementation
├── vdn.py               # Multi-agent system manager
├── trainer.py           # Training loop and utilities
├── evaluate.py          # Evaluation and visualization
├── main.py              # Entry point
├── utils.py             # Neighbor extraction utilities
└── README.md            # This file
```

## Key Components

### Networks (`networks.py`)

- **ActorNetwork**: Decentralized policy network
  - Input: Local observation s_i
  - Output: Action probabilities π_i(a_i | s_i)

- **IndividualQNetwork**: Individual Q-value network for VDN
  - Input: Local observation s_i + one-hot action a_i
  - Output: Q-value Q_i(s_i, a_i)

### Agent (`agent.py`)

- **VDNAgent**: Single agent with:
  - Actor network for policy
  - Individual Q-network for value estimation
  - Target networks for stable learning
  - Update methods for actor and Q-network

### System Manager (`vdn.py`)

- **VDNSystem**: Coordinates multiple agents
  - Manages agent creation and initialization
  - Coordinates action selection
  - Handles experience storage
  - Orchestrates training updates

## Usage

### Training

```bash
cd DMARL_VDN
python main.py --mode train --max_episodes 5000
```

### Evaluation

```bash
python main.py --mode eval --model_path ./checkpoints/best_model
```

### Custom Parameters

```bash
python main.py --mode train \
    --grid_x 5 \
    --grid_y 5 \
    --n_obs_neighbors 1 \
    --max_episodes 5000 \
    --lr_actor 1e-4 \
    --lr_critic 3e-4
```

## Algorithm Details

### Training Update

1. **Q-Network Update**:
   - For each agent i:
     - Compute current Q: Q_i(s_i, a_i)
     - Compute target Q: r_i + γ * Q_i_target(s'_i, a'_i)
     - Update: L_i = (Q_i - target_Q_i)^2

2. **Actor Update**:
   - For each agent i:
     - Compute expected Q-value: E[Q_i(s_i, a_i)] where a_i ~ π_i(·|s_i)
     - Update policy to maximize expected Q-value

3. **Target Network Update**:
   - Soft update: θ^- ← τ * θ + (1 - τ) * θ^-

### Key Features

- **Decentralized Execution**: Each agent acts based only on its local observation
- **Individual Q-Networks**: No centralized critic, each agent learns its own Q-function
- **Joint Value Decomposition**: Q_tot = Σ_i Q_i enables centralized training
- **Experience Replay**: Off-policy learning with replay buffer
- **Target Networks**: Stable learning with soft target updates

## Configuration

See `config.py` for all hyperparameters. Key parameters:

- `lr_actor`: Actor learning rate (default: 1e-4)
- `lr_critic`: Q-network learning rate (default: 3e-4)
- `gamma`: Discount factor (default: 0.95)
- `tau`: Soft update coefficient (default: 0.005)
- `epsilon_start/end/decay`: Exploration schedule
- `buffer_size`: Replay buffer capacity (default: 100000)
- `batch_size`: Training batch size (default: 256)

## Comparison with DMARL

| Feature | DMARL | VDN |
|---------|-------|-----|
| Critic Input | Neighbor observations + actions | Own observation + action |
| Q-function | Centralized with neighbor info | Individual per agent |
| Joint Q-value | Not explicitly computed | Sum of individual Q-values |
| Information Sharing | Uses κ_o-hop neighbors | No explicit sharing |
| Complexity | Higher (neighbor extraction) | Lower (individual networks) |

## References

- Sunehag, P., et al. "Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward." AAMAS, 2018.
