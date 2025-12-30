# IQN (Independent Q-Network) Implementation

This directory contains an implementation of IQN (Independent Q-Network) for multi-agent reinforcement learning on wireless communication environments.

## Overview

IQN is a value-based multi-agent RL algorithm where each agent learns completely independently:

**Each agent learns: Q_i(s_i, a_i) independently**

### Key Characteristics

1. **Fully Independent Learning**: Each agent has its own Q-network that takes only its own observation and action
2. **No Joint Value**: No joint Q-value computation - each agent learns independently
3. **Decentralized Execution**: Each agent's policy depends only on its local observation
4. **Decentralized Training**: Each agent updates based only on its own experience

## File Structure

```
DMARL_IQN/
├── __init__.py          # Package initialization
├── config.py            # Configuration and hyperparameters
├── buffer.py            # Experience replay buffer
├── networks.py          # Actor and Individual Q-network architectures
├── agent.py             # IQN agent implementation
├── iqn.py               # Multi-agent system manager
├── trainer.py           # Training loop and utilities
├── evaluate.py          # Evaluation and visualization
├── main.py              # Entry point
├── utils.py             # Neighbor extraction utilities
├── run_multiple_seeds.sh # Multi-seed training script
├── plot_multiple_seeds.py # Multi-seed plotting script
├── plot_algorithm_comparison.py # Algorithm comparison script
└── README.md            # This file
```

## Key Components

### Networks (`networks.py`)

- **ActorNetwork**: Decentralized policy network
  - Input: Local observation s_i
  - Output: Action probabilities π_i(a_i | s_i)

- **IndividualQNetwork**: Individual Q-value network for IQN
  - Input: Local observation s_i + one-hot action a_i
  - Output: Q-value Q_i(s_i, a_i)

### Agent (`agent.py`)

- **IQNAgent**: Single agent with:
  - Actor network for policy
  - Individual Q-network for value estimation
  - Target networks for stable learning
  - Update methods for actor and Q-network

### System Manager (`iqn.py`)

- **IQNSystem**: Coordinates multiple agents
  - Manages agent creation and initialization
  - Coordinates action selection
  - Handles experience storage
  - Orchestrates independent training updates

## Usage

### Training

```bash
cd DMARL_IQN
python main.py --mode train --max_episodes 5000
```

### Evaluation

```bash
python main.py --mode eval --model_path ./checkpoints/best_model
```

### Multi-Seed Training

```bash
./run_multiple_seeds.sh
```

### Custom Parameters

```bash
python main.py --mode train \
    --grid_x 5 \
    --grid_y 5 \
    --n_obs_neighbors 1 \
    --max_episodes 5000 \
    --lr_actor 1e-4 \
    --lr_critic 3e-4 \
    --seed 42
```

## Algorithm Details

### Training Update

1. **Q-Network Update** (Independent):
   - For each agent i:
     - Compute current Q: Q_i(s_i, a_i)
     - Compute target Q: r_i + γ * Q_i_target(s'_i, a'_i)
     - Update: L_i = (Q_i - target_Q_i)^2
   - No joint value computation

2. **Actor Update** (Independent):
   - For each agent i:
     - Compute expected Q-value: E[Q_i(s_i, a_i)] where a_i ~ π_i(·|s_i)
     - Update policy to maximize expected Q-value

3. **Target Network Update**:
   - Soft update: θ^- ← τ * θ + (1 - τ) * θ^-

### Key Features

- **Fully Decentralized**: Each agent acts and learns based only on its local observation
- **Individual Q-Networks**: No centralized critic, each agent learns its own Q-function
- **No Coordination**: Agents learn completely independently
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

## Comparison with Other Algorithms

| Feature | DMARL | VDN | IQN |
|---------|-------|-----|-----|
| Critic Input | Neighbor observations + actions | Own observation + action | Own observation + action |
| Q-function | Centralized with neighbor info | Individual per agent | Individual per agent |
| Joint Q-value | Not computed | Sum: Q_tot = Σ Q_i | Not computed |
| Information Sharing | Uses κ_o-hop neighbors | No explicit sharing | No explicit sharing |
| Learning Coordination | Centralized training | Joint value decomposition | Fully independent |
| Complexity | Highest | Medium | Lowest |

## References

- Tan, M. "Multi-agent reinforcement learning: Independent vs. cooperative agents." ICML, 1993.
- Tampuu, A., et al. "Multiagent deep reinforcement learning with extremely sparse rewards." arXiv, 2017.
