# SAC (Soft Actor-Critic) Implementation

This directory contains an implementation of SAC (Soft Actor-Critic) for multi-agent reinforcement learning on wireless communication environments.

## Overview

SAC is an off-policy actor-critic algorithm that uses entropy regularization for exploration:

**Soft Q-Learning**: Q(s,a) = r + γ * E[Q(s',a') - α * log π(a'|s')]

**Actor Update**: Maximize E[Q(s,a) - α * log π(a|s)]

### Key Characteristics

1. **Soft Q-Learning**: Uses entropy-regularized Q-learning for better exploration
2. **Two Q-Networks**: Uses Q1 and Q2 networks, taking minimum for stability
3. **Entropy Regularization**: Temperature parameter α balances exploration and exploitation
4. **Automatic Temperature Tuning**: Optional automatic tuning of temperature parameter
5. **Decentralized Execution**: Each agent's policy depends only on its local observation

## File Structure

```
DMARL_SAC/
├── __init__.py          # Package initialization
├── config.py            # Configuration and hyperparameters
├── buffer.py            # Experience replay buffer
├── networks.py          # Actor and Individual Q-network architectures
├── agent.py             # SAC agent implementation
├── sac.py               # Multi-agent system manager
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

- **IndividualQNetwork**: Individual Q-value network
  - Input: Local observation s_i + one-hot action a_i
  - Output: Q-value Q_i(s_i, a_i)

### Agent (`agent.py`)

- **SACAgent**: Single agent with:
  - Actor network for policy
  - Two Q-networks (Q1, Q2) for value estimation
  - Target networks for stable learning
  - Temperature parameter α for entropy regularization
  - Optional automatic temperature tuning
  - Update methods for actor and Q-networks

### System Manager (`sac.py`)

- **SACSystem**: Coordinates multiple agents
  - Manages agent creation and initialization
  - Coordinates action selection
  - Handles experience storage
  - Orchestrates soft Q-learning updates

## Usage

### Training

```bash
cd DMARL_SAC
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
    --alpha 0.2 \
    --seed 42
```

## Algorithm Details

### Training Update

1. **Soft Q-Network Update**:
   - For each agent i:
     - Compute current Q: Q1_i(s_i, a_i) and Q2_i(s_i, a_i)
     - Compute target Q: r_i + γ * (min(Q1_i, Q2_i)(s'_i, a'_i) - α * log π_i(a'_i|s'_i))
     - Update: L = MSE(Q1_i, target) + MSE(Q2_i, target)
   - Uses minimum of two Q-networks for stability

2. **Actor Update**:
   - For each agent i:
     - Compute expected value: E[Q_i(s_i, a_i) - α * log π_i(a_i|s_i)] where a_i ~ π_i(·|s_i)
     - Update policy to maximize expected value (minimize negative expected value)

3. **Temperature Update** (if automatic tuning enabled):
   - Update α to maintain target entropy: minimize α * (log π(a|s) + target_entropy)

4. **Target Network Update**:
   - Soft update: θ^- ← τ * θ + (1 - τ) * θ^-

### Key Features

- **Entropy Regularization**: Encourages exploration through entropy bonus
- **Two Q-Networks**: Reduces overestimation bias by taking minimum
- **Soft Updates**: Stable learning with soft target network updates
- **Automatic Temperature**: Optional automatic tuning of entropy coefficient
- **Decentralized Execution**: Each agent acts based only on its local observation

## Configuration

See `config.py` for all hyperparameters. Key parameters:

- `lr_actor`: Actor learning rate (default: 1e-4)
- `lr_critic`: Q-network learning rate (default: 3e-4)
- `gamma`: Discount factor (default: 0.95)
- `tau`: Soft update coefficient (default: 0.005)
- `alpha`: Temperature parameter for entropy regularization (default: 0.2)
- `use_automatic_entropy_tuning`: Enable automatic temperature tuning (default: False)
- `target_entropy_coef`: Target entropy coefficient for auto-tuning (default: 0.98)
- `buffer_size`: Replay buffer capacity (default: 100000)
- `batch_size`: Training batch size (default: 256)

## Comparison with Other Algorithms

| Feature | DMARL | VDN | IQN | SAC |
|---------|-------|-----|-----|-----|
| Critic Input | Neighbor observations + actions | Own observation + action | Own observation + action | Own observation + action |
| Q-function | Centralized with neighbor info | Individual per agent | Individual per agent | Individual per agent (Q1, Q2) |
| Joint Q-value | Not computed | Sum: Q_tot = Σ Q_i | Not computed | Not computed |
| Exploration | Epsilon-greedy | Epsilon-greedy | Epsilon-greedy | Entropy regularization |
| Learning Method | Actor-Critic | Value decomposition | Independent Q-learning | Soft Q-learning |
| Complexity | Highest | Medium | Lowest | Medium-High |

## References

- Haarnoja, T., et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." ICML, 2018.
- Haarnoja, T., et al. "Soft Actor-Critic Algorithms and Applications." arXiv, 2019.
